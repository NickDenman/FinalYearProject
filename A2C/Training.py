import copy
import random
import torch
import A2C.Runner as Runner
import A2C.ACNetwork as ac
from environment import ZeroTwentyPCBBoard as zt_pcb
import torch.nn.functional as F
from multiprocessing.spawn import freeze_support
import os
from baselines.common.vec_env import DummyVecEnv
from baselines.common.vec_env import ShmemVecEnv
from baselines.common.vec_env import SubprocVecEnv
from baselines import logger
import numpy as np
import matplotlib.pyplot as plt
import statistics


num_envs = 32
num_agents = 1
num_episodes = 10000

gamma = 0.98
lambda_ = 0.92
entropy_coeff = 0.01
value_coeff = 0.25
grad_norm_limit = 20
epochs = 4
clip = 0.2


def create_env(index):
    env = zt_pcb.ZeroTwentyPCBBoard("envs/v_small/v_small_" + str(index % 8) + ".txt", padded=True, num_agents=num_agents)
    return env


def get_env_attributes():
    env = create_env(0)
    steps = env.rows * env.cols * 2
    return env.get_observation_size(), env.get_action_size(), steps


def setup():
    ob_size, action_size, num_steps = get_env_attributes()

    actor = ac.Actor(ob_size, [64], action_size)  # 4x4 works best at [64]
    prev_actor = copy.deepcopy(actor)
    critic = ac.Critic(ob_size, [64], 1)

    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    return num_steps, actor, prev_actor, adam_actor, critic, adam_critic


def train(num_steps, actor, adam_actor, critic, adam_critic):
    total_rewards = []
    avg_reward = 0.0

    for episode in range(num_episodes):
        steps = []

        #  agent_observations[ENV_NUM][AGENT_NUM]
        agent_observations = venv.reset()
        episode_rewards = np.zeros(shape=num_envs, dtype=np.float32)
        for _ in range(num_steps):
            obs = torch.from_numpy(agent_observations[:, 0])  # TODO: update for multiple agents
            logits = actor(obs)

            action_probs = F.softmax(logits, dim=-1)
            action_dists = torch.distributions.Categorical(action_probs)
            actions = action_dists.sample()
            log_prob = action_dists.log_prob(actions)
            values = critic(obs)

            action_list = actions.cpu().tolist()
            action_dict = [{0: i} for i in action_list]

            agent_observations, rewards, dones, _ = venv.step(action_dict)

            episode_rewards += rewards

            masks = (1.0 - torch.FloatTensor(dones)).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            actions = actions.unsqueeze(1)

            steps.append((obs, actions, log_prob, rewards, masks, values))

        final_obs = torch.from_numpy(agent_observations[:, 0])
        final_values = critic(final_obs)

        steps.append((None, None, None, None, None, final_values))
        rollout = process_rollout(steps)

        learn(adam_actor, adam_critic, rollout)

        total_rewards.append(episode_rewards.mean())
        avg_reward += total_rewards[-1]
        if episode % (num_episodes // 100) == 0:
            avg_reward /= (num_episodes // 100)
            _, _, _, values, returns = rollout
            values = values.detach().numpy().flatten()
            returns = returns.detach().numpy().flatten()

            ev = explained_variance(returns, values)
            print((episode * 100) // num_episodes, "% :: ", ev, " :: ", avg_reward, sep="")
            avg_reward = 0.0

    return total_rewards


def explained_variance(y_true, y_pred):
    diff_var = statistics.variance((y_true - y_pred).tolist())
    var = statistics.variance(y_true.tolist())

    return 1 - (diff_var / var)


def process_rollout(steps):
    out_size = len(steps) - 1
    out = [None] * out_size

    advantages = torch.zeros(num_envs, 1)
    _, _, _, _, _, last_values = steps[-1]

    for t in reversed(range(out_size)):
        obs, actions, log_prob, rewards, masks, values = steps[t]
        _, _, _, _, _, next_values = steps[t + 1]

        deltas = rewards + next_values.data * gamma * masks - values.data
        advantages = advantages * gamma * lambda_ * masks + deltas

        returns = advantages + values.data

        out[t] = obs, actions, log_prob, values, returns

    return tuple(map(lambda x: torch.cat(x, 0), zip(*out)))


def learn(adam_actor, adam_critic, rollout):
    prev_actor.load_state_dict(actor.state_dict())
    total_steps = num_steps * num_envs
    for k in range(epochs):
        idx = [i for i in range(total_steps)]
        random.shuffle(idx)

        batch_obs, batch_actions, batch_log_probs, batch_values, batch_returns = (arr[idx] for arr in rollout)

        new_logits = actor(batch_obs.detach())
        new_action_probs = F.softmax(new_logits, dim=-1)
        new_log_dist = torch.distributions.Categorical(new_action_probs)
        new_log_prob = new_log_dist.log_prob(batch_actions.detach())
        entropy = new_log_dist.entropy()
        new_values = critic(batch_obs.detach())

        r_t = torch.exp(new_log_prob - batch_log_probs.detach())
        advantage = batch_returns - new_values.detach()
        surr1 = r_t * advantage
        surr2 = torch.clamp(r_t, min=(1 - clip), max=(1 + clip)) * advantage

        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = value_coeff * ((1 / 2) * (new_values - batch_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        policy_loss = action_loss + entropy_coeff * entropy_loss

        policy_loss.backward()
        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_norm_limit)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_norm_limit)

        adam_actor.step()
        adam_critic.step()

    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # Normalise the advantages


def route_board(actor):
    env = create_env(0)
    obs = env.reset()
    done = False
    steps = 0
    reward = 0

    while not done and steps < num_steps:
        t_obs = torch.from_numpy(obs[0])
        policies = actor(t_obs)
        a_probs = F.softmax(policies, dim=-1)
        action = torch.distributions.Categorical(a_probs).sample().item()

        obs, r, done, _ = env.step({0: action})

        reward += r
        steps += 1

    env.render_board()
    print(reward)


if __name__ == '__main__':
    freeze_support()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    envs = [lambda: create_env(i) for i in range(num_envs)]
    venv = DummyVecEnv(envs)
    num_steps, actor, prev_actor, adam_actor, critic, adam_critic = setup()
    rewards = train(num_steps, actor, adam_actor, critic, adam_critic)

    plt.plot(rewards)
    plt.show()
