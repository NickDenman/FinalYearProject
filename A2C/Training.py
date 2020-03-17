import random
import torch
import A2C.Runner as Runner
import A2C.ACNetwork as ac
from environment import ZeroTwentyPCBBoard as zt_pcb
import torch.nn.functional as F
import baselines.common.vec_env.subproc_vec_env as SubprocVecEnv
from multiprocessing.spawn import freeze_support
import os
from baselines.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt


num_envs = 8
num_agents = 1
num_episodes = 100000

gamma = 0.98
lambda_ = 0.92
entropy_coeff = 0.01
value_coeff = 0.25
grad_norm_limit = 20


def create_env(index):
    env = zt_pcb.ZeroTwentyPCBBoard("envs/v_small/v_small_" + str(index) + ".txt", padded=True, num_agents=num_agents)
    return env


def get_env_attributes():
    env = create_env(0)
    steps = env.rows * env.cols * 2
    return env.get_observation_size(), env.get_action_size(), steps


def setup():
    ob_size, action_size, num_steps = get_env_attributes()

    actor = ac.Actor(ob_size, [64], action_size)
    critic = ac.Critic(ob_size, [64], 1)

    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    return num_steps, actor, adam_actor, critic, adam_critic


def train(num_steps, actor, adam_actor, critic, adam_critic):
    total_rewards = []

    for episode in range(num_episodes):
        steps = []

        #  agent_observations[ENV_NUM][AGENT_NUM]
        agent_observations = venv.reset()
        episode_rewards = np.zeros(shape=num_envs, dtype=np.float32)
        for _ in range(num_steps):
            obs = torch.from_numpy(agent_observations[:, 0])  # TODO: update for multiple agents
            logits = actor(obs)

            action_probs = F.softmax(logits, dim=-1)
            actions = torch.distributions.Categorical(action_probs).sample()  # NOTE: these are different format to other project you're following `.unsqueeze(1)` brings them to the same format.
            values = critic(obs)

            action_list = actions.cpu().tolist()
            action_dict = [{0: i} for i in action_list]

            agent_observations, rewards, dones, _ = venv.step(action_dict)

            episode_rewards += rewards

            masks = (1.0 - torch.FloatTensor(dones)).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            actions = actions.unsqueeze(1)

            steps.append((rewards, masks, actions, logits, values))

        final_obs = torch.from_numpy(agent_observations[:, 0])
        final_values = critic(final_obs)

        steps.append((None, None, None, None, final_values))
        rollout = process_rollout(steps)

        learn(adam_actor, adam_critic, rollout)

        total_rewards.append(episode_rewards.mean())
        if episode % (num_episodes // 100) == 0:
            print(episode, " - ", (episode * 100) // num_episodes, "%", sep="")

    return total_rewards


def process_rollout(steps):
    out_size = len(steps) - 1
    out = [None] * out_size

    advantages = torch.zeros(num_envs, 1)
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data

    for t in reversed(range(out_size)):
        rewards, masks, actions, logits, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * gamma * masks

        deltas = rewards + next_values.data * gamma * masks - values.data
        advantages = advantages * gamma * lambda_ * masks + deltas

        out[t] = actions, logits, values, returns, advantages

    return map(lambda x: torch.cat(x, 0), zip(*out))


def learn(adam_actor, adam_critic, rollout):
    actions, logits, values, returns, advantages = rollout

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    log_action_probs = log_probs.gather(1, torch.autograd.Variable(actions))

    action_loss = (-log_action_probs * torch.autograd.Variable(advantages)).sum()  # TODO: could look at .mean()
    value_loss = value_coeff * ((1 / 2) * (values - torch.autograd.Variable(returns)) ** 2).sum()
    entropy_loss = (log_probs * probs).sum()
    policy_loss = action_loss + entropy_loss * entropy_coeff

    policy_loss.backward()
    value_loss.backward()

    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_norm_limit)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_norm_limit)

    adam_actor.step()
    adam_critic.step()


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
    num_steps, actor, adam_actor, critic, adam_critic = setup()
    rewards = train(num_steps, actor, adam_actor, critic, adam_critic)

    plt.plot(rewards)
    plt.show()
