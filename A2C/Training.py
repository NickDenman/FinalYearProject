import random
import torch
import A2C.Runner as Runner
import A2C.ACNetwork as ac
from environment import ZeroTwentyPCBBoard as zt_pcb
import torch.nn.functional as F


num_envs = 8
num_agents = 1
num_episodes = 10000

gamma = 0.98
lambda_ = 0.92
entropy_coeff = 0.01
value_coeff = 0.25
grad_norm_limit = 20


def create_env(index):
    env = zt_pcb.ZeroTwentyPCBBoard("envs/v_small/v_small_" + str(index) + ".txt", padded=True)
    return env


def get_env_attributes():
    env = create_env(0)
    steps = env.rows * env.cols * 2
    return env.get_observation_size(), env.get_action_size(), steps


def setup():
    ob_size, action_size, num_steps = get_env_attributes()
    runners = [Runner.Runner(i, num_agents, num_steps, gamma, lambda_) for i in range(num_envs)]

    actor = ac.Actor(ob_size, [64], action_size)
    critic = ac.Critic(ob_size, [64], 1)

    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    return num_steps, runners, actor, adam_actor, critic, adam_critic


def train(runners, actor, adam_actor, critic, adam_critic):
    rewards = []

    for episode in range(num_episodes):
        actions = []
        logits = []
        values = []
        returns = []

        for i in range(num_envs):
            runner = runners[i]
            env_id = random.randrange(0, num_envs)
            env = create_env(env_id)

            actions_i, logits_i, values_i, returns_i = runner.run(actor, critic, env)

            actions.extend(actions_i)
            logits.extend(logits_i)
            values.extend(values_i)
            returns.extend(returns_i)
        # TODO: Maybe need to torch.stack on the 4 lists?
        logits = torch.stack(logits)
        learn(adam_actor, adam_critic, actions, logits, values, returns)


    return rewards


def learn(adam_actor, adam_critic, actions, logits, values, returns):
    probs = F.softmax(logits)
    log_probs = F.log_softmax(logits)
    log_action_probs = log_probs.gather(1, actions)

    action_loss = (-log_action_probs * torch.autograd.Variable(actions)).sum()  # TODO: could look at .mean()
    value_loss = value_coeff * ((1 / 2) * (values - torch.autograd.Variable(returns)) ** 2).sum()
    entropy_loss = (log_probs * probs).sum()

    policy_loss = action_loss + entropy_loss * entropy_coeff
    policy_loss.backward()
    value_loss.backward()

    torch.nn.utils.clip_grad_norm(actor.parameters(), grad_norm_limit)
    torch.nn.utils.clip_grad_norm(critic.parameters(), grad_norm_limit)

    adam_actor.step()
    adam_critic.step()


num_steps, runners, actor, adam_actor, critic, adam_critic = setup()
rewards = train(runners, actor, adam_actor, critic, adam_critic)
