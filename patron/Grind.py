import numpy as np
import torch
import gym
import environment.PCB_Board_2D_Static as PCB
import patron.AC as ac
import patron.AC_1 as ac1


def t(x):
    return torch.from_numpy(x).float()


def observe_env_normalised(agent_id):
    grid, pos, dest = env.observe(agent_id)
    grid = grid.flatten() / 10.0

    agent_info = np.zeros(5)
    agent_info[0] = pos.row / env.observation_rows
    agent_info[1] = pos.col / env.observation_cols
    agent_info[2] = dest.row / env.rows
    agent_info[3] = dest.col / env.cols
    agent_info[4] = agent_id / env.num_agents

    return np.concatenate((grid, agent_info))


def observe_env(agent_id):
    grid, pos, dest = env.observe(agent_id)
    grid = grid.flatten()

    agent_info = np.zeros(5)
    agent_info[0] = pos.row
    agent_info[1] = pos.col
    agent_info[2] = dest.row
    agent_info[3] = dest.col
    agent_info[4] = agent_id

    return np.concatenate((grid, agent_info))


class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(1 - done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.masks)

    def __len__(self):
        return len(self.rewards)

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data


env = PCB.PCBBoard("test_env.txt")
state_dim = env.observation_rows * env.observation_cols + 2 + 2 + 1  # grid size, agent_loc, agent_dest, agent_id
n_actions = len(env.actions)
hidden_layer_1 = 256
hidden_layer_2 = 128
actor = ac.Actor(state_dim, 256, 128, n_actions)
critic = ac.Critic(state_dim, 64, 32, 1)

actor_1 = ac1.Actor(state_dim, 16, n_actions)
critic_1 = ac1.Critic(state_dim, 8, 1)

adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.99
memory = Memory()
max_steps = 40


def learn(memory, q_val):
    values = torch.stack(memory.values)
    q_values = np.zeros((len(memory), 1))
    for i, (_, _, reward, mask) in enumerate(memory.reversed()):
        q_val = reward + gamma * (q_val * mask)
        q_values[len(memory) - 1 - i] = q_val  # something is wrong...

    advantage = torch.from_numpy(q_values).float() - values

    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()

    actor_loss = (-torch.stack(memory.log_probs) * advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()


def train(episodes, steps_per_episode):
    rewards = []

    for episode in range(episodes):
        done = False
        episode_reward = 0.0
        env.reset_env()
        steps = 0

        while not done and steps < steps_per_episode:
            observation = observe_env_normalised(0)
            torch_state = t(observation)
            action_probs = actor(torch_state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            value = critic(torch_state)

            reward, done = env.joint_act({0: action.detach().item()})

            episode_reward += reward
            memory.add(action_dist.log_prob(action), value, reward, done)

            steps += 1

        final_value = 0.0
        if not done:
            observation = observe_env_normalised(0)
            torch_state = t(observation)
            final_value = critic(torch_state).detach().data.numpy()
        learn(memory, final_value)
        memory.clear()

        rewards.append(episode_reward)

        if episode % 100 == 0:
            print(str(episode) + ": " + str(episode_reward))

    return rewards


r = train(episodes=10000, steps_per_episode=max_steps)
print("is it working???")
