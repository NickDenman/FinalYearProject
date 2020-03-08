import numpy as np
import torch
import gym
import matplotlib as plt


def t(x):
    return torch.from_numpy(x).float()


class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, n_actions):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_layer_1)
        self.layer2 = torch.nn.Linear(hidden_layer_1, hidden_layer_2)
        self.layer3 = torch.nn.Linear(hidden_layer_2, n_actions)

    def forward(self, s):
        a_probs = torch.nn.functional.relu(self.layer1(s))
        a_probs = torch.nn.functional.relu(self.layer2(a_probs))
        a_probs = torch.nn.functional.softmax(self.layer3(a_probs), dim=0)

        return a_probs


class Critic(torch.nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, n_actions):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_layer_1)
        self.layer2 = torch.nn.Linear(hidden_layer_1, hidden_layer_2)
        self.layer3 = torch.nn.Linear(hidden_layer_2, n_actions)

    def forward(self, s):
        v = torch.nn.functional.relu(self.layer1(s))
        v = torch.nn.functional.relu(self.layer2(v))
        v = self.layer3(v)

        return v


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


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_layer_1 = 128
hidden_layer_2 = 256
actor = Actor(state_dim, hidden_layer_1, hidden_layer_2, n_actions)
critic = Critic(state_dim, hidden_layer_1, hidden_layer_2, 1)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99
memory = Memory()
max_steps = 200


def learn(memory, q_val):
    values = torch.stack(memory.values)
    q_values = np.zeros((len(memory), 1))
    for i, (_, _, reward, mask) in enumerate(memory.reversed()):
        q_val = reward + gamma * (q_val * mask)
        q_values[len(memory) - 1 - i] = q_val

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
        state = env.reset()
        steps = 0

        while not done and steps < steps_per_episode:
            torch_state = t(state)
            action_probs = actor(torch_state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            value = critic(torch_state)

            s_prime, reward, done, _ = env.step(action.detach().data.numpy())

            episode_reward += reward
            memory.add(action_dist.log_prob(action), value, reward, done)

            steps += 1
            state = s_prime

        final_value = 0.0
        if not done:
            final_value = critic(t(state)).detach().data.numpy()
        learn(memory, final_value)
        memory.clear()

        rewards.append(episode_reward)
        if episode % 10 == 0:
            print(str(episode) + ": " + str(episode_reward))


train(episodes=500, steps_per_episode=max_steps)
