import numpy as np
import torch
import environment.ZeroTwentyPCBBoard as zt_pcb
import NN.AC as ac


def t(x):
    return torch.from_numpy(x).float()


class Memory:
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
    avg_reward = 0
    print_count = 100

    for episode in range(episodes):
        done = False
        episode_reward = 0.0
        env.reset()
        steps = 0

        while not done and steps < steps_per_episode:
            observation = env.observe(0)
            torch_state = t(observation)
            action_probs = actor(torch_state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            value = critic(torch_state)

            reward, done = env.step({0: action.detach().item()})

            episode_reward += reward
            memory.add(action_dist.log_prob(action), value, reward, done)

            steps += 1

        final_value = 0.0
        if not done:
            observation = env.observe(0)
            torch_state = t(observation)
            final_value = critic(torch_state).detach().data.numpy()
        # else:
        #     print("huzzah")
        learn(memory, final_value)
        memory.clear()

        avg_reward += episode_reward
        rewards.append(episode_reward)

        if episode % print_count == 0:
            print(str(episode) + ": " + str(avg_reward / print_count))
            avg_reward = 0.0

    return rewards

env = zt_pcb.ZeroTwentyPCBBoard("env/small.txt", padded=True)
state_dim = env.get_observation_size()
n_actions = env.get_action_size()

hidden_layers = [256, 128, 256]
actor = ac.Actor(state_dim, hidden_layers, n_actions)
critic = ac.Critic(state_dim, hidden_layers, 1)

adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.99
memory = Memory()
max_steps = 60

r = train(episodes=10000, steps_per_episode=max_steps)
env.render_board()
print("is it working???")

