import torch
import torch.nn.functional as F


def t(x):
    return torch.from_numpy(x).float()


class Runner:
    def __init__(self, idx, n_agents, n_steps, gamma, lambda_):
        self.idx = idx
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_ = lambda_

    def run(self, actor, critic, env):
        done = False
        episode_reward = 0.0
        env.reset()
        steps = 0

        actions = []
        logits = []
        values = []
        rewards = []
        masks = []

        # TODO: might need to reset the env if done to maintain n_steps per runner
        while not done and steps < self.n_steps:
            observation = env.observe(0)
            torch_observation = t(observation)
            action_logits = actor(torch_observation)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.distributions.Categorical(action_probs).sample()
            value = critic(torch_observation)

            reward, done = env.step({0: action.detach().item()})

            actions.append(action)
            logits.append(action_logits)
            values.append(value)
            rewards.append(torch.FloatTensor([reward]))
            masks.append(1.0 - done)

            episode_reward += reward
            steps += 1

        final_value = 0.0
        if not done:
            observation = env.observe(0)
            torch_state = t(observation)
            final_value = critic(torch_state).detach().data.numpy()
        else:
            print("huzzah")

        returns = self.calculate_returns_gae(rewards, values, masks, final_value)

        return actions, logits, values, returns

    def calculate_returns_gae(self, rewards, values, masks, last_value):
        gae = 0
        returns = []
        for i, (reward, value, mask) in enumerate(zip(reversed(rewards), reversed(values), reversed(masks))):
            delta = reward + self.gamma * last_value * mask - value.data  # TD Error
            gae = delta + self.gamma * self.lambda_ * mask * gae
            returns.append(gae + value)

            last_value = value
        returns.reverse()

        return returns
