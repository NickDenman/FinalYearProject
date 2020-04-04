import torch
import torch.nn as nn
import numpy as np

from main_project.utils import init


class ACNetwork(nn.Module):
    def __init__(self, obs_shape, action_size, hidden_size):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(obs_shape[0], hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, action_size)))

        self.critic = nn.Sequential(
            init_(nn.Linear(obs_shape[0], hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.dist = FixedCategorical

        self.train()

    def forward(self, inputs):
        action_logits = self.actor(inputs)
        dist = self.dist(logits=action_logits)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        value = self.critic(inputs)

        return value, action, action_log_probs

    def act(self, inputs):
        action_logits = self.actor(inputs)
        dist = self.dist(logits=action_logits)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return action, action_log_probs

    def get_value(self, inputs):
        return self.critic(inputs)

    def evaluate(self, inputs, action):
        value = self.critic(inputs)
        action_logits = self.actor(inputs)
        dist = self.dist(logits=action_logits)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

