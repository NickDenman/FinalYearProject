import torch
import torch.nn as nn
import numpy as np

from main_project.utils import init, Flatten


class ACNetwork(nn.Module):
    def __init__(self, obs_shape, action_size, hidden_sizes):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # TODO: consider first conv2 with stride 1 and a max pooling layer
        self.conv = nn.Sequential(
            init_(nn.Conv2d(1, 8, 3, stride=3, padding_mode='zeros')), nn.ReLU(),
            init_(nn.Conv2d(8, 16, 3, stride=1, padding_mode='zeros')), nn.ReLU(),
            nn.MaxPool2d(3, 1, 0), Flatten())

        # out_size = 8 * ((((obs_shape[0].shape[-2] - 2) * (obs_shape[0].shape[-1] - 2)) * 4 + 2) // 3) + 2
        out_size = 402
        layers = []
        layers.append(init_(nn.Linear(out_size, hidden_sizes[0])))
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(init_(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
            layers.append(nn.ReLU())

        self.actor = nn.Sequential(*layers, init_(nn.Linear(hidden_sizes[-1], action_size)))
        self.critic = nn.Sequential(*layers, init_(nn.Linear(hidden_sizes[-1], 1)))

        self.dist = FixedCategorical

        self.train()

    def forward(self, obs):
        grid, dest = obs
        conv_out = self.conv(grid)
        inputs = torch.cat((conv_out, dest), dim=1)

        action_logits = self.actor(inputs)
        dist = self.dist(logits=action_logits)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        value = self.critic(inputs)

        return value, action, action_log_probs

    def act(self, obs):
        grid, dest = obs
        conv_out = self.conv(grid)
        inputs = torch.cat((conv_out, dest), dim=1)

        action_logits = self.actor(inputs)
        dist = self.dist(logits=action_logits)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return action, action_log_probs

    def get_value(self, obs):
        grid, dest = obs
        conv_out = self.conv(grid)
        inputs = torch.cat((conv_out, dest), dim=1)

        return self.critic(inputs)

    def evaluate(self, obs, action):
        grid, dest = obs
        conv_out = self.conv(grid)
        inputs = torch.cat((conv_out, dest), dim=1)

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

