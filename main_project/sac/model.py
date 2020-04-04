import numpy as np
import torch.nn as nn

from main_project.utils import init


class Policy(nn.Module):
    def __init__(self, obs_shape, action_size, hidden_size):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(obs_shape[0], hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, action_size)), nn.Softmax)

    def forward(self):
        pass

    def evaluate(self):
        pass


class QNetwork(nn.Module):
    def __init__(self, obs_shape, action_size, hidden_size):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(obs_shape[0], hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, action_size)))

    def forward(self):
        pass

    def evaluate(self):
        pass
