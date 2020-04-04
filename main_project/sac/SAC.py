import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
from main_project.sac.model import QNetwork, Policy


class SAC:
    def __init__(self, obs_shape, action_size, hidden_size, value_coef, learn_temp=True, alpha=None, lr=None, eps=None, max_grad_norm=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.value_coef = value_coef
        self.learn_temp = learn_temp
        self.lr = lr
        self.eps = eps
        self.max_grad_norm = max_grad_norm

        self.q_net_1 = QNetwork(obs_shape, action_size, hidden_size)
        self.q_net_2 = QNetwork(obs_shape, action_size, hidden_size)

        self.target_q_net_1 = QNetwork(obs_shape, action_size, hidden_size)
        self.target_q_net_2 = QNetwork(obs_shape, action_size, hidden_size)

        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())

        self.q_optimiser_1 = optim.Adam(self.q_net_1.parameters(), lr=self.lr, eps=self.eps)
        self.q_optimiser_2 = optim.Adam(self.q_net_2.parameters(), lr=self.lr, eps=self.eps)

        self.actor = Policy(obs_shape, action_size, hidden_size)
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.eps)

        if self.learn_temp:
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr, eps=self.eps)
        else:
            self.alpha = alpha

    # TODO: pretty sure the max action is only needed during eval so separate function
    def evaluate_actions(self, obs):
        action_probs = self.actor(obs)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return action, action_probs, log_action_probs

    def act(self, obs):
        action_probs = self.actor(obs)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        return action

    def __actor_loss(self, obs_batch):
        action, action_probs, log_action_probs = self.evaluate_actions(obs_batch)
        q_1 = self.q_net_1(obs_batch)
        q_2 = self.q_net_2(obs_batch)
        min_q = torch.min(q_1, q_2)
        policy_loss = (action_probs * (self.alpha * log_action_probs - min_q)).mean()

        return policy_loss, log_action_probs

    def __critic_loss(self, obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch):
        with torch.no_grad():
            next_state_action, next_action_probs, next_action_log_probs, _ = self.evaluate_actions(next_obs_batch)
            q1_next_target = self.target_q_net_1(next_obs_batch)
            q2_next_target = self.target_q_net_2(next_obs_batch)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            q_next_target = next_action_probs * (min_q_next_target - self.alpha * next_action_log_probs)
            next_q_value = reward_batch + mask_batch * self.value_coef * q_next_target

        q1 = self.q_net_1(obs_batch).gather(1, action_batch.long())
        q2 = self.q_net_2(obs_batch).gather(1, action_batch.long())

        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)

        return q1_loss, q2_loss

    def __alpha_loss(self, log_action_probs):
        if self.learn_temp:
            alpha_loss = -(self.log_alpha * (log_action_probs + self.target_entropy).detach()).mean()
        else:
            alpha_loss = None

        return alpha_loss
