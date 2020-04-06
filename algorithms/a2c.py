import torch.nn as nn
from torch import optim


class A2C:
    def __init__(self, network, value_coef, entropy_coef, num_mini_batch, lr=None, eps=None, max_grad_norm=None):
        self.network = network
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_mini_batch = num_mini_batch
        self.lr = lr
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.optimiser = optim.Adam(network.parameters(), lr=self.lr, eps=self.eps)

    def update(self, rollouts):
        obs_shape = rollouts.obs_shape
        action_shape = rollouts.action_shape
        num_steps = rollouts.num_steps
        num_processes = rollouts.num_processes

        values, action_log_probs, dist_entropy = self.network.evaluate(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimiser.zero_grad()
        (value_loss * self.value_coef + action_loss - dist_entropy * self.entropy_coef).backward()

        self.optimiser.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
