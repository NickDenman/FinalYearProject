import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape):
        self.obs_shape = obs_shape
        self.action_shape = 1
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.values = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

    def save_obs(self, obs, idx):
        self.obs[idx] = obs

    def get_obs(self, index):
        return self.obs[index]

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.values[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step += 1

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.values[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.values[step + 1] * \
                        self.masks[step + 1] - self.values[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.values[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = \
                    self.returns[step + 1] * gamma * self.masks[step + 1] + \
                    self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        batch_size = self.num_processes * self.num_steps
        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            values_batch = self.values[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            advantages_batch = advantages.view(-1, 1)[indices]

            yield obs_batch, \
                  actions_batch, \
                  values_batch, \
                  return_batch, \
                  old_action_log_probs_batch, \
                  advantages_batch
