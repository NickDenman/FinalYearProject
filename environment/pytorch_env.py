from functools import reduce

import gym
import numpy as np
import torch
from stable_baselines.common.vec_env import \
    VecNormalize, VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from gym import ObservationWrapper


def create_env(env,
               min_rows,
               min_cols,
               max_rows,
               max_cols,
               min_nets,
               max_nets,
               conv):
    env = env(min_rows,
              min_cols,
              max_rows,
              max_cols,
              min_nets=min_nets,
              max_nets=max_nets,
              padded=True)

    if conv:
        env = ConvEnvWrapper(env)
    else:
        env = LinearEnvWrapper(env)

    return lambda: env


def create_cp_env(seed, rank):
    env = gym.make("CartPole-v0")
    if seed is not None:
        env.seed(seed + rank)

    return lambda: env

def make_cart_pole_vec_env(seed, num_envs):
    envs = [create_cp_env(seed, i) for i in range(num_envs)]
    if num_envs == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)
    envs = VecPyTorchEnvLinear(envs)

    return envs


def make_vec_env(env,
                 device,
                 min_rows,
                 min_cols,
                 max_rows,
                 max_cols,
                 min_nets,
                 max_nets,
                 seed,
                 num_envs,
                 normalise=False,
                 conv=False):
    envs = [create_env(env,
                       min_rows,
                       min_cols,
                       max_rows,
                       max_cols,
                       min_nets,
                       max_nets,
                       conv) for _ in range(num_envs)]
    if num_envs == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    if normalise:
        envs = VecNormalize(envs)

    if conv:
        envs = VecPyTorchEnv(envs, device)
    else:
        envs = VecPyTorchEnvLinear(envs)

    return envs


class VecPyTorchEnv(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorchEnv, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        grid_obs, dest_obs = obs[0], obs[1]
        grid_obs = torch.from_numpy(grid_obs).float().to(self.device)
        dest_obs = torch.from_numpy(dest_obs).float().to(self.device)
        return grid_obs, dest_obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(-1)
        actions = actions.cpu().tolist()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        grid_obs = torch.from_numpy(obs[0]).float().to(self.device)
        dest_obs = torch.from_numpy(obs[1]).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return (grid_obs, dest_obs), reward, done, info


class LinearEnvWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        x = reduce(lambda x, y: x*y, self.observation_space.spaces[0].shape) + \
            self.observation_space.spaces[1].shape[0]
        self.observation_space = gym.spaces.Box(0, 1,
                                                shape=(x, ),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.concatenate((observation[0].flatten(), observation[1]))


class ConvEnvWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        grid_obs, dest_obs = observation
        grid_obs = np.expand_dims(grid_obs, axis=0)

        return grid_obs, dest_obs


class VecPyTorchEnvLinear(VecEnvWrapper):
    def __init__(self, venv):
        super(VecPyTorchEnvLinear, self).__init__(venv)

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float()
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(-1)
        actions = actions.cpu().tolist()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info