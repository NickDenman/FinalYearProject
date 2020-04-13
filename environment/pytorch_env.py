from functools import reduce

import gym
import numpy as np
import torch
from baselines.common.vec_env import ShmemVecEnv, VecNormalize, VecEnvWrapper, DummyVecEnv
from gym import ObservationWrapper
from gym.wrappers import TimeLimit


def create_env(env, min_rows, min_cols, max_rows, max_cols, seed, idx, conv):
    env = env(min_rows, min_cols, max_rows, max_cols, padded=True)
    # env = env(5, 5, rand_nets=False, filename="../main_project/envs/small/5x5_" + str(idx % 8) + ".txt", padded=True)
    # env = env(5, 5, rand_nets=False, filename="../main_project/envs/small/5x5_5.txt", padded=True)

    if conv:
        env = ConvEnvWrapper(env)
    else:
        env = LinearEnvWrapper(env)

    return lambda: env


def make_vec_env(env, min_rows, min_cols, max_rows, max_cols, seed, num_envs, normalise=False, conv=False):
    envs = [create_env(env, min_rows, min_cols, max_rows, max_cols, seed, i, conv) for i in range(num_envs)]
    envs = ShmemVecEnv(envs, context='fork')

    if normalise:
        envs = VecNormalize(envs, ret=False)

    if conv:
        envs = VecPyTorchEnv(envs)
    else:
        envs = VecPyTorchEnvLinear(envs)

    return envs


class VecPyTorchEnv(VecEnvWrapper):
    def __init__(self, venv):
        super(VecPyTorchEnv, self).__init__(venv)

    def reset(self):
        obs = self.venv.reset()
        grid_obs, dest_obs = obs[0], obs[1]
        grid_obs = torch.from_numpy(grid_obs).float()
        dest_obs = torch.from_numpy(dest_obs).float()
        return grid_obs, dest_obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(-1)
        actions = actions.cpu().tolist()
        # action_dict = [{i: a for i, a in enumerate(a_list)} for a_list in actions]
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        grid_obs = torch.from_numpy(obs[0]).float()
        dest_obs = torch.from_numpy(obs[1]).float()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return (grid_obs, dest_obs), reward, done, info


class LinearEnvWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        x = reduce(lambda x, y: x*y, self.observation_space.spaces[0].shape) + self.observation_space.spaces[1].shape[0]
        self.observation_space = gym.spaces.Box(0, 1, shape=(x, ), dtype=np.float32)

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
        # obs = np.concatenate((obs[0].reshape(self.venv.num_envs, -1), obs[1]), axis=1)
        obs = torch.from_numpy(obs).float()
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(-1)
        actions = actions.cpu().tolist()
        # action_dict = [{i: a for i, a in enumerate(a_list)} for a_list in actions]
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info