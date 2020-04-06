import gym
import torch
from baselines.common.vec_env import ShmemVecEnv, VecNormalize, VecEnvWrapper
from gym.wrappers import TimeLimit


def create_env(env, rows, cols, obs_rows, obs_cols, min_nets, max_nets, seed):
    env = env(rows, cols, obs_rows, obs_cols, min_nets=min_nets, max_nets=max_nets, padded=True)
    return lambda: env


def make_vec_env(env, rows, cols, obs_rows, obs_cols, min_nets, max_nets, seed, num_envs, normalise=False):
    envs = [create_env(env, rows, cols, obs_rows, obs_cols, min_nets, max_nets, seed) for _ in range(num_envs)]
    envs = ShmemVecEnv(envs, context='fork')

    if normalise:
        envs = VecNormalize(envs, ret=False)

    return VecPyTorchEnv(envs)


def make_gym_env(env_id, num_envs, device, normalise=False, time_limit=False):
    envs = [create_gym_env(env_id, time_limit) for _ in range(num_envs)]
    envs = ShmemVecEnv(envs, context='fork')

    if normalise:
        envs = VecNormalize(envs, ret=False)

    return VecPyTorchEnv(envs, device)


def create_gym_env(env_id, time_limit):
    env = gym.make(env_id)
    if time_limit:
        env = TimeLimit(env)

    return lambda: env


# TODO: BIG TODO! Check all the squeezes etc
class VecPyTorchEnv(VecEnvWrapper):
    def __init__(self, venv):
        """Return only every `skip`-th frame"""
        super(VecPyTorchEnv, self).__init__(venv)
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
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