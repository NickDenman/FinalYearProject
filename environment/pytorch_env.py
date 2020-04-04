import gym
import torch
from baselines.common.vec_env import ShmemVecEnv, VecNormalize, VecEnvWrapper
from gym.wrappers import TimeLimit

from root_file import get_full_path


def create_env(env, rank, num_agents):
    env_path = "envs/small/5x5_" + str(rank % 8) + ".txt"
    env = env(env_path, padded=True, num_agents=num_agents)
    return lambda: env


def make_vec_env(env, seed, num_envs, num_agents, device, normalise=False):
    envs = [create_env(env, i, num_agents) for i in range(num_envs)]
    envs = ShmemVecEnv(envs, context='fork')

    if normalise:
        envs = VecNormalize(envs, ret=False)

    return VecPyTorchEnv(envs, device)


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
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorchEnv, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
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
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info