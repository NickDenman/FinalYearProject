import math
import os
import numpy as np
import torch
from gym import ObservationWrapper, ActionWrapper

from environment.OrdinalEnv import OrdinalEnv
from environment.pytorch_env import make_vec_env
from main_project.AStar import astar
from main_project.args import get_args


def load_network(args):
    load_path = os.path.join(args.save_dir, "final", "ppo9", "ppo",
                             "network.pt")
    ac = torch.load(load_path)[0]

    return ac


class ObsTensorise(ObservationWrapper):
    def observation(self, observation):
        return torch.Tensor(observation[0]).unsqueeze(0).unsqueeze(0), torch.Tensor(observation[1]).unsqueeze(0)


class ActionTensorise(ActionWrapper):

    def action(self, action):
        return action.item()

    def reverse_action(self, action):
        pass


def route_board(env, network):
    obs = env.reset()
    done = False
    steps = 0
    reward = 0

    while not done and steps < 450:
        with torch.no_grad():
            action, _ = network.act(obs)

        obs, r, done, info = env.step(action)
        if done and info['completed']:
            pass
        reward += r
        steps += 1

    return reward


def route_unseen(network):
    envs = [(15, 15),
            (15, 11),
            (10, 14),
            (13, 15),
            (10, 10),
            (15, 12),
            (15, 15),
            (11, 15),
            (14, 10),
            (10, 15)]

    p = [2, 4, 7, 8]

    for i in p:
        size = envs[i]
        env = OrdinalEnv(*size, *size, 3, 3, rand_nets=False, filename="envs/unseen_envs/env_" + str(i) + ".txt")
        env = ObsTensorise(env)
        env = ActionTensorise(env)
        # env56.render_board(filename="results/unseen_envs/env_" + str(i) + ".png")

        print(i)
        best_r = -math.inf
        for _ in range(250):
            r = route_board(env, network)
            if r > best_r:
                best_r = r
                env.render_board(filename="results/unseen_envs/env_" + str(i) + ".png")


if __name__ == "__main__":
    actor_critic = load_network(get_args())
    print(actor_critic)
    route_unseen(actor_critic)
    # route_board(actor_critic, 0)
