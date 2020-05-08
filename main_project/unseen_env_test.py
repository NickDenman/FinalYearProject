import math
import os

import torch
from gym import ObservationWrapper, ActionWrapper

from environment.ordinal_env import OrdinalEnv
from main_project.args import get_args


def load_network(args):
    load_path = os.path.join(args.save_dir, "net_exp", "ppo", "2c3l", "ppo",
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

    while not done and steps < 250:
        with torch.no_grad():
            action, _ = network.act(obs)

        obs, r, done, info = env.step(action)
        if done and info['completed']:
            pass
        reward += r
        steps += 1

    return reward, info['completed']


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

    for i in range(1):
        size = envs[i]
        env = OrdinalEnv(*size, *size, 3, 3, rand_nets=False, filename="envs/unseen_envs/env_" + str(i) + ".txt")
        env = ObsTensorise(env)
        env = ActionTensorise(env)
        # env56.render_board(filename="results/unseen_envs/env_" + str(i) + ".png")
        first = True
        best_l = math.inf
        for j in range(250):
            if j % 25 == 0:
                print(str(i) + ": " + str(j))
            r, done = route_board(env, network)
            env.render_board(
                filename="results/unseen_envs/spam/env_" + str(i) + "-" + str(
                    j) + ".png")


if __name__ == "__main__":
    actor_critic = load_network(get_args())
    print(actor_critic)
    route_unseen(actor_critic)
    # route_board(actor_critic, 0)
