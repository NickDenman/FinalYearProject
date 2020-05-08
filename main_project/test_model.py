import math
import os

import numpy as np
import torch
from gym import ObservationWrapper, ActionWrapper

from environment.ordinal_env import OrdinalEnv
from main_project.AStar import astar
from main_project.args import get_args


def load_network(args, path):
    load_path = os.path.join(args.save_dir, path)
    ac = torch.load(load_path)[0]

    return ac


def route_board(env, network):
    obs = env.reset()
    done = False
    steps = 0
    reward = 0

    while not done and steps < 650:
        with torch.no_grad():
            action, _ = network.act(obs)

        obs, r, done, info = env.step(action)
        if done and info['completed']:
            pass
        reward += r
        steps += 1

    return reward, info['completed']


class ObsTensorise(ObservationWrapper):
    def observation(self, observation):
        return torch.Tensor(observation[0]).unsqueeze(0).unsqueeze(0), torch.Tensor(observation[1]).unsqueeze(0)


class ActionTensorise(ActionWrapper):

    def action(self, action):
        return action.item()

    def reverse_action(self, action):
        pass


def compare_to_optimal(network):
    env = OrdinalEnv(10, 10, 15, 15, 3, 3)
    env = ObsTensorise(env)
    env = ActionTensorise(env)

    p = []
    q = []
    while len(p) < 1000:
        print(len(p))

        obs = env.reset()
        done = False
        steps = 0
        reward = 0

        while not done and steps < 150:
            with torch.no_grad():
                action, _ = network.act(obs)

            obs, r, done, info = env.step(action)
            if done:
                q.append(int(done))
                if info['completed']:
                    a = astar(env)
                    if a is not None:
                        _, optimal = a
                        agent_length = env.get_trace_length()
                        percent_diff = (agent_length - optimal) / optimal
                        percent_diff = max(0, percent_diff)
                        p.append(percent_diff)
            reward += r
            steps += 1

    env.render_board()
    # print(p)
    print("optimal diff: " + str(np.mean(p)))
    print("success rate: " + str(np.mean(q)))


def success_percent(network):
    env = OrdinalEnv(10, 10, 15, 15, 3, 3)
    env = ObsTensorise(env)
    env = ActionTensorise(env)

    p = []
    for i in range(2500):
        print(i)
        _, done = route_board(env, network)
        p.append(int(done))

    print(np.mean(p))


def random_comparison():
    args = get_args()
    a2c = load_network(args, "net_exp/a2c/1c3l/a2c/network.pt")
    ppo = load_network(args, "net_exp/ppo/2c3l/ppo/network.pt")

    env = OrdinalEnv(25, 12, 25, 12, 3, 3, rand_nets=False, filename="envs/medium.txt")
    env = ObsTensorise(env)
    env = ActionTensorise(env)

    best_l = math.inf
    best_r = -math.inf
    max_n = 0
    for j in range(1000):
        if j % 25 == 0:
            print(j)
        r, done = route_board(env, ppo)
        # env.render_board()
        l = env.get_trace_length()
        if len(env.routed_nets) > max_n or (len(env.routed_nets) == max_n and l < best_l):
            env.render_board()
            best_l = l
            max_n = len(env.routed_nets)

    # best_l = math.inf
    # for j in range(500):
    #     if j % 25 == 0:
    #         print(j)
    #     r, done = route_board(env, ppo)
    #     l = env.get_trace_length()
    #     if done and l < best_l:
    #         env.render_board("results/ppo.png")
    #         best_l = l


if __name__ == "__main__":
    # actor_critic = load_network(get_args(), "net_exp/a2c/1c3l/a2c/network.pt")
    # success_percent(network=actor_critic)
    # print(actor_critic)
    # compare_to_optimal(actor_critic)
    # route_board(actor_critic, 0)
    random_comparison()
