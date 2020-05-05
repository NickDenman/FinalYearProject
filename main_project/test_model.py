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


def route_board(actor_critic, idx):
    device = torch.device("cpu")
    env = make_vec_env(OrdinalEnv, device, 10, 10, 15, 15, 3, 3, None, 1, conv=True)

    obs = env.reset()
    env.env_method("render_board")
    done = False
    steps = 0
    reward = 0

    while True:
        with torch.no_grad():
            action, action_log_prob = actor_critic.act(obs)

        obs, r, done, _ = env.step(action)
        env.env_method("render_board")
        if done:
            print(done)
        reward += r
        steps += 1

    env.env_method("render_board")
    print(reward)


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

    while len(p) < 10000:
        print(len(p))

        obs = env.reset()
        done = False
        steps = 0
        reward = 0

        while not done and steps < 450:
            with torch.no_grad():
                action, _ = network.act(obs)

            obs, r, done, info = env.step(action)
            if done and info['completed']:
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
    print(p)
    print(np.mean(p))


if __name__ == "__main__":
    actor_critic = load_network(get_args())
    print(actor_critic)
    compare_to_optimal(actor_critic)
    # route_board(actor_critic, 0)
