import os
import pickle
from collections import deque

import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit

from algorithms.a2c import A2C
from algorithms.ppo import PPO
from environment.ZeroTwentyPCBBoard import ZeroTwentyPCBBoard
from environment.pytorch_env import make_vec_env, make_gym_env
from main_project import utils
from main_project.args import get_args
from main_project.model_linear import ACNetwork
from main_project.storage_conv import RolloutStorage
from main_project.utils import cleanup_log_dir


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_gym_env("CartPole-v0", args.num_envs, device, time_limit=True)
    actor_critic = ACNetwork(envs.observation_space.shape, envs.action_space.n, 128)

    agent = get_agent(actor_critic, args)
    learn(actor_critic, agent, args, envs, device)


def learn(actor_critic, agent, args, envs, device):
    rollouts = RolloutStorage(args.num_steps, args.num_envs, envs.observation_space.shape)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    result_rewards = []
    episode_rewards = deque(maxlen=args.log_interval)
    env_rewards = [0 for _ in range(args.num_envs)]

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_envs

    for update in range(num_updates):
        if args.use_linear_lr_decay:
            lr = agent.optimizer.lr if args.algo == "acktr" else args.lr
            utils.update_linear_schedule(agent.optimiser, update, num_updates, lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                actions, action_log_probs = actor_critic.act(rollouts.obs[step])
                values = actor_critic.get_value(rollouts.obs[step])

            obs, rewards, dones, _ = envs.step(actions)

            for i, (reward, done) in enumerate(zip(rewards.detach().squeeze(1).tolist(), dones)):
                env_rewards[i] += reward
                if done:
                    episode_rewards.append(env_rewards[i])
                    env_rewards[i] = 0

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
            rollouts.insert(obs, actions, action_log_probs, values, rewards, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()  # TODO: check if you need the detach as well s the no grad

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        result_rewards.append(np.mean(episode_rewards))
        log_info(update, num_updates, episode_rewards, actor_critic, value_loss, action_loss, dist_entropy, args, result_rewards)
        rollouts.after_update()


def log_info(update, num_updates, episode_rewards, actor_critic, value_loss, action_loss, entropy, args, result_rewards):
    if (update % args.save_interval == 0 or update == num_updates - 1) and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        torch.save([actor_critic], os.path.join(save_path, "network.pt"))
        pickle.dump(result_rewards, open("rewards_dump", "wb"))

    if (update % args.log_interval == 0) and len(episode_rewards) > 1:
        total_num_steps = (update + 1) * args.num_envs * args.num_steps
        print(
            "Updates {}, num timesteps {} - Last {} training episodes:\n"
            "    Mean/median reward {:.2f}/{:.2f}\n"
            "    Min/max reward {:.2f}/{:.2f}\n"
            "    Value loss: {:.2f}\n"
            "    Actor loss {:.2f}\n"
            "    Entropy {:.2f}\n"
                .format(update, total_num_steps,
                        len(episode_rewards),
                        np.mean(episode_rewards), np.median(episode_rewards),
                        min(episode_rewards), max(episode_rewards),
                        value_loss, action_loss, entropy))


def get_agent(actor_critic, args):
    if args.algo == 'a2c':
        agent = A2C(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            args.num_mini_batch,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                    args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)

    return agent


if __name__ == "__main__":
    main()
