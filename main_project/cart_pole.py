import os
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.a2c import A2C
from algorithms.ppo import PPO
from environment.pytorch_env import make_cart_pole_vec_env
from main_project import utils
from main_project.args import get_args
from main_project.model_linear import ACNetwork as ACNetworkLinear
from main_project.storage_linear import RolloutStorage as RolloutStorageLinear
from main_project.utils import cleanup_log_dir


def main():
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(device)

    envs = make_cart_pole_vec_env(args.seed, args.num_envs)
    actor_critic = ACNetworkLinear(envs.observation_space.shape[0],
                                   envs.action_space.n,
                                   args.linear_layers)

    print(actor_critic)

    actor_critic.to(device)
    agent = get_agent(actor_critic, args)
    learn(actor_critic, agent, args, envs, device)


def learn(actor_critic, agent, args, envs, device):
    writer = SummaryWriter(log_dir=args.log_dir)
    rollouts = RolloutStorageLinear(args.num_steps,
                                    args.num_envs,
                                    envs.observation_space.shape)

    obs = envs.reset()
    rollouts.save_obs(obs, 0)
    rollouts.to(device)

    ep_rewards = [0 for _ in range(args.num_envs)]
    episode_rewards = deque(maxlen=args.log_interval * 2)

    value_losses = deque(maxlen=args.log_interval * 2)
    action_losses = deque(maxlen=args.log_interval * 2)
    entropies = deque(maxlen=args.log_interval * 2)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_envs

    for update in range(num_updates):
        graduate = False
        if args.use_linear_lr_decay:
            lr = args.lr
            utils.update_linear_schedule(agent.optimizer,
                                         update,
                                         num_updates,
                                         lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                actions, action_log_probs = \
                    actor_critic.act(rollouts.get_obs(step))
                values = actor_critic.get_value(rollouts.get_obs(step))

            obs, rewards, dones, info = envs.step(actions)

            for i, (reward, done) in \
                    enumerate(zip(rewards.detach().squeeze(1).tolist(), dones)):
                ep_rewards[i] += reward
                if done:
                    episode_rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0

            masks = torch.FloatTensor([[int(1 - done_)] for done_ in dones])
            rollouts.insert(obs, actions, action_log_probs, values, rewards,
                            masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.get_obs(-1)).detach()

        rollouts.compute_returns(next_value,
                                 args.use_gae,
                                 args.gamma,
                                 args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        value_losses.append(value_loss)
        action_losses.append(action_loss)
        entropies.append(dist_entropy)

        log_info(writer, update, num_updates, episode_rewards, actor_critic,
                 value_losses, action_losses, entropies, args)
        rollouts.after_update()


def log_info(writer,
             update,
             num_updates,
             episode_rewards,
             actor_critic,
             value_losses,
             action_losses,
             entropies,
             args):
    if (update % args.save_interval == 0 or update == num_updates - 1) and \
            args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        torch.save([actor_critic], os.path.join(save_path, "network.pt"))


    if (update % args.log_interval == 0) and len(episode_rewards) == 2 * \
            args.log_interval:
        total_num_steps = (update + 1) * args.num_envs * args.num_steps
        v_loss = np.mean(value_losses)
        a_loss = np.mean(action_losses)
        e = np.mean(entropies)
        writer.add_scalar('Loss/critic', v_loss, total_num_steps)
        writer.add_scalar('Loss/actor', a_loss, total_num_steps)
        writer.add_scalar('Entropy', e, total_num_steps)
        writer.add_scalar('Results/mean', np.mean(episode_rewards), total_num_steps)
        writer.flush()
        print(
            "Update {} of {}, num timesteps {} - Last {} training episodes: \n"
            "    mean/median reward {:.1f}/{:.1f}\n"
            "    min/max reward {:.1f}/{:.1f}\n"
            "    Entropy {:.2f}\n"
            "    Value loss {:.2f}\n"
            "    Actor loss {:.2f}\n"
            .format(update, num_updates, total_num_steps, len(episode_rewards),
                    np.mean(episode_rewards), np.median(episode_rewards),
                    np.min(episode_rewards), np.max(episode_rewards),
                    e,
                    v_loss,
                    a_loss))


def get_agent(actor_critic, args):
    if args.algo == 'a2c':
        agent = A2C(actor_critic, args.value_loss_coef, args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(actor_critic, args.clip_param, args.ppo_epoch,
                    args.num_mini_batch, args.value_loss_coef,
                    args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        pass

    return agent


if __name__ == "__main__":
    main()
