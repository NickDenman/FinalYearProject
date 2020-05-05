import os
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.a2c import A2C
from algorithms.ppo import PPO
from environment.OrdinalEnv import OrdinalEnv
from environment.pytorch_env import make_vec_env
from main_project import utils
from main_project.args import get_args
from main_project.model_conv1 import ACNetwork as ACNetworkConv1
from main_project.model_linear import ACNetwork as ACNetworkLinear
from main_project.storage_conv import RolloutStorage as RolloutStorageConv
from main_project.storage_linear import RolloutStorage as RolloutStorageLinear
from main_project.utils import cleanup_log_dir


def load_network(args):
    load_path = os.path.join(args.save_dir, args.algo, "network.pt")
    ac = torch.load(load_path)[0]

    return ac


def main(env):
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(device)

    if args.curriculum:
        envs = make_vec_env(env, device, args.min_rows, args.min_cols,
                            args.max_rows, args.max_cols, 1, 1, args.seed,
                            args.num_envs, conv=args.conv)
    else:
        envs = make_vec_env(env, device, 5, 5, 5, 5, 3, 3, args.seed,
                            args.num_envs, conv=args.conv)

    if args.continue_learning:
        actor_critic = load_network(args)
    else:
        if args.conv:
            actor_critic = ACNetworkConv1(envs.observation_space.spaces,
                                          envs.action_space.n,
                                          args.linear_layers)
        else:
            actor_critic = ACNetworkLinear(envs.observation_space.shape,
                                           envs.action_space.n,
                                           args.linear_layers)

    print(actor_critic)

    actor_critic.to(device)
    agent = get_agent(actor_critic, args)
    learn(actor_critic, agent, args, envs, device)


def learn(actor_critic, agent, args, envs, device):
    if args.curriculum:
        min_nets = 1
        max_nets = 1
    else:
        min_nets = 3
        max_nets = 3

    writer = SummaryWriter(log_dir=args.log_dir)
    if args.conv:
        rollouts = RolloutStorageConv(args.num_steps, args.num_envs,
                                      envs.observation_space)
    else:
        rollouts = RolloutStorageLinear(args.num_steps, args.num_envs,
                                        envs.observation_space.shape)

    obs = envs.reset()
    rollouts.save_obs(obs, 0)
    rollouts.to(device)

    graduate_len = 2500
    graduate_list = deque(maxlen=graduate_len)

    episode_rewards = deque(maxlen=args.log_interval * 2)
    boards_completed = deque(maxlen=args.log_interval * 2)
    env_rewards = [0 for _ in range(args.num_envs)]

    value_losses = deque(maxlen=args.log_interval * 2)
    action_losses = deque(maxlen=args.log_interval * 2)
    entropies = deque(maxlen=args.log_interval * 2)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_envs

    for update in range(num_updates):
        graduate = False
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, update,
                                         num_updates, args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                actions, action_log_probs = actor_critic.act(
                    rollouts.get_obs(step))
                values = actor_critic.get_value(rollouts.get_obs(step))

            obs, rewards, dones, info = envs.step(actions)

            for i, (reward, done) in \
                    enumerate(zip(rewards.detach().squeeze(1).tolist(), dones)):
                env_rewards[i] += reward
                if done:
                    episode_rewards.append(env_rewards[i])
                    boards_completed.append(int(info[i]["completed"]))
                    graduate_list.append(int(info[i]["completed"]))
                    env_rewards[i] = 0

            masks = torch.FloatTensor([[1.0 - done] for done in dones])
            rollouts.insert(obs, actions, action_log_probs, values, rewards,
                            masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.get_obs(-1)).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        value_losses.append(value_loss)
        action_losses.append(action_loss)
        entropies.append(dist_entropy)
        grad_percentage = np.sum(graduate_list) / graduate_len

        if args.curriculum and grad_percentage > 0.85 and \
                (min_nets < 3 or max_nets < 3):
            min_nets = min_nets + 1
            max_nets = max_nets + 1
            graduate_list.clear()
            envs.env_method("increase_env_size", min_nets, max_nets)
            graduate = True

        log_info(writer,
                 update,
                 num_updates,
                 episode_rewards,
                 boards_completed,
                 actor_critic,
                 value_losses,
                 action_losses,
                 entropies,
                 args,
                 min_nets,
                 max_nets,
                 graduate,
                 grad_percentage,
                 graduate_list)
        rollouts.after_update()


def log_info(writer,
             update,
             num_updates,
             episode_rewards,
             boards_completed,
             actor_critic,
             value_losses,
             action_losses,
             entropies,
             args,
             min_nets,
             max_nets,
             graduate,
             grad_percentage,
             graduate_list):
    if (update % args.save_interval == 0 or update == num_updates - 1) \
            and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        torch.save([actor_critic], os.path.join(save_path, "network.pt"))


    if (update % args.log_interval == 0 or graduate) and \
            len(episode_rewards) == 2 * args.log_interval:
        total_num_steps = (update + 1) * args.num_envs * args.num_steps
        v_loss = np.mean(value_losses)
        a_loss = np.mean(action_losses)
        e = np.mean(entropies)
        routed = np.mean(boards_completed) * 100
        writer.add_scalar('Loss/critic', v_loss, total_num_steps)
        writer.add_scalar('Loss/actor', a_loss, total_num_steps)
        writer.add_scalar('Entropy', e, total_num_steps)
        writer.add_scalar('Curriculum/mean_nets',
                          (min_nets + max_nets) / 2, total_num_steps)
        writer.add_scalar('Curriculum/min_nets', min_nets, total_num_steps)
        writer.add_scalar('Curriculum/max_nets', max_nets, total_num_steps)
        writer.add_scalar('Results/mean',
                          np.mean(episode_rewards), total_num_steps)
        writer.add_scalar('Results/percentage_routed', routed, total_num_steps)
        writer.flush()
        print(
            "Update {} of {}, num timesteps {} - Last {} training episodes: \n"
            "    Percentage_routed {:.1f}\n"
            "    Graduate_avg {:.5f}\n"
            "    Graduate_list avg {:.5f} @ {}\n"
            "    mean/median reward {:.1f}/{:.1f}\n"
            "    min/max reward {:.1f}/{:.1f}\n"
            "    Entropy {:.2f}\n"
            "    Value loss {:.2f}\n"
            "    Actor loss {:.2f}\n"
            .format(update, num_updates, total_num_steps, len(episode_rewards),
                    routed,
                    grad_percentage,
                    np.mean(graduate_list), len(graduate_list),
                    np.mean(episode_rewards), np.median(episode_rewards),
                    np.min(episode_rewards), np.max(episode_rewards),
                    e,
                    v_loss,
                    a_loss))

    if graduate:
        print("\n\n\n==================== GRADUATE =====================\n\n\n")


def get_agent(actor_critic, args):
    if args.algo == 'a2c':
        agent = A2C(actor_critic,
                    args.value_loss_coef,
                    args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm,
                    conv=True)
    elif args.algo == 'ppo':
        agent = PPO(actor_critic,
                    args.clip_param,
                    args.ppo_epoch,
                    args.num_mini_batch,
                    args.value_loss_coef,
                    args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)

    return agent


if __name__ == "__main__":
    main(OrdinalEnv)
