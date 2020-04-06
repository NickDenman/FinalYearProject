import os
import pickle
from collections import deque
import numpy as np
import torch
from algorithms.ppo import PPO
from environment.ZeroTwentyPCBBoard import ZeroTwentyPCBBoard
from environment.pytorch_env import make_vec_env
from main_project import utils
from main_project.args import get_args
from main_project.model import ACNetwork
from main_project.storage import RolloutStorage
from main_project.utils import cleanup_log_dir
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir="logs/")


def load_network(args):
    load_path = os.path.join(args.save_dir, args.algo, "network.pt")
    ac = torch.load(load_path)[0]

    return ac


def main(load_agent=False):
    args = get_args()
    torch.manual_seed(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    if load_agent:
        actor_critic, device = load_network(args)
        route_board(actor_critic, 7)
        envs = make_vec_env(ZeroTwentyPCBBoard, args.seed, args.num_envs)
    else:
        envs = make_vec_env(ZeroTwentyPCBBoard, args.rows, args.cols, args.obs_rows, args.obs_cols, args.min_nets, args.max_nets, args.seed, args.num_envs)
        actor_critic = ACNetwork(envs.observation_space.shape, envs.action_space.n, 128, 128)

    agent = get_agent(actor_critic, args)
    learn(actor_critic, agent, args, envs)


def learn(actor_critic, agent, args, envs):
    rollouts = RolloutStorage(args.num_steps, args.num_envs, envs.observation_space.shape)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    result_rewards = []
    episode_rewards = deque(maxlen=args.log_interval)
    env_rewards = [0 for _ in range(args.num_envs)]

    value_losses = deque(maxlen=args.log_interval)
    action_losses = deque(maxlen=args.log_interval)
    entropies = deque(maxlen=args.log_interval)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_envs

    for update in range(num_updates):
        if args.use_linear_lr_decay:
            lr = agent.optimizer.lr if args.algo == "acktr" else args.lr
            utils.update_linear_schedule(agent.optimizer, update, num_updates, lr)

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

        value_losses.append(value_loss)
        action_losses.append(action_loss)
        entropies.append(dist_entropy)

        result_rewards.append(np.mean(episode_rewards))
        log_info(update, num_updates, episode_rewards, actor_critic, value_losses, action_losses, entropies, args, result_rewards)
        rollouts.after_update()


def log_info(update, num_updates, episode_rewards, actor_critic, value_losses, action_losses, entropies, args, result_rewards):
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
        # agent = algo.A2C_ACKTR(
        #     actor_critic,
        #     args.value_loss_coef,
        #     args.entropy_coef,
        #     lr=args.lr,
        #     eps=args.eps,
        #     alpha=args.alpha,
        #     max_grad_norm=args.max_grad_norm)
        pass
    elif args.algo == 'ppo':
        agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                    args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        pass

    return agent


def route_board(actor_critic, idx):
    # env = ZeroTwentyPCBBoard(5, 5, 5, 5, rand_nets=True, min_nets=3, max_nets=7, padded=True)
    env = ZeroTwentyPCBBoard(5, 5, rand_nets=False, filename="envs/small/5x5_2.txt", padded=True)
    obs = env.reset()
    env.render_board()
    done = False
    steps = 0
    reward = 0

    while not done:
        with torch.no_grad():
            t_obs = torch.from_numpy(obs).float()
            action, action_log_prob = actor_critic.act(t_obs)

        obs, r, done, _ = env.step(action.cpu().item())

        reward += r
        steps += 1

    env.render_board(filename="results/small_render_" + str(idx) + ".png")
    print(reward)


if __name__ == "__main__":
    actor_critic = load_network(get_args())
    # for i in range(8):
    #     route_board(actor_critic, i)
    route_board(actor_critic, 0)

    main()
