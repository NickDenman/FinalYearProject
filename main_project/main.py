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


def load_network(args):
    load_path = os.path.join(args.save_dir, args.algo, "network.pt")
    ac = torch.load(load_path)[0]
    device = torch.device("cuda:0" if args.cuda else "cpu")

    return ac, device


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
        route_board(actor_critic, device, 7)
        envs = make_vec_env(ZeroTwentyPCBBoard, args.seed, args.num_envs, args.num_agents, device)
    else:
        device = torch.device("cuda:0" if args.cuda else "cpu")
        envs = make_vec_env(ZeroTwentyPCBBoard, args.seed, args.num_envs, args.num_agents, device)
        actor_critic = ACNetwork(envs.observation_space.shape, envs.action_space.n, 128)

    agent = get_agent(actor_critic, args)
    learn(actor_critic, agent, args, envs, device)


def learn(actor_critic, agent, args, envs, device):
    rollouts = RolloutStorage(args.num_steps, args.num_envs, envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    result_rewards = []
    episode_rewards = deque(maxlen=args.log_interval)
    env_rewards = [0 for _ in range(args.num_envs)]

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
        # agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
        pass

    return agent


def route_board(actor_critic, device, idx):
    env = single_zt_pcb("envs/small/5x5_" + str(idx) + ".txt", padded=True, num_agents=1)
    obs = env.reset()
    done = False
    steps = 0
    reward = 0

    while not done and steps < 64:
        with torch.no_grad():
            t_obs = torch.from_numpy(obs).float().to(device)
            action, action_log_prob = actor_critic.act(t_obs)

        obs, r, done, _ = env.step(action.cpu().item())

        reward += r
        steps += 1

    env.render_board()
    print(reward)


if __name__ == "__main__":
    main()
