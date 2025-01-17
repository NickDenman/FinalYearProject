import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--continue-learning',
        action='store_true',
        default=False,
        help='continue learning from some other network')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.05,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.3,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=16,
        help='how many training envs to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=100,
        help='number of forward steps in A2C (default: 10)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=4,
        help='number of batches for ppo (default: 4)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='log interval, one log per n updates (default: 100)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--min-rows',
        type=int,
        default=5,
        help='rows in board (default: 5)')
    parser.add_argument(
        '--min-cols',
        type=int,
        default=5,
        help='cols in board (default: 5)')
    parser.add_argument(
        '--max-rows',
        type=int,
        default=10,
        help='max rows in board (default: 10)')
    parser.add_argument(
        '--max-cols',
        type=int,
        default=10,
        help='max cols in board (default: 10)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1e8,
        help='number of environment steps to train (default: 10e8)')
    parser.add_argument(
        '--linear-layers',
        type=int,
        nargs='+',
        help='number of layers in linear network')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--conv',
        action='store_true',
        default=False,
        help='Use convolutional network.')
    parser.add_argument(
        '--curriculum',
        action='store_true',
        default=False,
        help='Use curriculum learning')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo']

    return args
