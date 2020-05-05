import glob
import os

from torch import nn


def generate_linear_layers(input, hidden_sizes, output, init_,
                           activation=nn.ReLU, final_activation=None):
    layers = [init_(nn.Linear(input, hidden_sizes[0])), activation()]
    for i in range(len(hidden_sizes) - 1):
        layers.append(init_(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
        layers.append(activation())
    layers.append(init_(nn.Linear(hidden_sizes[-1], output)))
    if final_activation is not None:
        layers.append(final_activation())

    return layers


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

