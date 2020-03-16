import torch


class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, n_actions):
        super().__init__()
        self.layers = []
        for hidden_layer in hidden_layers:
            self.layers.append(torch.nn.Linear(input_dim, hidden_layer))
            input_dim = hidden_layer
        self.output_layer = torch.nn.Linear(input_dim, n_actions)

    def forward(self, s):
        a_probs = s
        for layer in self.layers:
            a_probs = torch.nn.functional.relu(layer(a_probs))

        return self.output_layer(a_probs)


class Critic(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, n_actions):
        super().__init__()
        self.layers = []
        for hidden_layer in hidden_layers:
            self.layers.append(torch.nn.Linear(input_dim, hidden_layer))
            input_dim = hidden_layer
        self.output_layer = torch.nn.Linear(input_dim, n_actions)

    def forward(self, s):
        v = s
        for layer in self.layers:
            v = torch.nn.functional.relu(layer(v))

        return self.output_layer(v)
