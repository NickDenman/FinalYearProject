import torch


class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, n_actions):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_layer_1)
        self.layer2 = torch.nn.Linear(hidden_layer_1, hidden_layer_2)
        self.layer3 = torch.nn.Linear(hidden_layer_2, n_actions)

    def forward(self, s):
        a_probs = torch.nn.functional.relu(self.layer1(s))
        a_probs = torch.nn.functional.relu(self.layer2(a_probs))
        a_probs = torch.nn.functional.softmax(self.layer3(a_probs))

        return a_probs


class Critic(torch.nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, n_actions):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_layer_1)
        self.layer2 = torch.nn.Linear(hidden_layer_1, hidden_layer_2)
        self.layer3 = torch.nn.Linear(hidden_layer_2, n_actions)

    def forward(self, s):
        v = torch.nn.functional.relu(self.layer1(s))
        v = torch.nn.functional.relu(self.layer2(v))
        v = self.layer3(v)

        return v
