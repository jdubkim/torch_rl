import numpy as np
import torch
import torch.optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self,
                 obs_dim,
                 n_actions,
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=F.relu,
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False):
        super(DQN, self).__init__()
        self.network = MLP(obs_dim, n_actions, output_size,
                           output_limit, hidden_sizes, activation,
                           output_activation, use_output_layer, use_actor)

    def forward(self, input):
        return self.network.forward(x)


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=F.relu,
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False
                 ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))

        x = x * self.output_limit if self.use_actor else x
        return x

