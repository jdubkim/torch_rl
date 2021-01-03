import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

NUM_FRAMES = 4


def identity(x):
    return x


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


class CNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 output_limit=1.0,
                 activation=F.relu,
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False
                 ):
        super(CNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        self.conv1 = nn.Conv2d(
            in_channels=NUM_FRAMES,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.fc1 = nn.Linear(
            in_features=3200,
            out_features=256
        )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=output_size
        )

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))

        x = x * self.output_limit if self.use_actor else x

        return x
