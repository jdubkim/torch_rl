import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from buffers import *
from networks import *


class DQNAgent(object):
    def __init__(self,
                 env,
                 args,
                 device,
                 obs_dim,
                 n_actions,
                 steps=0,
                 gamma=0.99,
                 epsilon=0.1,
                 epsilon_decay=0.995,
                 buffer_size=int(1e4),
                 batch_size=64,
                 target_update_step=100,
                 eval_mode=False,
                 q_losses=list(),
                 logger=dict()
                 ):

        self.env = env
        self.args = args
        self.obs_dim = obs_dim
        self.device = device
        self.n_actions = n_actions
        self.steps = steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_step = target_update_step
        self.eval_mode = eval_mode
        self.q_losses = q_losses
        self.logger = logger

        # Main Network
        self.qf = DQN(self.obs_dim, self.n_actions).to(self.device)

        # Target Network
        self.qf_target = DQN(self.obs_dim, self.n_actions).to(self.device)

        # Create an optimiser
        self.optimiser = optim.Adam(self.qf.parameters(), lr=1e-3)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, 1, self.buffer_size, self.device)

    def select_action(self, obs):
        if self.epsilon >= 0.01:
            self.epsilon *= self.epsilon_decay

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            action = self.qf(obs).argmax()
            return action.detach().cpu().numpy()

    def train_model(self):
        batch = self.replay_buffer.sample(self.batch_size)
        o1_b = batch['obs1']
        o2_b  = batch['obs2']
        a_b = batch['acts']
        r_b = batch['rews']
        d_b =  batch['dones']

        if self.args.DEBUG:
            print("o1: ", o1_b.shape)
            print("o2: ", o2_b.shape)
            print("a: ", a_b.shape)
            print("r: ", r_b.shape)
            print("d: ", d_b.shape)

        # Prediction Q(s)
        q = self.qf(o1_b).gather(1, a_b.long()).squeeze(1)
        # Target for Q regression
        q_target = self.qf_target(o2_b)

        q_backup = r_b + self.gamma* (1-d_b)*q_target.max(1)[0]
        q_backup.to(self.device)

        qf_loss = F.mse_loss(q, q_backup.detach())
        self.optimiser.zero_grad()
        qf_loss.backward()
        self.optimiser.step()

        if self.steps % self.target_update_step == 0:
            hard_target_update(self.qf, self.qf_target)

        # Save loss
        self.q_losses.append(qf_loss.item())

    def run(self, max_step):
        step_number = 0
        total_reward = 0.

        obs = self.env.reset()
        done = False

        while not (done or step_number == max_step):
            if self.args.render:
                self.env.render()

            if self.eval_mode:
                q_value = self.qf(torch.Tensor(obs).to(self.device)).argmax()
                action = q_value.detach().cpu().numpy()
                next_obs, reward, done, _ = self.env.step(action)
            else:
                self.steps += 1

                action = self.select_action(torch.Tensor(obs).to(self.device))
                next_obs, reward, done, _ = self.env.step(action)

                # Add experience to reply buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)

                # Start training when the number of experience is greater htan batch_size

                if self.steps > self.batch_size:
                    self.train_model()

            total_reward += reward
            step_number += 1
            obs = next_obs

        self.logger['LossQ'] = round(np.mean(self.q_losses), 5)

        return step_number, total_reward


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
        self.network = MLP(obs_dim, n_actions,
                           output_limit, hidden_sizes, activation,
                           output_activation, use_output_layer, use_actor)

    def forward(self, input):
        return self.network.forward(input)


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

