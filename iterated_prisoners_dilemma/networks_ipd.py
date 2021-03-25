import torch
import torch.nn as nn
import torch.nn.functional as f

"""class MLP_policy_tan(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MLP_policy_tan, self).__init__()
        self.l0 = nn.Linear(obs_dim, 16)
        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, action_dim)
    
    def forward(self, obs):
        print('##############################################################################')
        print(obs)
        mu = torch.tanh(self.l0(obs))
        print(mu)
        
        print('params for layer 0\n')
        print(self.l0.weight)
        print(self.l0.bias)
        mu = torch.tanh(self.l1(mu))
        print(mu)
        
        print('params for layer 1\n')
        print(self.l1.weight)
        print(self.l1.bias)
        mu = torch.tanh(self.l2(mu))
        print(mu)
        
        print('params for layer 2\n')
        print(self.l2.weight)
        print(self.l2.bias)
        mu = f.softmax(mu, dim = -1)
        print(mu)

        print('##############################################################################')
        return mu"""

class MLP_policy_tan(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MLP_policy_tan, self).__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 16),
                                    nn.Tanh(),
                                    nn.Linear(16, 8),
                                    nn.Tanh(),
                                    nn.Linear(8, action_dim),
                                    nn.Softmax(dim = -1))
    
    def forward(self, obs):
        mu = self.actor(obs)
        return mu

class MLP_policy_relu(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MLP_policy_relu, self).__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 8),
                                    nn.ReLU(),
                                    nn.Linear(8, action_dim),
                                    nn.Softmax(dim = -1))
    
    def forward(self, obs):
        mu = self.actor(obs)
        return mu

class LSTM_policy(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass

class value_network(nn.Module):
    def __init__(self, state_dim):
        super(value_network, self).__init__()
        self.critic = nn.Sequential(nn.Linear(state_dim, 16),
                                    nn.Tanh(),
                                    nn.Linear(16, 8),
                                    nn.Tanh(),
                                    nn.Linear(8, 1))
    
    def forward(self, state):
        val = self.critic(state)
        return val
