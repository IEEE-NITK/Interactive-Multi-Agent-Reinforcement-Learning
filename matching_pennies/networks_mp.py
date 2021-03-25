# mp network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
num_players = 2

"""class policy1(nn.Module):
    def __init__(self):
        super(policy1, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.actor = nn.Parameter(torch.randn(num_players))

    def forward(self):
        mu = self.sm(self.actor)
        return mu"""

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.actor = nn.Parameter(torch.randn(num_players))

    def forward(self):
        mu = self.sm(self.actor)
        return mu