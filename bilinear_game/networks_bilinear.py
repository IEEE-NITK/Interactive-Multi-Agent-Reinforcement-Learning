import torch
import torch.nn as nn
import torch.nn.functional as F

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Parameter(torch.rand(1))

    def forward(self):
        mu = self.actor
        return mu