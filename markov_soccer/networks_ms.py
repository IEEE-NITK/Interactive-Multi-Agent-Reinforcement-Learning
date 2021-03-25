import torch
import torch.nn as nn

class policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(policy, self).__init__()
		self.actor = nn.Sequential(nn.Linear(state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 32),
			nn.Tanh(),
			nn.Linear(32, action_dim),
			nn.Softmax(dim=-1))

		#self.apply(init_params)

	def forward(self, state):
		mu = self.actor(state)
		return mu

class critic(nn.Module):
	def __init__(self, state_dim):
		super(critic, self).__init__()
		self.critic = nn.Sequential(nn.Linear(state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 32),
			nn.Tanh(),
			nn.Linear(32, 1))

		#self.apply(init_params)

	def forward(self, state):
		return self.critic(state)

def init_params(params):
	if isinstance(params, nn.Linear):
		nn.init.xavier_normal_(params.weight)
		nn.init.xavier_normal_(params.bias)