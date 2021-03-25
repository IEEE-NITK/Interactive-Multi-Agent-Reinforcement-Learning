import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
import torch
import torch.nn as nn
import time
import numpy as np
from torch.distributions import Categorical, Bernoulli
from ipd_batch import IPD_batch
#from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
	def __init__(self):
		self.self_logprobs = []
		self.other_logprobs = []
		self.values = []
		self.rewards = []

	def add(self, lp, other_lp, v, r):
		self.self_logprobs.append(lp)
		self.other_logprobs.append(other_lp)
		self.values.append(v)
		self.rewards.append(r)

	def dice_objective(self):
		# torch dim = (batch_size, repeated_steps)
		self_logprobs = torch.stack(self.self_logprobs, dim = 1)
		other_logprobs = torch.stack(self.other_logprobs, dim = 1)
		values = torch.stack(self.values, dim = 1)
		rewards = torch.stack(self.rewards, dim = 1)
		#print(self_logprobs.size())

		discount_mask = torch.cumprod(torch.ones(rewards.size()) * args.gamma, dim = 1)/args.gamma
		discounted_rewards = rewards * discount_mask
		discounted_values = values * discount_mask

		stochastic_nodes = self_logprobs + other_logprobs
		dependencies = torch.cumsum(stochastic_nodes, dim = 1)
		d_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim = 1))

		if args.use_baseline:
			baseline = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim = 1))
			d_objective += baseline

		return -d_objective

	def gae(self, rewards, values, gamma, tau, gamma_weighted, gamma_mask):
		deltas = rewards + values[:,1:] * gamma - values[:,:-1]
		advantages = torch.zeros_like(deltas).float()
		for t in range(deltas.size(1)-1,-1,-1):
			advantages[:,t] = advantages[:,t-1] * gamma * tau + deltas[:,t]
		if gamma_weighted:
			advantages *= gamma_mask
		return advantages

	def make_objective(self, tau, gamma, dice_lambda, use_baseline, dice_method = 'old', gamma_weighted = True):
		# torch dim = (batch_size, repeated_steps)
		self_logprobs = torch.stack(self.self_logprobs, dim = 1)
		other_logprobs = torch.stack(self.other_logprobs, dim = 1)
		values = torch.stack(self.values, dim = 1)
		values = torch.cat((values, torch.zeros(1, values.size()[1])))
		rewards = torch.stack(self.rewards, dim = 1)
		gamma_mask = torch.cumprod(torch.ones(rewards.size()) * gamma, dim = 1)/gamma
		advantages = self.gae(rewards, values, gamma, tau, gamma_weighted, gamma_mask)

		if dice_method == 'old':
			stochastic_nodes = self_logprobs + other_logprobs
			dep = torch.cumsum(stochastic_nodes, dim = 1)
			if gamma_weighted:
				rewards *= gamma_mask
			objective = torch.mean(torch.sum(magic_box(dep) * rewards, dim = 1))
			if use_baseline:
				baseline = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * values[:,:-1] * gamma_mask, dim = 1))
				objective += baseline
		
		elif dice_method == 'loaded':
			self_dep1 = torch.zeros_like(self_logprobs)
			other_dep1 = torch.zeros_like(other_logprobs)
			for t in range(1, args.repeated_steps):
				self_dep1[:,t] = self_logprobs[:,t-1] * dice_lambda + self_logprobs[:,t]
				other_dep1[:,t] = other_logprobs[:,t-1] * dice_lambda + other_logprobs[:,t]
			self_dep2 = self_dep1 - self_logprobs
			other_dep2 = other_dep1 - other_logprobs
			dep1 = self_dep1 + other_dep1
			dep2 = self_dep2 + other_dep2
			objective = torch.mean(torch.sum((magic_box(dep1) - magic_box(dep2)) * advantages, dim = 1))
		
		else:
			raise Exception('DiCE method passed: {}, allowed only old and loaded'.format(dice_method))
		
		return -objective

	def value_loss(self):
		rewards = torch.stack(self.rewards, dim = 1)
		values = torch.stack(self.values, dim = 1)
		#values = torch.cat((values, torch.zeros(1, values.size()[1])))
		#v_loss = torch.mean((rewards + args.gamma * values[1:,:] - values[:-1,:])**2)
		v_loss = torch.mean((rewards - values)**2)
		return v_loss

def act(batch_state, theta, phi):
	# take action of a time step for the whole batch
	batch_state = torch.from_numpy(batch_state).long()
	# probability of cooperation - for the whole batch
	probs = torch.sigmoid(theta)[batch_state]
	# Bernoulli(x) means 1 is sampled with probability of x
	dist = Bernoulli(1-probs)
	# 0 == cooperation, 1 == defection
	actions = dist.sample()
	log_probs = dist.log_prob(actions)
	return actions.numpy().astype(int), log_probs, phi[batch_state]

def get_gradient(objective, theta):
	grad = torch.autograd.grad(objective, (theta), create_graph = True)[0]
	return grad

class Agent():
	def __init__(self):
		self.theta = nn.Parameter(torch.zeros(5), requires_grad = True)
		self.theta_optim = torch.optim.Adam((self.theta,), lr = args.lr_outer)
		self.phi = nn.Parameter(torch.zeros(5), requires_grad = True)
		self.phi_optim = torch.optim.Adam((self.phi,), lr = args.lr_val)

	def theta_update(self, objective):
		self.theta_optim.zero_grad()
		objective.backward(retain_graph = True)
		self.theta_optim.step()
	
	def phi_update(self, value_loss):
		self.phi_optim.zero_grad()
		value_loss.backward()
		self.phi_optim.step()
	
	def in_lookahead(self, other_theta, other_phi):
		# rollout `batch_size` number of trajectories (each trajectory of length = `repeated_steps`) ->
		# under self.theta and other_theta
		# return the grad_wrt_(other_theta)_of_L_DiCE(self.theta, other_theta)

		# initialize the env with 0 as the initial state | s1 size is equal to the batch size
		(s1, s2), _, done = env.reset()
		# memory to keep track of other agent's data
		other_memory = Memory()
		# rollout `batch_size` number of time step plays (in parallel) for `repeated_steps`
		while done is not True:
			a1, lp1, v1 = act(s1, self.theta, self.phi)
			a2, lp2, v2 = act(s2, other_theta, other_phi)
			(s1, s2), (r1, r2), done = env.step((a1, a2))
			other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())
		
		other_objective = other_memory.dice_objective()
		grad = get_gradient(other_objective, other_theta)
		return grad

	def out_lookahead(self, other_theta, other_phi):
		# rollout `batch_size` number of trajectories (each trajectory of length = `repeated_steps`) ->
		# under self.theta and other_theta
		# update self.theta

		# initialize the env with 0 as the initial state | s0 size is equal to the batch size
		(s1, s2), _, done = env.reset()
		# memory to keep track of agent's own data
		memory = Memory()
		# rollout `batch_size` number of time step plays (in parallel) for `repeated_steps`
		while done is not True:
			a1, lp1, v1 = act(s1, self.theta, self.phi)
			a2, lp2, v2 = act(s2, other_theta, other_phi)
			(s1, s2), (r1, r2), done = env.step((a1, a2))
			memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

		d_objective = memory.dice_objective()
		self.theta_update(d_objective)
		value_loss = memory.value_loss()
		self.phi_update(value_loss)

def step(theta1, theta2, phi1, phi2):
	(s1, s2), _, done = env.reset()
	score1 = 0
	score2 = 0
	while done is not True:
		a1, lp1, v1 = act(s1, theta1, phi1)
		a2, lp2, v2 = act(s2, theta2, phi2)
		(s1, s2), (r1, r2), done = env.step((a1, a2))
		# reward of that time step averaged over the whole batch
		score1 += np.mean(r1)
		score2 += np.mean(r2)
	# reward averaged over all time steps
	return score1/float(args.repeated_steps), score2/float(args.repeated_steps)

def play(agent1, agent2, n_lookaheads):
	scores = []
	print('Starting iterations with number of lookaheads = {}'.format(n_lookaheads))
	for itr in range(args.n_itrs):
		# copy the parameters
		theta1_ = agent1.theta.clone().detach().requires_grad_(True)
		phi1_ = agent1.phi.clone().detach().requires_grad_(True)
		theta2_ = agent2.theta.clone().detach().requires_grad_(True)
		phi2_ = agent2.phi.clone().detach().requires_grad_(True)

		# both agents imagining other agents learning
		for k in range(n_lookaheads):
			grad2 = agent1.in_lookahead(theta2_, phi2_)
			grad1 = agent2.in_lookahead(theta1_, phi1_)
			theta2_ = theta2_ - args.lr_inner * grad2
			theta1_ = theta1_ - args.lr_inner * grad1

		# both agents updating their own parameters after the imagination
		agent1.out_lookahead(theta2_, phi2_)
		agent2.out_lookahead(theta1_, phi1_)

		# score1 is mean return over `repeated_steps` for agent 1 averaged over the full `batch_size`
		score1, score2 = step(agent1.theta, agent2.theta, agent1.phi, agent2.phi)
		scores.append((score1, score2))

		if itr%10==0 or itr == args.n_itrs-1:
			# probabilities of cooperation given the 5 types of states
			pC1 = [p.item() for p in torch.sigmoid(agent1.theta)]
			pC2 = [p.item() for p in torch.sigmoid(agent2.theta)]
			print('######################################################################')
			print('Iteration {}'.format(itr))
			print('Prob C1|init_state: {}, Prob C2|init_state: {}'.format(pC1[0], pC2[0]))
			print('Prob C1|CC: {}, Prob C2|CC: {}'.format(pC1[1], pC2[1]))
			print('Prob C1|CD: {}, Prob C2|CD: {}'.format(pC1[2], pC2[2]))
			print('Prob C1|DC: {}, Prob C2|DC: {}'.format(pC1[3], pC2[3]))
			print('Prob C1|DD: {}, Prob C2|DD: {}'.format(pC1[4], pC2[4]))
			print('Score agent 1: {}'.format(score1))
			print('Score agent 2: {}'.format(score2))
			print('######################################################################')
		if itr == args.n_itrs-1 and args.save_model:
			torch.save(agent1.theta, env_location + experiment_name + '/lookahead_' + str(n_lookaheads) + '_lola_dice_agent1_theta_policy.pth')
			torch.save(agent2.theta, env_location + experiment_name + '/lookahead_' + str(n_lookaheads) + '_lola_dice_agent2_theta_policy.pth')
			torch.save(agent1.phi, env_location + experiment_name + '/lookahead_' + str(n_lookaheads) + '_lola_dice_agent1_phi_value.pth')
			torch.save(agent2.phi, env_location + experiment_name + '/lookahead_' + str(n_lookaheads) + '_lola_dice_agent2_phi_value.pth')

	return scores

def main():
	parser = argparse.ArgumentParser(description = 'LOLA DICE IPD')
	parser.add_argument('--n-itrs', type = int, default = 200, metavar = 'N')
	parser.add_argument('--batch-size', type = int, default = 64, metavar = 'N')
	parser.add_argument('--repeated-steps', type = int, default = 150, metavar = 'N')
	parser.add_argument('--lr-inner', type = float, default = 0.3, metavar = 'LR')
	parser.add_argument('--lr-outer', type = float, default = 0.2, metavar = 'LR')
	parser.add_argument('--lr-val', type = float, default = 0.1, metavar = 'LR')
	parser.add_argument('--gamma', type = float, default = 0.96, metavar = 'GAMMA')
	parser.add_argument('--use-baseline', action = 'store_true', default = False)
	parser.add_argument('--num-lookaheads', type = int, default = 1)
	parser.add_argument('--exec-num', type = int, default = 0)
	parser.add_argument('--seed', type = int, default = 42)
	parser.add_argument('--save-model', action = 'store_true', default = False)

	global args
	args = parser.parse_args()
	global env
	env = IPD_batch(args.repeated_steps, args.batch_size)

	global env_location, experiment_name
	env_location = '../tensorboard/iterated_prisoner_dilemma'
	experiment_name = '/lola_dice'

	fig, axes = plt.subplots(1, 3, figsize = (22,6))
	x_labels = ["Iterations"]*3
	y_labels = ['Average reward Agent 1', 'Average reward Agent 2', 'Joint average reward']
  
	for lookahead in range(args.num_lookaheads):
		torch.manual_seed(args.seed)
		scores = play(Agent(), Agent(), lookahead)
		axes[0].plot([score[0] for score in scores], label = str(lookahead) + ' lookaheads')
		axes[1].plot([score[1] for score in scores], label = str(lookahead) + ' lookaheads')
		axes[2].plot([sum(score)/2 for score in scores], label = str(lookahead) + ' lookaheads')
  
	for i in range(3):
		axes[i].set_ylabel(y_labels[i], fontsize = 20)
		axes[i].set_xlabel(x_labels[i], fontsize = 20)
		axes[i].grid()
		axes[i].legend(loc = "lower left")
	fig.tight_layout()
	#fig.show()
	fig.savefig(env_location + experiment_name + '/ipd_lola_dice_avg_return.png')

if __name__ == '__main__':
	main()