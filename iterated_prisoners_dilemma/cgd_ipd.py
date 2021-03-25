import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
print(sys.path)
print('\n\n')
import torch
import time
import numpy as np
from torch.distributions import Categorical
from cgd_optimizer import CGD, RCGD
from cgd_optimizer.critic_functions import get_advantage
from iterated_prisoners_dilemma.networks_ipd import *
from iterated_prisoners_dilemma.ipd_game import iterated_prisoner_dilemma
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

def main():
	# training settings
	parser = argparse.ArgumentParser(description = 'MAVPG iterated prisoner\'s dilemma')
	parser.add_argument('--n-epochs', type = int, default = 151, metavar = 'N',
						help = 'number of epochs to train (default: 151)')
	parser.add_argument('--repeated-steps', type = int, default = 200, metavar = 'N',
						help = 'number of repetitions of the matrix game (not known to the agents) default: 200')
	parser.add_argument('--lamda1', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 1 (default: 0.5)')
	parser.add_argument('--lamda2', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 2 (default: 0.5)')
	parser.add_argument('--lr-p', type = float, default = 0.5, metavar = 'LR',
						help = 'mavpg learning rate for actors (default: 0.5)')
	parser.add_argument('--lr-q1', type = float, default = 0.01, metavar = 'LR',
						help = 'critic 1 learning rate (default: 0.01)')
	parser.add_argument('--lr-q2', type = float, default = 0.01, metavar = 'LR',
						help = 'critic 2 learning rate (default: 0.01)')
	parser.add_argument('--beta', type = float, default = 0.99, metavar = 'MOM',
						help = 'momemtum (default: 0.99)')
	parser.add_argument('--eps', type = float, default = 1e-8, metavar = 'EPS',
						help = 'epsilon (default: 1e-8)')
	parser.add_argument('--run-num', type = int, default = 0, metavar = 'NUM',
						help = 'index of experiment run (default: 0)')
	parser.add_argument('--save-model', action = 'store_true', default = False,
						help = 'save model parameters or not (default: False)')
	parser.add_argument('--rms', action = 'store_true', default = False,
						help = 'use mavpg with rms or not (default: False)')
	parser.add_argument('--cuda', action = 'store_true', default = False,
						help = 'use cuda or not (default: False)')
	parser.add_argument('--tensorboard', action = 'store_true', default = False,
						help = 'use tensorboard or not (default: False')
	parser.add_argument('--activation-function', type = str, default = 'tan',
						help = 'which activation function to use (relu or tan, default: tan)')
	parser.add_argument('--policy', type = str, default = 'mlp',
						help = 'which type of policy to use (lstm or mlp, default: mlp)')
	parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'GAMMA',
						help = 'discount factor (default: 0.99)')
	parser.add_argument('--tau', type = float, default = 0.95, metavar = 'TAU',
						help = 'GAE factor (default: 0.95)')
	parser.add_argument('--logs', action = 'store_true', default = False,
						help = 'write data to logs or not (default: False')

	args = parser.parse_args()

	"""###################### Hyperparameters ########################
	n_epochs, n_eps, lamda1, lamda2, lr_p, lr_q1, lr_q2, gamma, tau, 
	obs_dim, state_dim, cuda, save_model, beta, eps, run_num, rms
	###############################################################"""

	use_cuda = args.cuda and torch.cuda.is_available()
	print(use_cuda, args.cuda, torch.cuda.is_available())
	#if args.cuda and not torch.cuda.is_available():
	#	print('CUDA CONFLICT')
	#	raise 'error'
	device = torch.device("cuda" if use_cuda else "cpu")

	env_location = '../tensorboard/iterated_prisoner_dilemma'
	experiment_name = '/mavpg_lr=' + str(args.lr_p) + '/run_' + str(args.run_num) + '_' + args.activation_function + '_' + args.policy
	model_mavpg = env_location + experiment_name + '/model'
	data_mavpg = env_location + experiment_name + '/data'

	if not os.path.exists(model_mavpg):
		os.makedirs(model_mavpg)
	if not os.path.exists(data_mavpg):
		os.makedirs(data_mavpg)

#	try:
#		env = iterated_prisoner_dilemma(args.repeated_steps)
#	except:
#		env = iterated_prisoner_dilemma(200)

	env = iterated_prisoner_dilemma(args.repeated_steps)
	# 0: CC, 1: CD, 2: DC, 3: DD, 4: inital state (observation is the action taken in the previous iteration)
	obs_dim = 5
	# combined observation of both the agents: s1 = [o1, o2], s2 = [o2, o1]
	state_dim = 10
	# C: Cooperate, D: Defect
	action_dim = 2

	if args.policy == 'mlp':
		if args.activation_function == 'tan':
			p1 = MLP_policy_tan(obs_dim, action_dim).to(device)
			p2 = MLP_policy_tan(obs_dim, action_dim).to(device)
		else:
			p1 = MLP_policy_relu(obs_dim, action_dim).to(device)
			p2 = MLP_policy_relu(obs_dim, action_dim).to(device)
	else:
		if args.activation_function == 'tan':
			p1 = LSTM_policy_tan(obs_dim, action_dim).to(device)
			p2 = LSTM_policy_tan(obs_dim, action_dim).to(device)
		else:
			p1 = LSTM_policy_relu(obs_dim, action_dim).to(device)
			p2 = LSTM_policy_relu(obs_dim, action_dim).to(device)

	q1 = value_network(state_dim).to(device)
	q2 = value_network(state_dim).to(device)

	if not args.rms:
		policy_optim = CGD(p1.parameters(), p2.parameters(), lr = args.lr_p, device = device)
	else:
		policy_optim = RCGD(p1.parameters(), p2.parameters(), lr = args.lr_p, beta = args.beta, eps = args.eps, device = device)
	optim_q1 = torch.optim.Adam(q1.parameters(), lr = args.lr_q1)
	optim_q2 = torch.optim.Adam(q2.parameters(), lr = args.lr_q2)

	if args.logs:
		logs_file_path = env_location+experiment_name+'/ipd_mavpg_lr='+str(args.lr_p)+'.txt'
		f_ptr = open(logs_file_path, 'a')
	
	if args.tensorboard:
		writer = SummaryWriter(data_mavpg)

	prob_c_1 = []
	prob_d_1 = []
	prob_c_2 = []
	prob_d_2 = []
	avg_rew_1 = []
	avg_rew_2 = []

	total_t_p_opt = 0
	total_t_q_opt = 0
	avg_t_p_opt = 0
	avg_t_q_opt = 0
	start = time.time()
	for epoch in range(args.n_epochs):
		state_1 = []
		state_2 = []
		action_1_2 = []
		reward_1 = []
		reward_2 = []
		observations, rewards, done = env.reset()

		while done == False:
			pi1_a_s = p1(torch.FloatTensor(observations[0]).to(device))
			dist1 = Categorical(pi1_a_s)
			action1 = dist1.sample().cpu()

			pi2_a_s = p2(torch.FloatTensor(observations[1]).to(device))
			dist2 = Categorical(pi2_a_s)
			action2 = dist2.sample().cpu()

			action = np.array([action1, action2])

			state_1.append(torch.FloatTensor(np.array([observations[0], observations[1]]).reshape(state_dim)))
			state_2.append(torch.FloatTensor(np.array([observations[1], observations[0]]).reshape(state_dim)))
			action_1_2.append(torch.FloatTensor(action))
			
			observations, rewards, done = env.step(action)
			reward_1.append(torch.FloatTensor([rewards[0]]))
			reward_2.append(torch.FloatTensor([rewards[1]]))

			if done == True:
				break

		val1 = q1(torch.stack(state_1).to(device))
		v1 = val1.detach().squeeze()
		v1 = torch.cat((v1, torch.FloatTensor([0])))
		r1 = torch.tensor(reward_1)

		# advantage1 is on gpu and detached
		advantage1 = get_advantage(v1.cpu(), r1, args.gamma, args.tau).to(device)

		val2 = q2(torch.stack(state_2).to(device))
		v2 = val2.detach().squeeze()
		v2 = torch.cat((v2, torch.FloatTensor([0])))
		r2 = torch.tensor(reward_2)
		
		advantage2 = get_advantage(v2.cpu(), r2, args.gamma, args.tau).to(device)

		avg_rew_1.append(sum(reward_1)/len(reward_1))
		avg_rew_2.append(sum(reward_2)/len(reward_2))
		
		if args.tensorboard:
			writer.add_scalar('Average reward in the epoch/Agent1', sum(reward_1)/len(reward_1), epoch)
			writer.add_scalar('Average reward in the epoch/Agent2', sum(reward_2)/len(reward_2), epoch)
			writer.add_scalar('Mean advantage in the epoch/Agent1', advantage1.mean().cpu(), epoch)
			writer.add_scalar('Mean advantage in the epoch/Agent2', advantage2.mean().cpu(), epoch)

		q1_loss = (r1.to(device) + args.gamma * val1[1:] - val1[:-1]).pow(2).mean()
		q2_loss = (r2.to(device) + args.gamma * val2[1:] - val2[:-1]).pow(2).mean()
		optim_q1.zero_grad()
		optim_q2.zero_grad()

		start_q = time.time()
		q1_loss.backward()
		optim_q1.step()
		q2_loss.backward()
		optim_q2.step()
		end_q = time.time()

		t_q_opt = end_q - start_q
		total_t_q_opt += t_q_opt
		avg_t_q_opt = total_t_q_opt/(epoch + 1)

		action_both = torch.stack(action_1_2)

		# action is either cooperate or defect
		if args.tensorboard:
			writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), epoch)
			writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), epoch)

		pi1_a_s = p1((torch.stack(state_1)[:, :obs_dim]).to(device))
		dist1 = Categorical(pi1_a_s)
		log_prob1 = dist1.log_prob(action_both[:,0])

		pi2_a_s = p2((torch.stack(state_2)[:, obs_dim:]).to(device))
		dist2 = Categorical(pi2_a_s)
		log_prob2 = dist2.log_prob(action_both[:,1])
		
		if args.tensorboard:
			writer.add_scalar('Entropy/Agent1', dist1.entropy().mean().detach(), epoch)
			writer.add_scalar('Entropy/Agent2', dist2.entropy().mean().detach(), epoch)

		cum_log_prob1 = torch.zeros(log_prob1.shape[0]-1).to(device)
		cum_log_prob2 = torch.zeros(log_prob2.shape[0]-1).to(device)
		cum_log_prob1[0] = log_prob1[0]
		cum_log_prob2[0] = log_prob2[0]
		
		for i in range(1, log_prob1.shape[0]-1):
			cum_log_prob1[i] = cum_log_prob1[i-1] + log_prob1[i]
			cum_log_prob2[i] = cum_log_prob2[i-1] + log_prob2[i]

		lp_x_1 = (log_prob1 * (-advantage1)).mean()
		lp_x_2 = (log_prob1 * (-advantage2)).mean()
		lp_y_1 = (log_prob2 * (-advantage1)).mean()
		lp_y_2 = (log_prob2 * (-advantage2)).mean()

		lp_x = args.lamda1 * lp_x_1 + args.lamda2 * lp_x_2
		lp_y = args.lamda1 * lp_y_1 + args.lamda2 * lp_y_2

		mh1_1 = (log_prob1 * log_prob2 * (-advantage1)).mean()
		mh2_1 = (log_prob1[1:] * cum_log_prob2 * (-advantage1)[1:]).mean()
		#mh2_1 = mh2_1.sum()/(mh2_1.size(0)-args.repeated_steps+1)
		mh3_1 = (log_prob2[1:] * cum_log_prob1 * (-advantage1)[1:]).mean()
		#mh3_1 = mh3_1.sum()/(mh3_1.size(0)-args.repeated_steps+1)

		mh1_2 = (log_prob1 * log_prob2 * (-advantage2)).mean()
		mh2_2 = (log_prob1[1:] * cum_log_prob2 * (-advantage2)[1:]).mean()
		#mh2_2 = mh2_2.sum()/(mh2_2.size(0)-args.repeated_steps+1)
		mh3_2 = (log_prob2[1:] * cum_log_prob1 * (-advantage2)[1:]).mean()
		#mh3_2 = mh3_2.sum()/(mh3_2.size(0)-args.repeated_steps+1)
			
		mh_1 = mh1_1 + mh2_1 + mh3_1
		mh_2 = mh1_2 + mh2_2 + mh3_2

		mh = args.lamda1 * mh_1 + args.lamda2 * mh_2

		policy_optim.zero_grad()
		
		start_p = time.time()
		policy_optim.step(lp_x, lp_y, mh)
		end_p = time.time()

		t_p_opt = end_p - start_p
		total_t_p_opt += t_p_opt
		avg_t_p_opt = total_t_p_opt/(epoch + 1)

		if args.tensorboard:
			writer.add_scalar('Mean_prob_cooperate/Agent1', pi1_a_s.data[:, 0].mean(), epoch)
			writer.add_scalar('Mean_prob_defect/Agent1', pi1_a_s.data[:, 1].mean(), epoch)
			writer.add_scalar('Mean_prob_cooperate/Agent2', pi2_a_s.data[:, 0].mean(), epoch)
			writer.add_scalar('Mean_prob_defect/Agent2', pi2_a_s.data[:, 1].mean(), epoch)

			writer.add_scalar('Time/avg_t_p_opt', avg_t_p_opt, epoch)
			writer.add_scalar('Time/avg_t_q_opt', avg_t_q_opt, epoch)
			writer.add_scalar('Time/t_p_opt', t_p_opt, epoch)
			writer.add_scalar('Time/t_q_opt', t_q_opt, epoch)

		if args.logs:
			s = 'Epoch: {}\nMean probability of Cooperate/Agent1: {}, Mean probability of Defect/Agent1: {}\nMean probability of Cooperate/Agent2: {}, Mean probability of Defect/Agent2: {}\n\n'.format(epoch, pi1_a_s.data[:, 0].mean(), pi1_a_s.data[:, 1].mean(), pi2_a_s.data[:, 0].mean(), pi2_a_s.data[:, 1].mean())
			f_ptr.write(s)
			s = 'Average reward in the epoch for agent 1: {}\nAverage reward in the epoch for agent 2: {}\n\n'.format(sum(reward_1)/len(reward_1), sum(reward_2)/len(reward_2))
			f_ptr.write(s)
			s = 'Time for policy optimization in this epoch: {} seconds\nTime for critic optimization in this epoch: {}seconds\n\n'.format(t_p_opt, t_q_opt)
			f_ptr.write(s + '###############################################################################\n\n')

		prob_c_1.append(pi1_a_s.data[:, 0].mean())
		prob_d_1.append(pi1_a_s.data[:, 1].mean())
		prob_c_2.append(pi2_a_s.data[:, 0].mean())
		prob_d_2.append(pi2_a_s.data[:, 1].mean())

		if args.save_model and epoch % 50 == 0:
			#print(epoch)
			torch.save(p1.state_dict(), model_mavpg + '/policy_agent1_' + str(epoch) + ".pth")
			torch.save(p2.state_dict(), model_mavpg + '/policy_agent2_' + str(epoch) + ".pth")
			torch.save(q1.state_dict(), model_mavpg + '/value_agent1_' + str(epoch) + ".pth")
			torch.save(q2.state_dict(), model_mavpg + '/value_agent2_' + str(epoch) + ".pth")

	end = time.time()
	total_time = end - start
	
	if args.logs:
		s = 'Total time taken: {} seconds\nTotal time for policy optimization steps only: {} seconds\nTotal time for critic optimization steps only: {} seconds\n'.format(total_time, total_t_p_opt, total_t_q_opt)
		f_ptr.write(s)
		s = 'Average time for policy optimization steps only: {} seconds\nAverage time for critic optimization steps only: {} seconds\n\n'.format(avg_t_p_opt, avg_t_q_opt)
		f_ptr.write(s)
		np_file = env_location+experiment_name+'/ipd_mavpg_lr='+str(args.lr_p)+'_rew_prob_c_d_time.npz'
		with open(np_file, 'wb') as np_f:
			np.savez(np_f, avg_rew_1 = np.array(avg_rew_1), avg_rew_2 = np.array(avg_rew_2), prob_c_1 = np.array(prob_c_1),\
					prob_d_1 = np.array(prob_d_1), prob_c_2 = np.array(prob_c_2), prob_d_2 = np.array(prob_d_2),\
					total_time = np.array(total_time), total_time_p_opt = np.array(total_t_p_opt), total_time_q_opt = np.array(total_t_q_opt),\
					avg_time_p_opt = np.array(avg_t_p_opt), avg_time_q_opt = np.array(avg_t_q_opt))

	fig = plt.figure(figsize = (15,15))
	ax = plt.subplot()
	ax.clear()
	fig.suptitle('Probabilities of cooperate and defect v/s #iterations/epochs (repeated steps = {})'.format(args.repeated_steps), fontsize = 25)
	plt.xlabel('$Iterations/Epochs$', fontsize = 20)
	plt.ylabel('$Probability$', fontsize = 20)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	ax.plot(np.array(prob_c_1), label = 'MAVPG, Prob_C1, lr = {}'.format(args.lr_p))
	ax.plot(np.array(prob_d_1), label = 'MAVPG, Prob_D1, lr = {}'.format(args.lr_p))
	ax.plot(np.array(prob_c_2), label = 'MAVPG, Prob_C2, lr = {}'.format(args.lr_p))
	ax.plot(np.array(prob_d_2), label = 'MAVPG, Prob_D2, lr = {}'.format(args.lr_p))
	plt.legend(loc = 'upper right')
	plt.grid()
	plt.savefig(env_location+experiment_name+'/ipd_mavpg_lr='+str(args.lr_p)+'_prob_c_d.png')

	fig = plt.figure(figsize = (15,15))
	ax = plt.subplot()
	ax.clear()
	fig.suptitle('Avg reward/epoch for Agent1 & Agent2 v/s #iterations/epochs (repeated steps = {})'.format(args.repeated_steps), fontsize = 20)
	plt.xlabel(r'$Iterations/Epochs$', fontsize = 20)
	plt.ylabel(r'$Average\ reward\ per\ epoch$', fontsize = 20)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	ax.plot(np.array(avg_rew_1), label = 'MAVPG, Avg rew/epoch Ag1, lr = {}'.format(args.lr_p))
	ax.plot(np.array(avg_rew_2), label = 'MAVPG, Avg rew/epoch Ag2, lr = {}'.format(args.lr_p))
	plt.legend(loc = 'upper right')
	plt.grid()
	plt.savefig(env_location+experiment_name+'/ipd_mavpg_lr='+str(args.lr_p)+'_avg_rew_per_epoch.png')

if __name__ == '__main__':
	main()
