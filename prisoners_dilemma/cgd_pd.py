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
from prisoners_dilemma.networks_pd import policy
from prisoners_dilemma.pd_game import prisoner_dilemma
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

def main():
	# training settings
	parser = argparse.ArgumentParser(description = 'MAVPG prisoner dilemma')
	parser.add_argument('--n-epochs', type = int, default = 151, metavar = 'N',
						help = 'number of epochs to train (default: 151)')
	parser.add_argument('--n-eps', type = int, default = 1000, metavar = 'N',
						help = 'number of episodes in an epoch (default: 1000)')
	parser.add_argument('--lamda1', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 1 (default: 0.5)')
	parser.add_argument('--lamda2', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 2 (default: 0.5)')
	parser.add_argument('--lr-p', type = float, default = 0.5, metavar = 'LR',
						help = 'mavpg learning rate for actors (default: 0.5)')
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

	env_location = '../tensorboard/prisoner_dilemma'
	experiment_name = '/mavpg_lr=' + str(args.lr_p) + '/run_' + str(args.run_num)
	model_mavpg = env_location + experiment_name + '/model'
	data_mavpg = env_location + experiment_name + '/data'

	if not os.path.exists(model_mavpg):
		os.makedirs(model_mavpg)
	if not os.path.exists(data_mavpg):
		os.makedirs(data_mavpg)

	p1 = policy().to(device)
	p2 = policy().to(device)

	env = prisoner_dilemma()

	if not args.rms:
		policy_optim = CGD(p1.parameters(), p2.parameters(), lr = args.lr_p, device = device)
	else:
		policy_optim = RCGD(p1.parameters(), p2.parameters(), lr = args.lr_p, beta = args.beta, eps = args.eps, device = device)

	logs_file_path = env_location+experiment_name+'/pd_mavpg_lr='+str(args.lr_p)+'.txt'
	f_ptr = open(logs_file_path, 'a')
	
	if args.tensorboard:
		writer = SummaryWriter(data_mavpg)

	prob_c_1 = []
	prob_d_1 = []
	prob_c_2 = []
	prob_d_2 = []
	avg_rew_1 = []
	avg_rew_2 = []

	t_opt = 0
	start = time.time()
	for epoch in range(args.n_epochs):

		state_a = []
		state_b = []
		action_a_b = []
		reward_a = []
		reward_b = []
		done_eps = []
		state, _, _, _, _ = env.reset()

		for _ in range(args.n_eps):

			dist1 = Categorical(p1())
			# sample either cooperate or defect
            # 0 == cooperate, 1 == defect
			action1 = dist1.sample().cpu()

			dist2 = Categorical(p2())
            # sample either cooperate or defect
            # 0 == cooperate, 1 == defect
			action2 = dist2.sample().cpu()

			action = np.array([action1, action2])

			state_a.append(torch.FloatTensor(state))
			state_b.append(torch.FloatTensor(state))
			action_a_b.append(torch.FloatTensor(action))
		
			state, reward1, reward2, done, _ = env.step(action)

			reward_a.append(torch.FloatTensor([reward1]))
			reward_b.append(torch.FloatTensor([reward2]))

			done_eps.append(torch.FloatTensor([1-done]))

		action_both = torch.stack(action_a_b)

		# action is either cooperate or defect
		if args.tensorboard:
			writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), epoch)
			writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), epoch)

		advantage1 = torch.stack(reward_a).to(device).transpose(0,1)
		advantage2 = torch.stack(reward_b).to(device).transpose(0,1)

		avg_rew_1.append(advantage1.mean().cpu())
		avg_rew_2.append(advantage2.mean().cpu())

		if args.tensorboard:
			writer.add_scalar('Average reward in the epoch/Agent1', advantage1.mean().cpu(), epoch)
			writer.add_scalar('Average reward in the epoch/Agent2', advantage2.mean().cpu(), epoch)
        
		pi1_a_s = p1()
		dist1 = Categorical(pi1_a_s)
		log_prob1 = dist1.log_prob(action_both[:,0])

		pi2_a_s = p2()
		dist2 = Categorical(pi2_a_s)
		log_prob2 = dist2.log_prob(action_both[:,1])

		if args.tensorboard:
			writer.add_scalar('Entropy/Agent1', dist1.entropy().data, epoch)
			writer.add_scalar('Entropy/Agent2', dist2.entropy().data, epoch)

		lp_x_1 = (log_prob1 * (-advantage1)).mean()
		lp_x_2 = (log_prob1 * (-advantage2)).mean()
		lp_y_1 = (log_prob2 * (-advantage1)).mean()
		lp_y_2 = (log_prob2 * (-advantage2)).mean()

		lp_x = args.lamda1 * lp_x_1 + args.lamda2 * lp_x_2
		lp_y = args.lamda1 * lp_y_1 + args.lamda2 * lp_y_2

		mh_1 = (log_prob1 * log_prob2 * (-advantage1)).mean()
		mh_2 = (log_prob1 * log_prob2 * (-advantage2)).mean()

		mh = args.lamda1 * mh_1 + args.lamda2 * mh_2

		policy_optim.zero_grad()
		
		start_opt = time.time()
		policy_optim.step(lp_x, lp_y, mh)
		end_opt = time.time()
		
		t_opt += (end_opt - start_opt)

		if args.tensorboard:
			writer.add_scalar('Time/t_opt', t_opt, epoch)

			writer.add_scalar('Prob_cooperate/Agent1', pi1_a_s.data[0], epoch)
			writer.add_scalar('Prob_defect/Agent1', pi1_a_s.data[1], epoch)
			writer.add_scalar('Prob_cooperate/Agent2', pi2_a_s.data[0], epoch)
			writer.add_scalar('Prob_defect/Agent2', pi2_a_s.data[1], epoch)

		prob_c_1.append(pi1_a_s.data[0])
		prob_d_1.append(pi1_a_s.data[1])
		prob_c_2.append(pi2_a_s.data[0])
		prob_d_2.append(pi2_a_s.data[1])

		s = 'epoch: {}\nProbability of Cooperate/Agent1: {}, Probability of Defect/Agent1: {}\nProbability of Cooperate/Agent2: {}, Probability of Defect/Agent2: {}\n\n'.format(epoch, pi1_a_s.data[0], pi1_a_s.data[1], pi2_a_s.data[0], pi2_a_s.data[1])
		f_ptr.write(s)

		if args.save_model and epoch % 50 == 0:
			#print(epoch)
			torch.save(p1.state_dict(), model_mavpg + '/policy_agent1_' + str(epoch) + ".pth")
			torch.save(p2.state_dict(), model_mavpg + '/policy_agent2_' + str(epoch) + ".pth")
	
	end = time.time()
	total_time = end - start
	s = 'Total time taken: {} seconds, Total time for optimization steps only: {} seconds \n\n'.format(total_time, t_opt)
	f_ptr.write(s)
	np_file = env_location+experiment_name+'/pd_mavpg_lr='+str(args.lr_p)+'_rew_prob_c_d_time.npz'
	with open(np_file, 'wb') as np_f:
		np.savez(np_f, avg_rew_1 = np.array(avg_rew_1), avg_rew_2 = np.array(avg_rew_2), prob_c_1 = np.array(prob_c_1), prob_d_1 = np.array(prob_d_1),\
			prob_c_2 = np.array(prob_c_2), prob_d_2 = np.array(prob_d_2), total_time = np.array(total_time), time_opt = np.array(t_opt))
	
	fig = plt.figure(figsize = (15,15))
	ax = plt.subplot()
	ax.clear()
	fig.suptitle('Probabilities of cooperate and defect v/s #iterations/epochs', fontsize = 25)
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
	plt.savefig(env_location+experiment_name+'/pd_mavpg_lr='+str(args.lr_p)+'_prob_c_d.png')

	fig = plt.figure(figsize = (15,15))
	ax = plt.subplot()
	ax.clear()
	fig.suptitle('Average reward per epoch for Agent 1 and Agent 2 v/s #iterations/epochs', fontsize = 25)
	plt.xlabel('$Iterations/Epochs$', fontsize = 20)
	plt.ylabel('$Average reward per epoch$', fontsize = 20)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	ax.plot(np.array(avg_rew_1), label = 'MAVPG, Avg rew/epoch Ag1, lr = {}'.format(args.lr_p))
	ax.plot(np.array(avg_rew_2), label = 'MAVPG, Avg rew/epoch Ag2, lr = {}'.format(args.lr_p))
	plt.legend(loc = 'upper right')
	plt.grid()
	plt.savefig(env_location+experiment_name+'/pd_mavpg_lr='+str(args.lr_p)+'_avg_rew_per_epoch.png')

if __name__ == '__main__':
	main()
