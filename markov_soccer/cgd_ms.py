import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
sys.path.append('/scratch/scratch1/videh/open_spiel')
sys.path.append('/scratch/scratch1/videh/open_spiel/build/python')
print(sys.path)
print('\n\n\n')
import torch
import time
import numpy as np
from open_spiel.python import rl_environment
from torch.distributions import Categorical
from markov_soccer.networks_ms import policy, critic
from cgd_optimizer import maVPG, RmaVPG
from cgd_optimizer.critic_functions import get_advantage
from torch.utils.tensorboard import SummaryWriter
from markov_soccer.trans_dynamics import get_two_state

def main():
	# training settings
	parser = argparse.ArgumentParser(description = 'MAVPG markov soccer')
	parser.add_argument('--n-epochs', type = int, default = 30001, metavar = 'N',
						help = 'number of epochs to train (default: 30001)')
	parser.add_argument('--n-eps', type = int, default = 10, metavar = 'N',
						help = 'number of episodes in an epoch (default: 10)')
	parser.add_argument('--lamda1', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 1 (default: 0.5)')
	parser.add_argument('--lamda2', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 2 (default: 0.5)')
	parser.add_argument('--lr-p', type = float, default = 0.01, metavar = 'LR',
						help = 'mavpg learning rate for actors (default: 0.01)')
	parser.add_argument('--lr-q1', type = float, default = 0.01, metavar = 'LR',
						help = 'critic 1 learning rate (default: 0.01)')
	parser.add_argument('--lr-q2', type = float, default = 0.01, metavar = 'LR',
						help = 'critic 2 learning rate (default: 0.01)')
	parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'GAMMA',
						help = 'discount factor (default: 0.99)')
	parser.add_argument('--tau', type = float, default = 0.95, metavar = 'TAU',
						help = 'GAE factor (default: 0.95)')
	parser.add_argument('--obs-dim', type = int, default = 12, metavar = 'DIM',
						help = 'dimension of observation space of each agent (default: 12)')
	parser.add_argument('--state-dim', type = int, default = 12, metavar = 'DIM',
						help = 'dimension of state space (default: 12)')
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

	args = parser.parse_args()

	"""###################### Hyperparameters ########################
	n_epochs, n_eps, lamda1, lamda2, lr_p, lr_q1, lr_q2, gamma, tau, 
	obs_dim, state_dim, cuda, save_model, beta, eps, run_num, rms
	###############################################################"""

	use_cuda = args.cuda and torch.cuda.is_available()
	print(use_cuda, args.cuda, torch.cuda.is_available())
	device = torch.device("cuda" if use_cuda else "cpu")

	env_location = '../tensorboard/markov_soccer'
	experiment_name = '/mavpg_lr=' + str(args.lr_p)
	model_mavpg = env_location + experiment_name + '/model'
	data_mavpg = env_location + experiment_name + '/data'

	if not os.path.exists(model_mavpg):
		os.makedirs(model_mavpg)
	if not os.path.exists(data_mavpg):
		os.makedirs(data_mavpg)

	writer = SummaryWriter(data_mavpg)

	game = 'markov_soccer'
	env = rl_environment.Environment(game)
	action_dim = env.action_spec()['num_actions']

	p1 = policy(args.obs_dim, action_dim).to(device)
	p2 = policy(args.obs_dim, action_dim).to(device)
	q1 = critic(args.state_dim).to(device)
	q2 = critic(args.state_dim).to(device)

	if not args.rms:
		policy_optim = CGD(p1.parameters(), p2.parameters(), lr = args.lr_p, device = device)
	else:
		policy_optim = RCGD(p1.parameters(), p2.parameters(), lr = args.lr_p, beta = args.beta, eps = args.eps, device = device)
	optim_q1 = torch.optim.Adam(q1.parameters(), lr = args.lr_q1)
	optim_q2 = torch.optim.Adam(q2.parameters(), lr = args.lr_q1)

	n_wins_a = n_wins_a_500 = 0
	n_wins_b = n_wins_b_500 = 0

	type_obs = 'full' if args.state_dim == args.obs_dim else 'partial'
	
	# file to keep all training logs
	# log_file_path = ../tensorboard/markov_soccer/mavpg_lr=0.01/ms_full_mavpg_lr=0.01_run_num=0.txt
	logs_file_path = env_location+experiment_name+'/ms_'+type_obs+'_mavpg_lr='+str(args.lr_p)+'_run_num='+str(args.run_num)+'.txt'
	f_ptr = open(logs_file_path, 'a')

	# file to track the time in episodes
	# time_in_episode_file_path = ../tensorboard/markov_soccer/mavpg_lr=0.01/time_in_episode_full_mavpg_lr=0.01_run_num=0.txt
	
	#file_path = '../tensorboard/markov_soccer/copg_lr0_01/avg_time_in_epoch_run0.txt'
	#f_t2 = open(file_path, 'a')

	avg_t_opt = 0
	avg_t_p_opt = 0
	avg_t_q_opt = 0
	t_episode = []

	start = time.time()
	for epoch in range(args.n_epochs):
		state_a = []
		state_b = []
		action_a_b = []
		reward_a = []
		reward_b = []

		timestep = env.reset()
		a_status, b_status, s_a, s_b = get_two_state(timestep)

		time_in_epoch = 0
		avg_time_in_epoch = 0
		#avg_time_in_eps_per_epoch = 0
		for eps in range(args.n_eps):
			time_in_episode = 0
			#print(timestep.last())
			timestep = env.reset()
			a_status, b_status, s_a, s_b = get_two_state(timestep)
			while timestep.last() == False:

				#print('HI')
				time_in_episode += 1
				# pi1 == softmax output
				pi1 = p1(torch.FloatTensor(s_a).to(device))
				dist1 = Categorical(pi1)
				action1 = dist1.sample().cpu()

				# pi2 == softmax output
				pi2 = p2(torch.FloatTensor(s_b).to(device))
				dist2 = Categorical(pi2)
				action2 = dist2.sample().cpu()

				action = np.array([action1, action2])

				state_a.append(torch.FloatTensor(s_a))
				state_b.append(torch.FloatTensor(s_b))
				action_a_b.append(torch.FloatTensor(action))
				
				timestep = env.step(action)
				a_status, b_status, s_a, s_b = get_two_state(timestep)

				rew_a = timestep.rewards[0]
				rew_b = timestep.rewards[1]

				reward_a.append(torch.FloatTensor([rew_a]))
				reward_b.append(torch.FloatTensor([rew_b]))
				
				"""if timestep.last() == True:
					done == True
				else:
					done == False"""

				# 0 if either A or B wins (and hence game ends), 1 if game lasts more than 1000 steps (game ended in draw)
				#done_eps.append(torch.FloatTensor([1-done]))

				if timestep.last == True:
					writer.add_scalar('reward_zero_sum for agent1', reward_a, epoch)
					#timestep = env.reset()
					#a_status, b_status, s_a, s_b = get_two_state(timestep)
					break

			t_episode.append(time_in_episode)
			s = 'epoch number: {}, time in episode {}: {}, Done reached'.format(epoch, eps, time_in_episode)
			#print(s)
			f_ptr.write(s + '\n')
			if rew_a > 0:
				s = 'A won episode {} of epoch {}'.format(eps, epoch)
				# print(s)
				f_ptr.write(s + '\n')
				n_wins_a += 1
				n_wins_a_500 += 1
			if rew_b > 0:
				s = 'B won episode {} of epoch {}'.format(eps, epoch)
				# print(s)
				f_ptr.write(s + '\n')
				n_wins_b += 1
				n_wins_b_500 += 1

			time_in_epoch += time_in_episode
			s = 'Total time in episode {} of epoch {}: {}'.format(eps, epoch, time_in_episode)
			#print(s)
			f_ptr.write(s + '\n\n')
			writer.add_scalar('Total time in this episode', time_in_episode, epoch * args.n_eps + eps)
			#avg_time_in_epoch = time_in_episode/n_eps #(avg_time_in_eps_per_epoch * eps + time_in_episode)/(eps + 1)
			#writer.add_scalar('Average time in this epoch', avg_time_in_epoch, n_epochs)

		avg_time_in_epoch = time_in_epoch/args.n_eps
		s = 'Total time in epoch {}: {}'.format(epoch, time_in_epoch)
		#print(s)
		f_ptr.write(s + '\n')
		s = 'Average time in epoch {}: {}'.format(epoch, avg_time_in_epoch)
		#print(s)
		f_ptr.write(s + '\n\n')
		writer.add_scalar('Total time in this epoch', time_in_epoch, epoch)
		writer.add_scalar('Average time in this epoch', avg_time_in_epoch, epoch)

		start_opt = time.time()

		val1 = q1(torch.stack(state_a).to(device))
		v1 = val1.detach().squeeze()
		v1 = torch.cat((v1, torch.FloatTensor([0])))
		r1 = torch.tensor(reward_a)

		# advantage1 is on gpu and detached
		advantage1 = get_advantage(v1.cpu(), r1, args.gamma, args.tau).to(device)

		val2 = q2(torch.stack(state_b).to(device))
		v2 = val2.detach().squeeze()
		v2 = torch.cat((v2, torch.FloatTensor([0])))
		r2 = torch.tensor(reward_b)
		
		advantage2 = get_advantage(v2.cpu(), r2, args.gamma, args.tau).to(device)

		q1_loss = (r1.to(device) + args.gamma * val1[1:] - val1[:-1]).pow(2).mean()
		optim_q1.zero_grad()
		q1_loss.backward()
		optim_q1.step()

		q2_loss = (r2.to(device) + args.gamma * val2[1:] - val2[:-1]).pow(2).mean()
		optim_q2.zero_grad()
		q2_loss.backward()
		optim_q2.step()

		end_q = time.time()
		t_q_opt = end_q - start_opt
		avg_t_q_opt = (avg_t_q_opt * epoch + t_q_opt)/(epoch + 1)

		s = 'Mean advantage for agent 1 (eta_new - eta_old): {}\nMean advantage for agent 2 -(eta_new - eta_old): {}'.format(advantage1.mean().cpu(), advantage2.mean().cpu())
		# print(s)
		f_ptr.write(s + '\n')
		writer.add_scalar('Mean advantage for agent1', advantage1.mean().cpu(), epoch)
		writer.add_scalar('Mean advantage for agent2', advantage2.mean().cpu(), epoch)

		action_both = torch.stack(action_a_b)

		pi1_a_s = p1(torch.stack(state_a).to(device))
		dist1 = Categorical(pi1_a_s)
		log_prob1 = dist1.log_prob(action_both[:,0])

		pi2_a_s = p2(torch.stack(state_b).to(device))
		dist2 = Categorical(pi2_a_s)
		log_prob2 = dist2.log_prob(action_both[:,1])

		cum_log_prob1 = torch.zeros(log_prob1.shape[0]-1).to(device)
		cum_log_prob2 = torch.zeros(log_prob2.shape[0]-1).to(device)
		cum_log_prob1[0] = log_prob1[0]
		cum_log_prob2[0] = log_prob2[0]
		
		for i in range(1, log_prob1.shape[0]-1):
			cum_log_prob1[i] = cum_log_prob1[i-1] + log_prob1[i]
			cum_log_prob2[i] = cum_log_prob2[i-1] + log_prob2[i]

		lp_x_1 = (log_prob1 * advantage1).mean()
		lp_x_2 = (log_prob1 * advantage2).mean()
		lp_y_1 = (log_prob2 * advantage1).mean()
		lp_y_2 = (log_prob2 * advantage2).mean()

		lp_x = args.lamda1 * lp_x_1 - args.lamda2 * lp_x_2
		lp_y = args.lamda1 * lp_y_1 - args.lamda2 * lp_y_2

		mh1_1 = (log_prob1 * log_prob2 * advantage1).mean()
		mh2_1 = log_prob1[1:] * cum_log_prob2 * advantage1[1:]
		mh2_1 = mh2_1.sum()/(mh2_1.size(0)-args.n_eps+1)
		mh3_1 = log_prob2[1:] * cum_log_prob1 * advantage1[1:]
		mh3_1 = mh3_1.sum()/(mh3_1.size(0)-args.n_eps+1)

		mh1_2 = (log_prob1 * log_prob2 * advantage2).mean()
		mh2_2 = log_prob1[1:] * cum_log_prob2 * advantage2[1:]
		mh2_2 = mh2_2.sum()/(mh2_2.size(0)-args.n_eps+1)
		mh3_2 = log_prob2[1:] * cum_log_prob1 * advantage2[1:]
		mh3_2 = mh3_2.sum()/(mh3_2.size(0)-args.n_eps+1)
			
		mh_1 = mh1_1 + mh2_1 + mh3_1		
		mh_2 = mh1_2 + mh2_2 + mh3_2

		mh = args.lamda1 * mh_1 - args.lamda2 * mh_2

		policy_optim.zero_grad()
		policy_optim.step(lp_x, lp_y, mh)
		#policy_optim.step(log_prob1, log_prob2, cum_log_prob1, cum_log_prob2, advantage1, advantage2, lamda1, lamda2, n_eps)

		end_opt = time.time()
		t_p_opt = end_opt - end_q
		avg_t_p_opt = (avg_t_p_opt * epoch + t_p_opt)/(epoch + 1)

		t_opt = end_opt - start_opt
		avg_t_opt = (avg_t_opt * epoch + t_opt)/(epoch + 1)

		if (epoch + 1) % 500 == 0:
			s = '\nA won {} of games till now | B won {} of games in last 500 episodes'.format(n_wins_a_500/500, n_wins_b_500/500)
	        #print(s)
			f_ptr.write(s + '\n')
			writer.add_scalar('Win rate last 500 episodes for agent1', (n_wins_a_500)/500, epoch)
			writer.add_scalar('Win rate last 500 episodes for agent2', (n_wins_b_500)/500, epoch)
			n_wins_a_500 = 0
			n_wins_b_500 = 0

		tot_games = n_wins_a + n_wins_b
		s = '\nA won {} of games till now | B won {} of games till now'.format(n_wins_a/tot_games, n_wins_b/tot_games)
		#print(s)
		#print('\n')
		#print('##################################################################################################################')
		#print('\n')
		f_ptr.write(s+'\n'+'##################################################################################################################'+'\n\n')

		writer.add_scalar('Entropy for agent1', dist1.entropy().mean().detach(), epoch)
		writer.add_scalar('Entropy for agent2', dist2.entropy().mean().detach(), epoch)

		writer.add_scalar('Cum win rate for agent1', (n_wins_a/tot_games), epoch)
		writer.add_scalar('Cum win rate for agent2', (n_wins_b/tot_games), epoch)

		writer.add_scalar('Time/avg_t_opt', avg_t_opt, epoch)
		writer.add_scalar('Time/avg_t_p_opt', avg_t_p_opt, epoch)
		writer.add_scalar('Time/avg_t_q_opt', avg_t_q_opt, epoch)
		writer.add_scalar('Time/t_opt', t_opt, epoch)
		writer.add_scalar('Time/t_p_opt', t_p_opt, epoch)
		writer.add_scalar('Time/t_q_opt', t_q_opt, epoch)

		if args.save_model and epoch % 500 == 0:
			#print(epoch)
			torch.save(p1.state_dict(), model_mavpg + '/policy_agent1_' + str(epoch) + ".pth")
			torch.save(p2.state_dict(), model_mavpg + '/policy_agent2_' + str(epoch) + ".pth")
			torch.save(q1.state_dict(), model_mavpg + '/value_agent1_' + str(epoch) + ".pth")
			torch.save(q2.state_dict(), model_mavpg + '/value_agent2_' + str(epoch) + ".pth")

	end = time.time()
	total_time = end - start
	s = 'Total time taken: {} seconds \n\n'.format(total_time)
	f_ptr.write(s)
	np_file = env_location+experiment_name+'/ms_mavpg_lr='+str(args.lr_p)+'_n_wins_time.npz'
	with open(np_file, 'wb') as np_f:
		np.savez(np_f, n_wins_a = np.array(n_wins_a), n_wins_b = np.array(n_wins_b), time_in_episode = np.array(t_episode), total_time = np.array(total_time))
	#time_in_episode_file_path = env_location+experiment_name+'/time_in_episode_ms_'+type_obs+'_mavpg_lr='+str(args.lr_p)+'_run_num='+str(args.run_num)+'.txt'
	#f_t1 = open(time_in_episode_file_path, 'a')

if __name__ == '__main__':
	main()
