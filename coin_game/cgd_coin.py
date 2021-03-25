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
from networks_coin import *
from coin_game import *
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import yaml

def main():
	# training settings
	parser = argparse.ArgumentParser(description = 'MAVPG coin game')
	parser.add_argument('--n-epochs', type = int, default = 151, metavar = 'N',
						help = 'number of epochs to train (default: 151)')
	parser.add_argument('--n-eps', type = int, default = 10, metavar = 'N',
						help = 'number of episodes in an epoch (default: 10)')
	parser.add_argument('--max-steps', type = int, default = 500, metavar = 'N',
						help = 'maximum number of steps for which the episode lasts (default: 500)')
	parser.add_argument('--interval', type = int, default = 50, metavar = 'N',
						help = 'Interval of epochs to plot stats for last N steps and save model (default: N = 50)')
	parser.add_argument('--lamda1', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 1 (default: 0.5)')
	parser.add_argument('--lamda2', type = float, default = 0.5, metavar = 'LAM',
						help = 'weight on performance of agent 2 (default: 0.5)')
	parser.add_argument('--state-dim', type = int, default = 4, metavar = 'DIM',
						help = 'dimension of the square matrix in the state input (default: 4)')
	parser.add_argument('--action-dim', type = int, default = 5, metavar = 'DIM',
						help = 'number of actions (default: 5 - UP, DOWN, LEFT, RIGHT, NOOP)')
	parser.add_argument('--num-players', type = int, default = 2, metavar = 'N',
						help = 'number of players in the game - one color is given to each player (default: 2)')
	parser.add_argument('--num-coins', type = int, default = 1, metavar = 'N',
						help = 'number of coins in the game (default: 1)')
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
	parser.add_argument('--activation-function', type = str, default = 'relu',
						help = 'which activation function to use (relu or tanh, default: relu)')
	parser.add_argument('--policy', type = str, default = 'mlp',
						help = 'which type of policy to use (lstm or mlp, default: mlp)')
	parser.add_argument('--hidden-dim', type = int, default = 32, metavar = 'DIM',
						help = 'number of features in the hidden state of the LSTM (default: 32)')
	parser.add_argument('--conv', action = 'store_true', default = False,
						help = 'use convolutions or not (default: False)')
	parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'GAMMA',
						help = 'discount factor (default: 0.99)')
	parser.add_argument('--tau', type = float, default = 0.95, metavar = 'TAU',
						help = 'GAE factor (default: 0.95)')
	parser.add_argument('--logs', action = 'store_true', default = False,
						help = 'write data to logs or not (default: False')
	parser.add_argument('--lola', action = 'store_true', default = False,
						help = 'whether to use LOLA or not (default: False')

	args = parser.parse_args()

	use_cuda = args.cuda and torch.cuda.is_available()
	#print(use_cuda, args.cuda, torch.cuda.is_available())
	if args.cuda and not torch.cuda.is_available():
		raise Exception('torch.cuda is not available!')
	#if args.cuda and not torch.cuda.is_available():
	#	print('CUDA CONFLICT')
	#	raise 'error'
	device = torch.device("cuda" if use_cuda else "cpu")

	env_location = '../tensorboard/coin_game'
	experiment_name = '/mavpg/run_' + str(args.run_num)#+'_mavpg_lr='+str(args.lr_p)+'_'+args.activation_function+'_'+args.policy
	job_path = r'../coin_game/job_coin.yaml'
	if os.path.exists(env_location + experiment_name):
		raise Exception('Run index {} already exists!'.format(args.run_num))
	os.makedirs(env_location + experiment_name)
	info_str = env_location + experiment_name + '/info_mavpg_run_' + str(args.run_num) +'.txt'
	with open(info_str, 'a') as info_ptr:
		info_ptr.write('This is run number {} of MAVPG. The hyperparameters are:\n\n'.format(args.run_num))
		info_ptr.write('Running torch version {}\n\n'.format(torch.__version__))
		for arg in vars(args):
			info_ptr.write(str(arg) + ' ' + str(getattr(args, arg)) + '\n')
		#info_ptr.write('\nuse_cuda == args.cuda and torch.cuda.is_available() == {}\n'.format(use_cuda))
		with open(job_path) as f_y:
			d_y = yaml.full_load(f_y)
			info_ptr.write('\nResources: {}'.format(d_y['spec']['containers'][0]['resources']))
		if use_cuda:
			current_device = torch.cuda.current_device()
			info_ptr.write('\ntorch.cuda.device(current_device): {}'.format(torch.cuda.device(current_device)))
			info_ptr.write('\ntorch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
			info_ptr.write('\ntorch.cuda.get_device_name(current_device): {}\n\n'.format(torch.cuda.get_device_name(current_device)))
	
	model_mavpg = env_location + experiment_name + '/model'
	data_mavpg = env_location + experiment_name + '/data'

	if not os.path.exists(model_mavpg):
		os.makedirs(model_mavpg)
	if not os.path.exists(data_mavpg):
		os.makedirs(data_mavpg)

	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

	if args.num_players > 2:
		env = coin_game_Np(args.max_steps, args.state_dim, args.action_dim, args.num_players, args.num_coins)
	else:
		env = coin_game_2p(args.max_steps, args.state_dim)

	if args.policy == 'mlp':
		p1 = MLP_policy(args.state_dim, args.num_coins*args.num_players+args.num_players, args.action_dim, args.conv, args.activation_function).to(device)
		p2 = MLP_policy(args.state_dim, args.num_coins*args.num_players+args.num_players, args.action_dim, args.conv, args.activation_function).to(device)
	elif args.policy == 'lstm':
		p1 = LSTM_policy(args.state_dim, args.num_coins*args.num_players+args.num_players, args.action_dim, args.hidden_dim, args.conv, args.activation_function).to(device)
		p2 = LSTM_policy(args.state_dim, args.num_coins*args.num_players+args.num_players, args.action_dim, args.hidden_dim, args.conv, args.activation_function).to(device)
	else:
		raise Exception('Policy type not recognized')

	q1 = MLP_value(args.state_dim, args.num_coins*args.num_players+args.num_players, args.conv, args.activation_function).to(device)
	q2 = MLP_value(args.state_dim, args.num_coins*args.num_players+args.num_players, args.conv, args.activation_function).to(device)

	with open(info_str, 'a') as info_ptr:
		info_ptr.write('Policy1 architecture:\n{}\n'.format(next(p1.named_modules())))
		info_ptr.write('Policy2 architecture:\n{}\n'.format(next(p2.named_modules())))
		info_ptr.write('Value1 architecture:\n{}\n'.format(next(q1.named_modules())))
		info_ptr.write('Value2 architecture:\n{}\n'.format(next(q2.named_modules())))

	if args.lola:
		policy_optim = LOLA(p1.parameters(), p2.parameters(), lr = args.lr_p, device = device)
	else:
		if not args.rms:
			policy_optim = CGD(p1.parameters(), p2.parameters(), lr = args.lr_p, device = device)
		else:
			policy_optim = RCGD(p1.parameters(), p2.parameters(), lr = args.lr_p, beta = args.beta, eps = args.eps, device = device)
	optim_q1 = torch.optim.Adam(q1.parameters(), lr = args.lr_q1)
	optim_q2 = torch.optim.Adam(q2.parameters(), lr = args.lr_q2)

	if args.logs:
		logs_file_path = env_location + experiment_name + '/coin_mavpg_run_' + str(args.run_num) + '.txt' 
		f_ptr = open(logs_file_path, 'a')
		s = 'This is run number {} of MAVPG\n\n'.format(args.run_num)
		f_ptr.write(s)
	
	if args.tensorboard:
		writer = SummaryWriter(data_mavpg)

	n_pR_cR = n_pR_cB = n_pR_cR_interval = n_pR_cB_interval = 0
	n_pB_cR = n_pB_cB = n_pB_cR_interval = n_pB_cB_interval = 0
	n_draw = n_draw_interval = 0

	avg_rew_1 = []
	avg_rew_2 = []
	arr_pR_cR = []
	arr_pR_cB = []
	arr_pB_cR = []
	arr_pB_cB = []
	arr_draw = []

	total_t_p_opt = 0
	total_t_q_opt = 0
	total_t_play = 0
	avg_t_p_opt = 0
	avg_t_q_opt = 0
	avg_t_play = 0
	t_episodes = []
	num_episodes = 0
	start = time.time()
	for epoch in range(args.n_epochs):
		state_1_2 = []
		if args.policy == 'lstm':
			hidden_state_1 = []
			cell_state_1 = []
			hidden_state_2 = []
			cell_state_2 = []
		action_1_2 = []
		reward_1 = []
		reward_2 = []
		state, rewards, done, info = env.reset()
		time_in_epoch = 0
		avg_time_in_epoch = 0
		n_rr_e = n_rb_e = n_br_e = n_bb_e = n_d_e = 0

		start_play = time.time()
		for eps in range(args.n_eps):
			time_in_episode = 0
			num_episodes += 1
			state, rewards, done, info = env.reset()
			if args.policy == 'lstm':
				hx_1 = torch.zeros(1, args.hidden_dim).to(device)
				cx_1 = torch.zeros(1, args.hidden_dim).to(device)
				hx_2 = torch.zeros(1, args.hidden_dim).to(device)
				cx_2 = torch.zeros(1, args.hidden_dim).to(device)
			while done == False:
				time_in_episode += 1
				if args.policy == 'lstm':
					hidden_state_1.append(hx_1)
					cell_state_1.append(cx_1)
					hidden_state_2.append(hx_2)
					cell_state_2.append(cx_2)
					pi1_a_s, hx_1, cx_1 = p1((FloatTensor(state).unsqueeze(0)).to(device), hx_1, cx_1)
					pi2_a_s, hx_2, cx_2 = p2((FloatTensor(state).unsqueeze(0)).to(device), hx_2, hx_2)
				else:
					pi1_a_s = p1((FloatTensor(state).unsqueeze(0)).to(device))
					pi2_a_s = p2((FloatTensor(state).unsqueeze(0)).to(device))
				
				dist1 = Categorical(pi1_a_s)
				action1 = dist1.sample()
				
				dist2 = Categorical(pi2_a_s)
				action2 = dist2.sample()
				
				action = np.array([action1.cpu(), action2.cpu()])

				state_1_2.append(FloatTensor(state))
				action_1_2.append(FloatTensor(action))

				state, rewards, done, info = env.step(action)
				reward_1.append(torch.FloatTensor([rewards[0]]))
				reward_2.append(torch.FloatTensor([rewards[1]]))

				if done == True:
					break
			t_episodes.append(time_in_episode)
			s = 'Epoch number: {}, timesteps in episode {}: {}, Done reached\n'.format(epoch, eps, time_in_episode)
			#print(s)
			if args.logs:
				f_ptr.write(s)
			if info[0] == 'RED':
				s = '{} player (player1) got {} coin in episode {} of epoch {}\n'.format(info[0], info[1], eps, epoch)
				# print(s)
				if args.logs:
					f_ptr.write(s)
				if info[1] == 'RED':
					n_pR_cR += 1
					n_pR_cR_interval += 1
					n_rr_e += 1
					#arr_pR_cR.append(n_pR_cR/num_episodes)
				elif info[1] == 'BLUE':
					n_pR_cB += 1
					n_pR_cB_interval += 1
					n_rb_e += 1
					#arr_pR_cB.append(n_pR_cB/num_episodes)
			elif info[0] == 'BLUE':
				s = '{} player (player2) got {} coin in episode {} of epoch {}\n'.format(info[0], info[1], eps, epoch)
				# print(s)
				if args.logs:
					f_ptr.write(s)
				if info[1] == 'RED':
					n_pB_cR += 1
					n_pB_cR_interval += 1
					n_br_e += 1
					#arr_pB_cR.append(n_pB_cR/num_episodes)
				elif info[1] == 'BLUE':
					n_pB_cB += 1
					n_pB_cB_interval += 1
					n_bb_e += 1
					#arr_pB_cB.append(n_pB_cB/num_episodes)
			elif info[0] == 'NONE':
				s = 'NONE of the players got the coin in episode {} of epoch {}\n'.format(eps, epoch)
				n_draw += 1
				n_draw_interval += 1
				n_d_e += 1
				#arr_draw.append(n_draw/num_episodes)
				if args.logs:
					f_ptr.write(s)
			arr_pR_cR.append(n_pR_cR/num_episodes)
			arr_pR_cB.append(n_pR_cB/num_episodes)
			arr_pB_cR.append(n_pB_cR/num_episodes)
			arr_pB_cB.append(n_pB_cB/num_episodes)
			arr_draw.append(n_draw/num_episodes)
			time_in_epoch += time_in_episode
			s = 'Total timesteps in episode {} of epoch {}: {}\n\n'.format(eps, epoch, time_in_episode)
			#print(s)
			if args.logs:
				f_ptr.write(s)
			if args.tensorboard:
				writer.add_scalar('Total timesteps in episode', time_in_episode, epoch * args.n_eps + eps)
			#avg_time_in_epoch = time_in_episode/n_eps #(avg_time_in_eps_per_epoch * eps + time_in_episode)/(eps + 1)
			#writer.add_scalar('Average time in this epoch', avg_time_in_epoch, n_epochs)
		end_play = time.time()
		
		t_play = end_play - start_play
		total_t_play += t_play
		avg_t_play = total_t_play/(epoch + 1)
		
		avg_time_in_epoch = time_in_epoch/args.n_eps
		s = 'Total number of episodes in epoch {}: {}\n'.format(epoch, args.n_eps)
		if args.logs:
			f_ptr.write(s)
		s = 'Total timesteps in epoch {}: {}\n'.format(epoch, time_in_epoch)
		#print(s)
		if args.logs:
			f_ptr.write(s)
		s = 'Average timesteps in epoch {}: {}\n'.format(epoch, avg_time_in_epoch)
		#print(s)
		if args.logs:
			f_ptr.write(s)

		val1 = q1(torch.stack(state_1_2).to(device))
		# v1 is (1, n)
		v1 = val1.detach().squeeze().cpu()
		v1 = torch.cat((v1, torch.FloatTensor([0])))
		r1 = torch.tensor(reward_1)

		# advantage1 is on gpu and detached
		# advantage1 is (1, n)
		advantage1 = get_advantage(v1, r1, args.gamma, args.tau).to(device)

		val2 = q2(torch.stack(state_1_2).to(device))
		v2 = val2.detach().squeeze().cpu()
		v2 = torch.cat((v2, torch.FloatTensor([0])))
		r2 = torch.tensor(reward_2)
		
		advantage2 = get_advantage(v2, r2, args.gamma, args.tau).to(device)

		avg_rew_1.append(sum(reward_1)/len(reward_1))
		avg_rew_2.append(sum(reward_2)/len(reward_2))

		s = '\nAverage reward in this epoch for Agent1 (RED): {}\nAverage reward in this epoch for Agent2 (BLUE): {}\n'.format(sum(reward_1)/len(reward_1), sum(reward_2)/len(reward_2))
		# print(s)
		if args.logs:
			f_ptr.write(s)		
		s = '\nMean advantage for Agent1 (RED) (eta_new - eta_old): {}\nMean advantage for Agent2 (BLUE) -(eta_new - eta_old): {}\n'.format(advantage1.mean().cpu(), advantage2.mean().cpu())
		# print(s)
		if args.logs:
			f_ptr.write(s)

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

		action_both = torch.stack(action_1_2).to(device)

		if args.policy == 'lstm':
			pi1_a_s, _, _ = p1(torch.stack(state_1_2).to(device), torch.cat(hidden_state_1).to(device), torch.cat(cell_state_1).to(device))
			pi2_a_s, _, _ = p2(torch.stack(state_1_2).to(device), torch.cat(hidden_state_2).to(device), torch.cat(cell_state_2).to(device))
		else:
			pi1_a_s = p1(torch.stack(state_1_2).to(device))
			pi2_a_s = p2(torch.stack(state_1_2).to(device))

		dist1 = Categorical(pi1_a_s)
		log_prob1 = dist1.log_prob(action_both[:,0])

		dist2 = Categorical(pi2_a_s)
		log_prob2 = dist2.log_prob(action_both[:,1])

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

		# f -> trying to maximize both agent's performance
		lp_x = args.lamda1 * lp_x_1 + args.lamda2 * lp_x_2
		# g -> trying to maximize both agent's performance
		lp_y = args.lamda1 * lp_y_1 + args.lamda2 * lp_y_2

		mh1_1 = (log_prob1 * log_prob2 * (-advantage1)).mean()
		mh2_1 = (log_prob1[1:] * cum_log_prob2 * (-advantage1[1:])).mean()
		#mh2_1 = mh2_1.sum()/(mh2_1.size(0)-args.n_eps+1)
		mh3_1 = (log_prob2[1:] * cum_log_prob1 * (-advantage1[1:])).mean()
		#mh3_1 = mh3_1.sum()/(mh3_1.size(0)-args.n_eps+1)

		mh1_2 = (log_prob1 * log_prob2 * (-advantage2)).mean()
		mh2_2 = (log_prob1[1:] * cum_log_prob2 * (-advantage2[1:])).mean()
		#mh2_2 = mh2_2.sum()/(mh2_2.size(0)-args.n_eps+1)
		mh3_2 = (log_prob2[1:] * cum_log_prob1 * (-advantage2[1:])).mean()
		#mh3_2 = mh3_2.sum()/(mh3_2.size(0)-args.n_eps+1)
			
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

		s = '\nRED player got {} RED coins in epoch {} | RED player got {} BLUE coins in epoch {}\nBLUE player got {} RED coins in epoch {} | BLUE player got {} BLUE coins in epoch {}\n'.format(n_rr_e, epoch, n_rb_e, epoch, n_br_e, epoch, n_bb_e, epoch)
		if args.logs:
			f_ptr.write(s)
		s = 'Number of draws in epoch {}: {}\n'.format(epoch, n_d_e)
		if args.logs:
			f_ptr.write(s)
		if (epoch + 1) % args.interval == 0:
			s = '\nRED player got {} RED coins in last {} epochs | RED player got {} BLUE coins in last {} epochs\n'.format(n_pR_cR_interval, args.interval, n_pR_cB_interval, args.interval)
			if args.logs:
				f_ptr.write(s)
			s = 'BLUE player got {} RED coins in last {} epochs | BLUE player got {} BLUE coins in last {} epochs\n'.format(n_pB_cR_interval, args.interval, n_pB_cB_interval, args.interval)
			if args.logs:
				f_ptr.write(s)
			s = 'Number of draws in last {} epochs: {}'.format(args.interval, n_draw_interval)
			if args.logs:
				f_ptr.write(s)
			s = 'RED-player-RED-coin rate in last {} epochs: {} | RED-player-BLUE-coin rate in last {} epochs: {}\n'.format(args.interval, n_pR_cR_interval/(args.interval*args.n_eps), args.interval, n_pR_cB_interval/(args.interval*args.n_eps))
			if args.logs:
				f_ptr.write(s)
			s = 'BLUE-player-RED-coin rate in last {} epochs: {} | BLUE-player-BLUE-coin rate in last {} epochs: {}\n'.format(args.interval, n_pB_cR_interval/(args.interval*args.n_eps), args.interval, n_pB_cB_interval/(args.interval*args.n_eps))
			if args.logs:
				f_ptr.write(s)
			s = 'DRAW rate in last {} epochs: {}\n'.format(args.interval, n_draw_interval/(args.interval*args.n_eps))
			if args.logs:
				f_ptr.write(s)
			if args.tensorboard:
				writer.add_scalar('Coin take rate for RED player last {} epochs/RED coin'.format(args.interval), (n_pR_cR_interval)/(args.interval*args.n_eps), epoch)
				writer.add_scalar('Coin take rate for RED player last {} epochs/BLUE coin'.format(args.interval), (n_pR_cB_interval)/(args.interval*args.n_eps), epoch)
				writer.add_scalar('Coin take rate for BLUE player last {} epochs/RED coin'.format(args.interval), (n_pB_cR_interval)/(args.interval*args.n_eps), epoch)
				writer.add_scalar('Coin take rate for BLUE player last {} epochs/BLUE coin'.format(args.interval), (n_pB_cB_interval)/(args.interval*args.n_eps), epoch)
				writer.add_scalar('Draw rate last {} epochs'.format(args.interval), (n_draw_interval)/(args.interval*args.n_eps), epoch)
			n_pR_cR_interval = n_pR_cB_interval = 0
			n_pB_cR_interval = n_pB_cB_interval = 0
			n_draw_interval = 0

		tot_games = (epoch + 1) * args.n_eps
		s = '\nRED player got {} RED coins till now | RED player got {} BLUE coins till now\n'.format(n_pR_cR, n_pR_cB)
		if args.logs:
			f_ptr.write(s)
		s = 'BLUE player got {} RED coins till now | BLUE player got {} BLUE coins till now\n'.format(n_pB_cR, n_pB_cB)
		if args.logs:
			f_ptr.write(s)
		s = 'Number of draws till now: {}\n\n'.format(n_draw)
		if args.logs:
			f_ptr.write(s)
		s = 'RED-player-RED-coin rate till now: {} | RED-player-BLUE-coin rate till now: {}\n'.format(n_pR_cR/tot_games, n_pR_cB/tot_games)
		if args.logs:
			f_ptr.write(s)
		s = 'BLUE-player-RED-coin rate till now: {} | BLUE-player-BLUE-coin rate till now: {}\n'.format(n_pB_cR/tot_games, n_pB_cB/tot_games)
		if args.logs:
			f_ptr.write(s)
		s = 'DRAW rate till now: {}\n'.format(n_draw/tot_games)
		if args.logs:
			f_ptr.write(s)
		s = '\nTime for game play in this epoch: {} seconds\n'.format(t_play)
		if args.logs:
			f_ptr.write(s)
		s = '\nTime for policy optimization in this epoch: {} seconds\nTime for critic optimization in this epoch: {}seconds\n\n'.format(t_p_opt, t_q_opt)
		if args.logs:
			f_ptr.write(s+'##################################################################################################################'+'\n\n')

		if args.tensorboard:
			writer.add_scalar('Total timesteps in epoch', time_in_epoch, epoch)
			writer.add_scalar('Average timesteps in epoch', avg_time_in_epoch, epoch)

			writer.add_scalar('Average reward in the epoch/Agent1', sum(reward_1)/len(reward_1), epoch)
			writer.add_scalar('Average reward in the epoch/Agent2', sum(reward_2)/len(reward_2), epoch)
			writer.add_scalar('Mean advantage in the epoch/Agent1', advantage1.mean().cpu(), epoch)
			writer.add_scalar('Mean advantage in the epoch/Agent2', advantage2.mean().cpu(), epoch)

			writer.add_scalar('Entropy/Agent1', dist1.entropy().mean().detach().cpu(), epoch)
			writer.add_scalar('Entropy/Agent2', dist2.entropy().mean().detach().cpu(), epoch)

			writer.add_scalar('Coin take rate for RED player/RED coin', (n_pR_cR)/tot_games, epoch)
			writer.add_scalar('Coin take rate for RED player/BLUE coin', (n_pR_cB)/tot_games, epoch)
			writer.add_scalar('Coin take rate for BLUE player/RED coin', (n_pB_cR)/tot_games, epoch)
			writer.add_scalar('Coin take rate for BLUE player/BLUE coin', (n_pB_cB)/tot_games, epoch)
			writer.add_scalar('Draw rate', (n_draw)/tot_games, epoch)
			
			writer.add_scalar('Time/avg_t_p_opt', avg_t_p_opt, epoch)
			writer.add_scalar('Time/avg_t_q_opt', avg_t_q_opt, epoch)
			writer.add_scalar('Time/avg_t_play', avg_t_play, epoch)
			writer.add_scalar('Time/t_p_opt', t_p_opt, epoch)
			writer.add_scalar('Time/t_q_opt', t_q_opt, epoch)
			writer.add_scalar('Time/t_play', t_play, epoch)

		if args.save_model and epoch == args.n_epochs - 1:#epoch % args.interval == 0:
			#print(epoch)
			torch.save(p1.state_dict(), model_mavpg + '/policy_agent1_' + str(epoch) + ".pth")
			torch.save(p2.state_dict(), model_mavpg + '/policy_agent2_' + str(epoch) + ".pth")
			torch.save(q1.state_dict(), model_mavpg + '/value_agent1_' + str(epoch) + ".pth")
			torch.save(q2.state_dict(), model_mavpg + '/value_agent2_' + str(epoch) + ".pth")
	
	end = time.time()
	total_time = end - start
	if args.logs:
		s = 'Total time taken: {} seconds\nTotal time for game play only: {} seconds\nTotal time for policy optimization steps only: {} seconds\nTotal time for critic optimization steps only: {} seconds\n'.format(total_time, total_t_play, total_t_p_opt, total_t_q_opt)
		f_ptr.write(s)
		s = 'Average time for policy optimization steps only: {} seconds\nAverage time for game play only: {} seconds\nAverage time for critic optimization steps only: {} seconds\n\n'.format(avg_t_p_opt, avg_t_play, avg_t_q_opt)
		f_ptr.write(s)
		np_file = env_location + experiment_name + '/coin_mavpg_run_' + str(args.run_num) + '_lr=' + str(args.lr_p) + '_stuff.npz'
		with open(np_file, 'wb') as np_f:
			np.savez(np_f, avg_rew_1 = np.array(avg_rew_1), avg_rew_2 = np.array(avg_rew_2), n_pR_cR = np.array(n_pR_cR),\
			n_pR_cB = np.array(n_pR_cB), n_pB_cR = np.array(n_pB_cR), n_pB_cB = np.array(n_pB_cB), n_draw = np.array(n_draw), arr_pR_cR = np.array(arr_pR_cR), \
			arr_pR_cB = np.array(arr_pR_cB), arr_pB_cR = np.array(arr_pB_cR), arr_pB_cB = np.array(arr_pB_cB), arr_draw = np.array(arr_draw),\
			total_time = np.array(total_time), total_play_time = np.array(total_t_play), total_time_p_opt = np.array(total_t_p_opt), total_time_q_opt = np.array(total_t_q_opt),\
			avg_time_p_opt = np.array(avg_t_p_opt), avg_play_time = np.array(avg_t_play), avg_time_q_opt = np.array(avg_t_q_opt), time_in_episodes = np.array(t_episodes))

	fig = plt.figure(figsize = (15,15))
	ax = plt.subplot()
	ax.clear()
	fig.suptitle('Coin picking rates for RED and BLUE agents', fontsize = 25)
	plt.xlabel(r'$Number\ of\ episodes\ (Num\ epochs\ =\ Total\ num\ eps/num\ eps\ per\ epoch)$', fontsize = 20)
	plt.ylabel(r'$Coin\ picking\ rate$', fontsize = 20)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	ax.plot(np.array(arr_pR_cR), label = 'MAVPG, RED player picks RED coin, lr = {}'.format(args.lr_p))
	ax.plot(np.array(arr_pR_cB), label = 'MAVPG, RED player picks BLUE coin, lr = {}'.format(args.lr_p))
	ax.plot(np.array(arr_pB_cR), label = 'MAVPG, BLUE player picks RED coin, lr = {}'.format(args.lr_p))
	ax.plot(np.array(arr_pB_cB), label = 'MAVPG, BLUE player picks BLUE coin, lr = {}'.format(args.lr_p))
	ax.plot(np.array(arr_draw), label = 'MAVPG, DRAW, lr = {}'.format(args.lr_p))
	plt.legend(loc = 'upper right')
	plt.grid()
	plt.savefig(env_location + experiment_name + '/coin_mavpg_run_' + str(args.run_num) + '_lr=' + str(args.lr_p) + '_pick_rate.png')

	fig = plt.figure(figsize = (15,15))
	ax = plt.subplot()
	ax.clear()
	fig.suptitle('Avg reward per epoch for Agent1 (RED) & Agent2 (BLUE) v/s #epochs/iterations', fontsize = 25)
	plt.xlabel(r'$Epochs/Iterations$', fontsize = 20)
	plt.ylabel(r'$Average\ reward\ per\ epoch$', fontsize = 20)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	ax.plot(np.array(avg_rew_1), label = 'MAVPG, Avg rew per epoch Ag1 (RED), lr = {}'.format(args.lr_p))
	ax.plot(np.array(avg_rew_2), label = 'MAVPG, Avg rew per epoch Ag2 (BLUE), lr = {}'.format(args.lr_p))
	plt.legend(loc = 'upper right')
	plt.grid()
	plt.savefig(env_location + experiment_name + '/coin_mavpg_run_' + str(args.run_num) + '_lr=' + str(args.lr_p) + '_avg_rew_per_epoch.png')

if __name__ == '__main__':
	main()
