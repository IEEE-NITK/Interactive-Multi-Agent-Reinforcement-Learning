import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
sys.path.append('/scratch/scratch1/videh/open_spiel')
sys.path.append('/scratch/scratch1/videh/open_spiel/build/python')
print(sys.path)
print('\n\n\n')
import numpy as np
from open_spiel.python import rl_environment
from torch.distributions import Categorical
from markov_soccer.networks_ms import policy
from markov_soccer.trans_dynamics import get_two_state

def compete(player1, player2, num_episodes, p1, p2, env, device, f_ptr):
	n_w_a = 0
	n_w_b = 0
	n_draw = 0

	if player1 == 'mavpg':
		p1_params = torch.load('../tensorboard/markov_soccer/mavpg_lr=0.01/model/policy_agent1_30000.pth')
	else:
		p1_params = torch.load('../tensorboard/markov_soccer/gda_lr=0.01_0.01/model/policy_agent1_30000.pth')

	if player2 == 'mavpg':
		p2_params = torch.load('../tensorboard/markov_soccer/mavpg_lr=0.01/model/policy_agent2_30000.pth')
	else:
		p2_params = torch.load('../tensorboard/markov_soccer/gda_lr=0.01_0.01/model/policy_agent2_30000.pth')

	p1.load_state_dict(p1_params)
	p2.load_state_dict(p2_params)

	win_mat = []
	originality_status = []
	ball_status = []
	game_lengths = []

	for t_eps in range(num_episodes):
		s = 't_eps: {}'.format(t_eps)
		f_ptr.write(s + '\n')
		mat_state1 = []
		mat_state2 = []
		mat_reward1 = []
		mat_reward2 = []

		time_step = env.reset()
		pre_a_status, pre_b_status, rel_state1, rel_state2 = get_two_state(time_step)
        
		temp_ball_stat = 0
		originality = 0
		time_in_episode = 0

		while time_step.last() == False:
			action_prob1 = p1(torch.FloatTensor(rel_state1)).to(device)
			dist1 = Categorical(action_prob1)
			action1 = dist1.sample()

			action_prob2 = p2(torch.FloatTensor(rel_state2)).to(device)
			dist2 = Categorical(action_prob2)
			action2 = dist2.sample()

			action = np.array([action1, action2])

			mat_state1.append(rel_state1)
			mat_state2.append(rel_state2)

			time_step = env.step(action)
			a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)

			if pre_a_status == 0 and a_status == 1:
                # A got the ball
				s = 'A got the ball'
				f_ptr.write(s + '\n')
				if temp_ball_stat % 2 == 0 and temp_ball_stat != 0:
                    # B had the ball, then A took it
					temp_ball_stat += 1
				else:
                    # A took the ball
					temp_ball_stat = 1
					originality = 1

			if pre_b_status == 0 and b_status == 1:
                # B got the ball
				s = 'B got the ball'
				f_ptr.write(s + '\n')
				if temp_ball_stat % 2 == 1:
                    # A had the ball, then B took it
					temp_ball_stat += 1
				else:
                    # B took the ball
					temp_ball_stat = 2
					originality = 2

			pre_a_status = a_status
			pre_b_status = b_status

			reward1 = time_step.rewards[0]
			reward2 = time_step.rewards[1]
			mat_reward1.append(reward1)
			mat_reward2.append(reward2)

			time_in_episode += 1

			if time_step.last() == True:
				s = 'time_in_episode {}'.format(time_in_episode)
				f_ptr.write(s + '\n')
				temp = 0
				if reward1 > 0:
					s = 'A won'
					f_ptr.write(s + '\n')
					temp = 1
					n_w_a += 1
				if reward2 > 0:
					s = 'B won'
					f_ptr.write(s + '\n')
					temp = 2
					n_w_b += 1
				if temp == 0:
					n_draw += 1

                # if ball was exchanged, yet no one scored goal
				if temp == 0 and temp_ball_stat != 0:
					temp_ball_stat = 0

				ball_status.append(temp_ball_stat)
				originality_status.append(originality)
				win_mat.append(temp)
				game_lengths.append(time_in_episode)
				s = 'n_win_A: {}, n_win_B: {}, n_draw: {}'.format(n_w_a, n_w_b, n_draw)
				f_ptr.write(s + '\n')
				s = '##########################################################'
				f_ptr.write(s + '\n')
				break
	return win_mat, originality_status, ball_status, game_lengths, n_w_a, n_w_b, n_draw

def main():
	# command line arguments: test settings
	parser = argparse.ArgumentParser(description = 'Markov soccer 1-vs-1 competitions')
	parser.add_argument('--n-eps', type = int, default = 100, metavar = 'N',
						help = 'number of 1-vs-1 episodes (default: 100)')
	parser.add_argument('--algo-p1', type = str, default = 'mavpg', metavar = 'STR',
						help = 'algorithm for player 1 (default: mavpg)')
	parser.add_argument('--algo-p2', type = str, default = 'mavpg', metavar = 'STR',
						help = 'algorithm for player 2 (default: mavpg)')
	parser.add_argument('--obs-dim', type = int, default = 12, metavar = 'DIM',
						help = 'dimension of observation space of each agent (default: 12)')
	parser.add_argument('--state-dim', type = int, default = 12, metavar = 'DIM',
						help = 'dimension of state space (default: 12)')
	parser.add_argument('--run-num', type = int, default = 0, metavar = 'NUM',
						help = 'index of experiment run (default: 0)')
	parser.add_argument('--cuda', action = 'store_true', default = False,
						help = 'use cuda or not (default: False)')

	args = parser.parse_args()

	f_ptr = open('../tensorboard/markov_soccer/competition1.txt', 'a')

	use_cuda = args.cuda and torch.cuda.is_available()
	f_ptr.write('##########################################################################################\n')
	s = 'use_cuda: {}, args.cuda: {}, torch.cuda.is_available(): {}'.format(use_cuda, args.cuda, torch.cuda.is_available())
	f_ptr.write(s + '\n\n')
	device = torch.device("cuda" if use_cuda else "cpu")

	game = 'markov_soccer'
	env = rl_environment.Environment(game)
	action_dim = env.action_spec()['num_actions']

	p1 = policy(args.obs_dim, action_dim).to(device)
	p2 = policy(args.obs_dim, action_dim).to(device)

	s = 'Playing {} for player 1 v/s {} for player 2'.format(args.algo_p1, args.algo_p2)
	f_ptr.write(s + '\n\n')
	win_mat, originality_status, ball_status, game_lengths, n_w_a, n_w_b, n_draw = compete(args.algo_p1, args.algo_p2, args.n_eps, p1, p2, env, device, f_ptr)
	s = '\nTotal number of times A wins: {} ({}), Total number of times B wins: {} ({}), Total number of times the game is draw: {} ({})'\
							.format(n_w_a, n_w_a/args.n_eps, n_w_b, n_w_b/args.n_eps, n_draw, n_draw/args.n_eps)
	f_ptr.write(s + '\n\n')

if __name__ == '__main__':
	main()