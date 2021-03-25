import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from coin_game_batch import coin_game_batch
from networks_lola_dice_coin import LSTM_policy, critic
import copy

def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
    def __init__(self):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []
        self.mask = torch.ones((args.batch_size, args.max_steps)).to(device)

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self):
        self_logprobs = torch.stack(self.self_logprobs, dim=1).to(device)
        other_logprobs = torch.stack(self.other_logprobs, dim=1).to(device)
        self.values = torch.stack(self.values, dim=1).to(device)
        self.rewards = torch.stack(self.rewards, dim=1).to(device)

        discount_mask = torch.cumprod(torch.ones(self.rewards.size()) * args.gamma, dim = 1)/args.gamma
        discounted_rewards = discount_mask * self.rewards
        discounted_values = discount_mask * self.values

        stochastic_nodes = self_logprobs + other_logprobs
        dependencies = torch.cumsum(stochastic_nodes, dim = 1)
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards * self.mask, dim = 1))

        if args.use_baseline:
            baseline = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values * self.mask, dim = 1))
            dice_objective += baseline
        return -dice_objective

    def critic_loss(self):
        values = torch.cat((self.values * self.mask, torch.zeros(args.batch_size, 1).to(device)), dim = 1)
        c_loss = torch.mean(torch.pow(self.rewards * self.mask + values[:,1:] * args.gamma - values[:,:-1], 2))
        return c_loss

def act(states_batch, hx, cx, policy, critic):
    probs, hx, cx = policy(states_batch, hx, cx).to(device)
    dist = Categorical(probs)
    actions_batch = dist.sample()
    logprob_actions_batch = dist.log_prob(actions_batch)
    values_batch = critic(states_batch).to(device)
    return actions_batch.cpu().numpy().astype(int), logprob_actions_batch, values_batch, hx, cx

def get_gradient():
    pass

def set_hidden():
    h0 = torch.zeros((args.batch_size, args.hidden_dim)).to(device)
    c0 = torch.zeros((args.batch_size, args.hidden_dim)).to(device)
    return h0, c0

def step(policy1, critic1, policy2, critic2):
    score1 = 0
    score2 = 0
    (s1, s2), _, done = env.reset()
    hx1, cx1 = set_hidden()
    hx2, cx2 = set_hidden()
    while not done.all():
        a1, _, _, hx1, cx1 = act(torch.from_numpy(s1), hx1, cx1, policy1, critic1)
        a2, _, _, hx2, cx2 = act(torch.from_numpy(s2), hx2, cx2, policy2, critic2)
        (s1, s2), (r1, r2), done, _ = env.step((a1, a2))
        score1 += np.sum(r1)
        score2 += np.sum(r2)
    info = env.get_info()
    score1 /= args.batch_size
    score2 /= args.batch_size
    return score1, score2, info

class Agent():
    def __init__(self):
        self.policy = LSTM_policy(args.num_channels, args.grid_size, args.hidden_dim, env.n_actions).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = args.lr_outer)
        self.critic = critic(args.num_channels, args.grid_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = args.lr_val)
    
    def policy_update(self, objective):
        self.policy_optimizer.zero_grad()
        objective.backward(retain_graph = True)
        self.policy_optimizer.step()

    def critic_update(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def mask_update(self, done, timestep, mask):
        for i in range(args.batch_size):
            if done[i]:
                mask[i][timestep:] = 0.
    
    def in_lookahead(self, other_policy, other_policy_optimizer, other_critic):
        (s1, s2), _, done = env.reset()
        hx1, cx1 = set_hidden()
        hx2, cx2 = set_hidden()
        other_memory = Memory()
        timestep = 0
        n_dones = 0
        while not done.all():
            timestep += 1
            a1, lp1, v1, hx1, cx1 = act(torch.from_numpy(s1), hx1, cx1, self.policy, self.critic)
            a2, lp2, v2, hx2, cx2 = act(torch.from_numpy(s2), hx2, cx2, other_policy, other_critic)
            (s1, s2), (r1, r2), done = env.step((a1, a2))
            if np.sum(done) > n_dones:
                self.mask_update(done, timestep, other_memory.mask)
                n_dones = np.sum(done)
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float().to(device))
        other_dice_objective = other_memory.dice_objective()
        other_dice_objective.backward(retain_graph = True)
        other_policy_optimizer.step()

    def out_lookahead(self, other_policy, other_critic):
        (s1, s2), _, done = env.reset()
        hx1, cx1 = set_hidden()
        hx2, cx2 = set_hidden()
        memory = Memory()
        timestep = 0
        n_dones = 0
        while not done.all():
            timestep += 1
            a1, lp1, v1, hx1, cx1 = act(torch.from_numpy(s1), hx1, cx1, self.policy, self.critic)
            a2, lp2, v2, hx2, cx2 = act(torch.from_numpy(s2), hx2, cx2, other_policy, other_critic)
            (s1, s2), (r1, r2), done = env.step((a1, a2))
            if np.sum(done) > n_dones:
                self.mask_update(done, timestep, memory.mask)
                n_dones = np.sum(done)
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float().to(device))
        dice_objective = memory.dice_objective()
        self.policy_update(dice_objective)
        critic_loss = memory.critic_loss()
        self.critic_update(critic_loss)

def play(agent1, agent2, lookahead):
    scores = []
    for itr in range(args.n_itrs):
        print('Starting iterations with number of lookaheads = {}'.format(lookahead))
        # create copies
        policy1_ = copy.deepcopy(agent1.policy)
        policy1_optimizer = torch.optim.SGD(policy1_.parameters(), lr = args.lr_inner)
        critic1_ = copy.deepcopy(agent1.critic)
        policy2_ = copy.deepcopy(agent2.policy)
        policy2_optimizer = torch.optim.SGD(policy2_.parameters(), lr = args.lr_inner)
        critic2_ = copy.deepcopy(agent2.critic)

        # inner imagination (imagine moving ahead in time)
        for k in range(lookahead):
            agent1.in_lookahead(policy2_, policy2_optimizer, critic2_)
            agent2.in_lookahead(policy1_, policy1_optimizer, critic1_)

        # actual parameter update
        agent1.out_lookahead(agent2.policy, agent2.critic)
        agent2.out_lookahead(agent1.policy, agent1.critic)

        # evaluate progress
        # score1 == average reward (averaged over the batch) for agent 1
        score1, score2, info = step(agent1.policy, agent1.critic, agent2.policy, agent2.critic)
        scores.append((score1, score2))
        if itr % 10 == 0:
            info = np.divide(info, args.batch_size)
            print('######################################################################')
            print('Stats averaged over a batch of {} with max steps {}'.format(args.batch_size, args.max_steps))
            print('Iteration {}'.format(itr))
            print('Prob RED picks RED coin: {}'.format(info[1]))
            print('Prob RED picks BLUE coin: {}'.format(info[2]))
            print('Prob BLUE picks RED coin: {}'.format(info[3]))
            print('Prob BLUE picks BLUE coin: {}'.format(info[4]))
            print('Prob NONE {}'.format(info[0]))
            print('Score RED: {}'.format(score1))
            print('Score BLUE: {}'.format(score2))
            print('######################################################################')
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='LOLA DICE COIN')
    parser.add_argument('--n-itrs', type=int, default=201)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr-inner', type=float, default=0.3)
    parser.add_argument('--lr-outer', type=float, default=0.2)
    parser.add_argument('--lr-val', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    #parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--use-baseline', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--grid-size', type=int, default=4)
    parser.add_argument('--n-lookaheads', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--num-channels', type=int, default=4)
    parser.add_argument('--cuda', action='store_true', default=False)

    global args
    args = parser.parse_args()
    global env
    env = coin_game_batch(args.max_steps, args.batch_size, args.num_channels, args.grid_size)

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not torch.cuda.is_available():
        raise Exception('torch.cuda is not available!')
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    for lookahead in range(args.n_lookaheads):
        torch.manual_seed(args.seed)
        scores = play(Agent(), Agent(), lookahead)

if __name__ == '__main__':
    main()