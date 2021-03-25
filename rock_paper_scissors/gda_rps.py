# rps_gda_game - use tensorboard to monitor training, dump data in mat files
# torch.device = gpu
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
import numpy as np
from rock_paper_scissors.rps_game import rockpaperscissor
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from rock_paper_scissors.network import policy1, policy2
import time

env_location = '../tensorboard/rps_game/'#'tensorboard/lq_game2/15_04_b1000/'
experiment_name = 'gda_lr0_5/'
model_gda = env_location + experiment_name + 'model'#directory = '../' + folder_location + experiment_name + 'model'
data_gda = env_location + experiment_name + 'data'

if not os.path.exists(model_gda):
    os.makedirs(model_gda)
if not os.path.exists(data_gda):
    os.makedirs(data_gda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(data_gda)
p1 = policy1().to(device)
p2 = policy2().to(device)

"""for p in p1.parameters():
    print(p)

for name, data in p1.named_parameters():
    print(name, data.size())

for p in p2.parameters():
    print(p)

for name, data in p2.named_parameters():
    print(name, data.size())"""

#optim = CoPG(p1.parameters(), p2.parameters(), lr = 0.5, device = device) #0.0001

optim_1 = torch.optim.SGD(p1.parameters(), lr = 0.5)
optim_2 = torch.optim.SGD(p2.parameters(), lr = 0.5)

#sys.exit()

#q = critic(1).to(device)

"""for p in q.parameters():
  print(p)

for name, data in q.named_parameters():
    print(name, data.size())"""

#optim_q = torch.optim.Adam(q.parameters(), lr=0.01)


batch_size = 1#1000
num_episode = 1#2001
#horizon = 5
#adv_copg_1 = []
#adv_copg_2 = []
prob_r_1 = []
prob_p_1 = []
prob_s_1 = []
prob_r_2 = []
prob_p_2 = []
prob_s_2 = []

env = rockpaperscissor()

#f_ptr = open('/rps_gda_lr0_5.txt', 'a')

#sys.exit()

for t_eps in range(num_episode):
    mat_action = []
    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    # data_collection
    # collects num_trajectories = batch_size, each with num_timesteps = horizon
    state, _, _, _, _ = env.reset()
    for _ in range(batch_size): 
        dist1 = Categorical(p1())
        action1 = dist1.sample()

        dist2 = Categorical(p2())
        action2 = dist2.sample()
        state = np.array([0])
        mat_state1.append(torch.FloatTensor(state))
        mat_state2.append(torch.FloatTensor(state))
        #mat_action1.append(torch.FloatTensor(action1))
        #mat_action2.append(torch.FloatTensor(action2))
        action = np.array([action1, action2])
        mat_action.append(torch.FloatTensor([action1, action2]))
        #writer.add_scalar('State/State', state, i)
        state, reward1, reward2, done, _ = env.step(action)
            # state = torch.FloatTensor(state).reshape(1)
            #print(state,action1,action2,reward1)
            # print('a1', action_app[0], reward1, action1_prob)
            # print('a2', action_app[1], reward2, action2_prob)
            #print(state, reward, done)
        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))
        mat_done.append(torch.FloatTensor([1 - done]))
    #writer.add_scalar('Reward/zerosum', torch.mean(torch.stack(mat_reward1)), t_eps)

    action_both = torch.stack(mat_action)

    #writer.add_scalar('Mean/Agent1', dist1.mean, t_eps)
    #writer.add_scalar('Mean/agent2', dist2.mean, t_eps)
    
    #writer.add_scalar('Variance/Agent1', dist1.variance, t_eps)
    #writer.add_scalar('Variance/agent2', dist2.variance, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)

    # val1_p -> column vector
    val1_p = torch.stack(mat_reward1).to(device).transpose(0,1)#advantage_mat1.transpose(0,1).to(device)#torch.stack(mat_reward1)

    if val1_p.size(0)!=1:
        raise 'error'

    pi1 = p1()
    pi_a1_s = Categorical(pi1)
    log_probs1 = pi_a1_s.log_prob(action_both[:,0].to(device))

    pi2 = p2()
    pi_a2_s = Categorical(pi2)
    log_probs2 = pi_a2_s.log_prob(action_both[:,1].to(device))

    writer.add_scalar('Entropy/Agent1', pi_a1_s.entropy().data, t_eps)
    writer.add_scalar('Entropy/Agent2', pi_a2_s.entropy().data, t_eps)

    # element-wise product
    #objective = log_probs1*log_probs2*(val1_p)
    #if objective.size(0)!=1:
    #    raise 'error'

    #ob = objective.mean()

    #s_log_probs1 = log_probs1.clone() # otherwise it doesn't change values
    #s_log_probs2 = log_probs2.clone()

    #if horizon is 4, log_probs1.size =4 and for loop will go till [0,3]

    #mask = torch.stack(mat_done).to(device)

    #s_log_probs1[0] = 0
    #s_log_probs2[0] = 0

    #for i in range(1,log_probs1.size(0)):
        # interaction from 0 to i-1 step
     #   s_log_probs1[i] = torch.add(s_log_probs1[i - 1], log_probs1[i-1])# * mask[i-1]
     #   s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i-1])# * mask[i-1]

    #objective2 = s_log_probs1*log_probs2*(val1_p)
    #ob2 = objective2.mean()

    #objective3 = log_probs1*s_log_probs2*(val1_p)
    #ob3 = objective3.mean()

    # for j in p1.parameters():
    #     print(j)
    #print(advantage_mat1.mean(),advantage_mat2.mean())
    val2_p = -val1_p
    lp1 = - log_probs1*val1_p
    lp1=lp1.mean()
    optim_1.zero_grad()
    lp1.backward()
    optim_1.step()

    lp2 = - log_probs2*val2_p
    lp2=lp2.mean()
    optim_2.zero_grad()
    lp2.backward()
    optim_2.step()
    
    #optim.step(ob, lp1, lp2)

    writer.add_scalar('Prob_rock/Agent1', pi1.data[0], t_eps)
    writer.add_scalar('Prob_paper/Agent1', pi1.data[1], t_eps)
    writer.add_scalar('Prob_scissor/Agent1', pi1.data[2], t_eps)
    writer.add_scalar('Prob_rock/Agent2', pi2.data[0], t_eps)
    writer.add_scalar('Prob_paper/Agent2', pi2.data[1], t_eps)
    writer.add_scalar('Prob_scissor/Agent2', pi2.data[2], t_eps)
    
    prob_r_1.append(pi1.data[0])
    prob_p_1.append(pi1.data[1])
    prob_s_1.append(pi1.data[2])
    prob_r_2.append(pi2.data[0])
    prob_p_2.append(pi2.data[1])
    prob_s_2.append(pi2.data[2])
    
    print('t_eps: ', t_eps)
    print('Probability of Rock/Agent1: {}, Probability of Paper/Agent1: {}, Probability of Scissor/Agent1: {}'.format(pi1.data[0], pi1.data[1], pi1.data[2]))
    print('Probability of Rock/Agent2: {}, Probability of Paper/Agent2: {}, Probability of Scissor/Agent2: {}'.format(pi2.data[0], pi2.data[1], pi2.data[2]))
    print('\n')

    #f_ptr.write('t_eps: {} \nProbability of Rock/Agent1: {}, Probability of Paper/Agent1: {}, Probability of Scissor/Agent1: {}\nProbability of Rock/Agent2: {}, Probability of Paper/Agent2: {}, Probability of Scissor/Agent2: {}\n\n'.format(t_eps, pi1.data[0], pi1.data[1], pi1.data[2], pi2.data[0], pi2.data[1], pi2.data[2]))

    if t_eps%200==0:
        #print(t_eps)
        torch.save(p1.state_dict(), model_gda + '/agent1_' + str(t_eps) + ".pth")
        torch.save(p2.state_dict(), model_gda + '/agent2_' + str(t_eps) + ".pth")
        #torch.save(q.state_dict(), model_copg + '/val_' + str(t_eps) + ".pth")