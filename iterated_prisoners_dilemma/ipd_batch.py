import numpy as np

class IPD_batch():
    def __init__(self, repeated_steps, batch_size):
        self.repeated_steps = repeated_steps
        self.batch_size = batch_size
        self.payout_matrix = np.array([[-1, -3], [0, -2]])
        # 0 == initial state, 1 == CC, 2 == CD, 3 == DC, 4 == DD
        self.states = np.array([[1, 2], [3, 4]])
        self.num_players = 2
        self.num_actions = 2
        self.num_states = 5
        self.num_steps = None

    def reset(self):
        self.num_steps = 0
        init_state = np.zeros(self.batch_size)
        #init_state[-1] = 1
        observations = [init_state, init_state]
        return observations, [None, None], False

    def step(self, actions):
        action1, action2 = actions
        self.num_steps += 1
        rewards = [self.payout_matrix[action1, action2], self.payout_matrix[action2, action1]]
        # action1 is like - np.array([0, 1, 1, 0, ....]) - these are actions for agent0 at time step t for whole batch 
        state1 = self.states[action1, action2]
        state2 = self.states[action2, action1]
        # state1 is like - np.array([2, 1, 4, 3, ...])
        observations = [state1, state2]
        done = (self.repeated_steps == self.num_steps)
        return observations, rewards, done
