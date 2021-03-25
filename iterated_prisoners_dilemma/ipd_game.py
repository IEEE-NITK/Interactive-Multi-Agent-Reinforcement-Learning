import numpy as np

class iterated_prisoner_dilemma():
    def __init__(self, repeated_steps):
        self.repeated_steps = repeated_steps
        self.payout_matrix = np.array([[-1, -3], [0, -2]])
        self.num_players = 2
        self.num_actions = 2
        self.num_states = 5
        self.num_steps = None

    def reset(self):
        self.num_steps = 0
        init_state = np.zeros(self.num_states)
        init_state[-1] = 1
        observations = [init_state, init_state]
        return observations, [None, None], False

    def step(self, actions):
        action1, action2 = actions
        self.num_steps += 1
        rewards = [self.payout_matrix[action1][action2], self.payout_matrix[action2][action1]]
        # the state at which the system arrives when take action1 and action2 => memory-1 2-agent MDP
        state = np.zeros(self.num_states)
        state[2 * action1 + action2] = 1
        observations = [state, state]
        done = (self.repeated_steps == self.num_steps)
        return observations, rewards, done
