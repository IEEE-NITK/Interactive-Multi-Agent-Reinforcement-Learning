import numpy as np

# prisoner's dilemma
class prisoner_dilemma():
    def __init__(self):
        self.num_players = 2
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        # 0 == cooperate
        # 1 == defect
        if action[0] == 0:
            if action[1] == 0:
                rew1 = -1
                rew2 = -1
            else:
                rew1 = -3
                rew2 = 0
        else:
            if action[1] == 0:
                rew1 = 0
                rew2 = -3
            else:
                rew1 = -2
                rew2 = -2
        self.reward = np.array([rew1, rew2])
        info = {}
        return 0, self.reward[0], self.reward[1], True, info
        
    def reset(self):
        self.reward = np.zeros(2)
        info = {}
        return 0, self.reward[0], self.reward[1], False, info

if __name__ == '__main__':
    env = prisoner_dilemma()

    for i in range(10):
        state, _, _, _, _ = env.reset()
        action = (np.random.randint(2,size=2)).reshape(2)
        state, reward1, reward2, done, _ =  env.step(action)
        print(action, reward1, reward2)