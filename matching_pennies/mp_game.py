import numpy as np

# matching pennies
class matching_pennies():
    def __init__(self):
        self.num_players = 2
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        if action[0] == 0:
            if action[1] == 0:
                rew = 1
            else:
                rew = -1
        else:
            if action[1] == 0:
                rew = -1
            else:
                rew = 1
        self.reward = np.array([rew, -rew])
        info = {}
        return 0, self.reward[0], self.reward[1], True, info
        
    def reset(self):
        self.reward = np.zeros(2)
        info = {}
        return 0, self.reward[0], self.reward[1], False, info

if __name__ == '__main__':
    env = matching_pennies()

    for i in range(10):
        state, _, _, _, _ = env.reset()
        action = (np.random.randint(2,size=2)).reshape(2)
        state, reward1, reward2, done, _ =  env.step(action)
        print(action, reward1, reward2)