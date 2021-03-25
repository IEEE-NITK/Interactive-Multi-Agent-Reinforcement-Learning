import numpy as np

class rockpaperscissor():
    def __init__(self):
        super(rockpaperscissor, self).__init__()
        self.num_players = 2
        self.done = False
        self.reward = np.zeros(2)

    def step(self, action):
        if action[0] == 0:
            if action[1] == 0:
                rew = 0
            elif action[1] == 1:
                rew = -1
            else:
                rew = 1
        elif action[0] == 1:
            if action[1] == 0:
                rew = 1
            elif action[1] == 1:
                rew = 0
            else:
                rew = -1
        else:
            if action[1] == 0:
                rew = -1
            elif action[1] == 1:
                rew = 1
            else:
                rew = 0
        
        self.reward = np.array([rew, -rew])
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        self.reward = np.zeros(2)
        return 0, self.reward[0], self.reward[1], False, {}

if name == '__main__':
    env = rockpaperscissor()

    for i in range(10):
        state, _, _, _, _ = env.reset()
        action = (np.random.randint(3,size=3)).reshape(3)
        state, reward1, reward2, done, _ =  env.step(action)
        print(action, reward1, reward2)