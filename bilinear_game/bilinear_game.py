import numpy as np
import random

class bilinear_game():
    def __init__(self):
        self.number_of_players = 2
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        self.reward = np.array((action[0]*action[1],action[0]*action[1]))
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False
        info = {}
        return 0, self.reward[0], self.reward[1], self.done, info