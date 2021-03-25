import numpy as np

# coin game for 2 players with only 1 coin of 2 possible colors
class coin_game_2p():
    def __init__(self, max_steps, matrix_size):
        self.matrix_size = matrix_size
        self.cells = self.matrix_size * self.matrix_size
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: NOOP
        self.num_actions = 5
        # 0: RED PLAYER, 1: BLUE PLAYER
        self.num_players = 2
        # 0: RED COIN, 1: BLUE COIN
        self.num_coins = 1
        # 0: RED COLOR, 1: BLUE COLOR
        self.num_colors = 2
        # 0: pos for RED coin, 1: pos for BLUE coin, 2: pos for RED player, 3: pos for BLUE player
        self.num_channels = 4
        self.max_steps = max_steps
        self.colors = {0: 'RED', 1: 'BLUE'}

    def reset(self):
        # position of coin and players
        while True:
            self.pos = np.random.randint(0, self.cells-1, self.num_players+self.num_coins)
            if self.pos[0] != self.pos[1] and self.pos[1] != self.pos[2] and self.pos[2] != self.pos[0]:
                break
        self.state = np.zeros((self.num_channels, self.matrix_size, self.matrix_size))
        # 0: RED COIN, 1: BLUE COIN
        self.coin_color = np.random.randint(0, self.num_colors, 1)[0]
        # randomly place the coin
        self.state[self.coin_color, int(self.pos[0]/self.matrix_size), int(self.pos[0]%self.matrix_size)] = 1
        # randomly place the players
        self.state[self.num_coins*self.num_colors+0, int(self.pos[1]/self.matrix_size), int(self.pos[1]%self.matrix_size)] = 1
        self.state[self.num_coins*self.num_colors+1, int(self.pos[2]/self.matrix_size), int(self.pos[2]%self.matrix_size)] = 1
        self.done = False
        # self.info[0] == player color, self.info[1] == coin color
        self.info = ['NONE', 'NONE']
        self.num_steps = 0
        # rewards[0]: RED PLAYER REWARD, rewards[1]: BLUE PLAYER REWARD
        self.rewards = [0, 0]
        return self.state, self.rewards, self.done, self.info

    def step(self, actions):
        self.num_steps += 1
        # player_pos[0]: RED PLAYER, player_pos[1]: BLUE PLAYER
        player_pos = [None, None]
        for i in range(self.num_players):
            # UP
            if actions[i] == 0:
                player_pos[i] = self.pos[self.num_coins+i] - self.matrix_size if self.pos[self.num_coins+i] - self.matrix_size >= 0 else self.pos[self.num_coins+i]
            # DOWN
            elif actions[i] == 1:
                player_pos[i] = self.pos[self.num_coins+i] + self.matrix_size if self.pos[self.num_coins+i] + self.matrix_size < self.cells else self.pos[self.num_coins+i]
            # LEFT
            elif actions[i] == 2:
                player_pos[i] = self.pos[self.num_coins+i] - 1 if self.pos[self.num_coins+i]%self.matrix_size else self.pos[self.num_coins+i]
            # RIGHT
            elif actions[i] == 3:
                player_pos[i] = self.pos[self.num_coins+i] + 1 if (self.pos[self.num_coins+i]+1)%self.matrix_size else self.pos[self.num_coins+i]
            # NOOP
            elif actions[i] == 4:
                player_pos[i] = self.pos[self.num_coins+i]

        self.state = np.zeros((self.num_channels, self.matrix_size, self.matrix_size))
        self.state[self.coin_color, int(self.pos[0]/self.matrix_size), int(self.pos[0]%self.matrix_size)] = 1

        # red and blue players move to the same cell
        if player_pos[0] == player_pos[1] and player_pos[0] != None:
            # the cell contains a coin
            if player_pos[0] == self.pos[0]:
                # both players will take the coin
                self.rewards[1-self.coin_color] = 1
                self.rewards[self.coin_color] = -1
                self.done = True
                self.info = [self.colors[1-self.coin_color], self.colors[self.coin_color]]
            # the cell does not contain a coin
            else:
                # both players will remain where they were
                self.state[self.num_coins*self.num_colors+0, int(self.pos[1]/self.matrix_size), int(self.pos[1]%self.matrix_size)] = 1
                self.state[self.num_coins*self.num_colors+1, int(self.pos[2]/self.matrix_size), int(self.pos[2]%self.matrix_size)] = 1
                #rand = np.random.randint(0, self.num_players, 1)[0]
                #self.state[self.num_coins*self.num_colors+rand, int(player_pos[rand]/self.matrix_size), int(player_pos[rand]%self.matrix_size)] = 1
                #self.state[self.num_coins*self.num_colors+1-rand, int(self.pos[1+1-rand]/self.matrix_size), int(self.pos[1+1-rand]%self.matrix_size)] = 1
        
        # simple state transition when none of the players land in the same cell
        else:
            self.pos[self.num_coins+0] = player_pos[0]
            self.pos[self.num_coins+1] = player_pos[1]
            self.state[self.num_coins*self.num_colors+0, int(self.pos[1]/self.matrix_size), int(self.pos[1]%self.matrix_size)] = 1
            self.state[self.num_coins*self.num_colors+1, int(self.pos[2]/self.matrix_size), int(self.pos[2]%self.matrix_size)] = 1
            # if red gets the coin
            if self.pos[1] == self.pos[0]:
                self.done = True
                self.info[0] = self.colors[0]
                # coin color is red
                if self.coin_color == 0:
                    self.rewards[0] = 1
                    self.rewards[1] = 0
                    self.info[1] = self.colors[0]
                # coin color is blue
                else:
                    self.rewards[0] = 1
                    self.rewards[1] = -2
                    self.info[1] = self.colors[1]
            # if blue gets the coin
            if self.pos[2] == self.pos[0]:
                self.done = True
                self.info[0] = self.colors[1]
                # coin color is red
                if self.coin_color == 0:
                    self.rewards[0] = -2
                    self.rewards[1] = 1
                    self.info[1] = self.colors[0]
                # coin color is blue
                else:
                    self.rewards[0] = 0
                    self.rewards[1] = 1
                    self.info[1] = self.colors[1]
        self.done = self.done or (self.num_steps == self.max_steps)
        return self.state, self.rewards, self.done, self.info

# coin game for N players with only M coins
class coin_game_Np():
    def __init__(self, max_steps, matrix_size, num_actions, num_players, num_coins):
        self.matrix_size = matrix_size
        self.blocks = self.matrix_size * self.matrix_size
        self.num_actions = num_actions
        self.num_players = num_players
        self.num_coins = num_coins
        # num colors is the same as num players
        self.num_colors = num_players
        self.num_channels = self.num_players + self.num_coins * self.num_colors
        self.max_steps = max_steps

    def reset(self):
        while True:
            self.pos = np.random.randint(0, self.blocks-1, self.num_players+self.num_coins)
            if self.pos[0] != self.pos[1] and self.pos[1] != self.pos[2] and self.pos[2] != self.pos[0]:
                break
        self.state = np.zeros(self.num_channels, self.matrix_size, self.matrix_size)
        for i in range(self.num_coins):
            coin_color = np.random.randint(0, self.num_colors, 1)[0]
            self.state[coin_color % self.num_colors, self.pos[i] / self.matrix_size, self.pos[i] % self.matrix_size] = 1
        for i in range(self.num_players):
            self.state[self.num_coins*self.num_colors+i, self.pos[self.num_coins+i]/self.matrix_size, self.pos[self.num_coins+i]%self.matrix_size] = 1
        return self.state, [None, None], False, {}

    def step(self, actions):
        for i in range(self.num_players):
            if actions[i] == 0:
                # UP
                player_pos = self.pos[self.num_coins+i] - self.matrix_size if self.pos[self.num_coins+i] - self.matrix_size >= 0 else self.pos[self.num_coins+i]
                self.state[self.num_coins*self.num_colors+i, player_pos/self.matrix_size, player_pos%self.matrix_size] = 1
            elif actions[i] == 1:
                # DOWN
                player_pos = self.pos[self.num_coins+i] + self.matrix_size if self.pos[self.num_coins+i] + self.matrix_size < self.blocks else self.pos[self.num_coins+i]
                self.state[self.num_coins*self.num_colors+i, player_pos/self.matrix_size, player_pos%self.matrix_size] = 1
            elif actions[i] == 2:
                # LEFT
                player_pos = self.pos[self.num_coins+i] - 1 if self.pos[self.num_coins+i]%self.matrix_size else self.pos[self.num_coins+i]
                self.state[self.num_coins*self.num_colors+i, player_pos/self.matrix_size, player_pos%self.matrix_size] = 1
            elif actions[i] == 3:
                # RIGHT
                player_pos = self.pos[self.num_coins+i] + 1 if (self.pos[self.num_coins+i]+1)%self.matrix_size else self.pos[self.num_coins+i]
                self.state[self.num_coins*self.num_colors+i, player_pos/self.matrix_size, player_pos%self.matrix_size] = 1
            elif actions[i] == 4:
                pass