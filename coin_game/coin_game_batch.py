import numpy as np

class coin_game_batch():
    def __init__(self, max_steps, batch_size, num_channels, grid_size = 4):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.grid_size = grid_size
        # 0 == red, 1 == blue
        self.n_players = self.n_colors = 2
        self.n_coins = 1
        self.n_actions = 5
        self.obs_space_shape = [num_channels, grid_size, grid_size]
        self.n_states = np.prod(self.obs_space_shape)
        # 0 == up, 1 == down, 2 == left, 3 == right, 4 == noop
        self.moves = [np.array([-1,0]),
                    np.array([1,0]),
                    np.array([0,-1]),
                    np.array([0,1]),
                    np.array([0,0])]
        self.n_steps = None

    def is_same_pos(self, x, y, axis = None):
        return np.all((x == y), axis = axis)
    
    def check_coin_pos(self, i):
        success = 0
        while success < 2:
            success = 0
            self.coin_pos[i] = np.random.randint(self.grid_size, size = 2)
            success += int(not self.is_same_pos(self.coin_pos[i], self.red_p_pos[i]))
            success += int(not self.is_same_pos(self.coin_pos[i], self.blue_p_pos[i]))
    
    def generate_state(self):
        state = np.zeros(self.batch_size + self.obs_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_p_pos[i][0], self.red_p_pos[i][1]] = 1
            state[i, 1, self.blue_p_pos[i][0], self.blue_p_pos[i][1]] = 1
            state[i, 3 if self.coin_color[i] else 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        return state

    def get_info(self):
        return self.info
    
    def reset(self):
        # reset the whole batch
        self.n_steps = 0
        self.coin_color = np.random.randint(self.n_colors, size = self.batch_size)
        self.red_p_pos = np.random.randint(self.grid_size, size = (self.batch_size, 2))
        self.blue_p_pos = np.random.randint(self.grid_size, size = (self.batch_size, 2))
        self.coin_pos = np.random.randint(self.grid_size, size = (self.batch_size, 2))
        for i in range(self.batch_size):
            while self.is_same_pos(self.red_p_pos[i], self.blue_p_pos[i]):
                self.red_p_pos[i] = np.random.randint(self.grid_size, size = 2)
            self.check_coin_pos(i)
        state = self.generate_state()
        #state = np.reshape(state, (self.batch_size, -1))
        observations = [state, state]
        rewards = [None, None]
        self.done = np.zeros(self.batch_size, dtype = bool)
        # info[0] == none, info[1] == RR, info[2] == RB, info[3] == BR, info[4] == BB
        self.info = np.zeros(5, dtype = int)
        return observations, rewards, self.done

    def step(self, actions):
        # one time step for the whole batch
        self.n_steps += 1
        a1_batch, a2_batch = actions
        
        #self.red_p_pos = (self.red_p_pos + np.array([self.moves[a1] for a1 in a1_batch if a1 in {0,1,2,3,4}]))%4
        #self.blue_p_pos = (self.blue_p_pos + np.array([self.moves[a2] for a2 in a2_batch if a2 in {0,1,2,3,4}]))%4

        for i in range(self.batch_size):
            if self.done[i]:
                continue
            a1, a2 = a1_batch[i], a2_batch[i]
            assert a1 in {0, 1, 2, 3, 4} and a2 in {0, 1, 2, 3, 4}

            # execute actions
            self.red_p_pos[i] = (self.red_p_pos[i] + self.moves[a1])%self.grid_size
            self.blue_p_pos[i] = (self.blue_p_pos[i] + self.moves[a2])%self.grid_size

        # distribute rewards
        reward1 = np.zeros(self.batch_size)
        reward2 = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if self.done[i]:
                continue
            if self.is_same_pos(self.red_p_pos[i], self.coin_pos[i]):
                self.done[i] = True
                #self.check_coin_pos(i)
                # red got coin
                reward1[i] += 1
                if self.coin_color[i]:
                    # blue coin
                    reward2[i] += -2
                    self.info[2] += 1
                else:
                    # red coin
                    self.info[1] += 1
            if self.is_same_pos(self.blue_p_pos[i], self.coin_pos[i]):
                self.done[i] = True
                #self.check_coin_pos(i)
                # blue got coin
                reward2[i] += 1
                if not self.coin_color[i]:
                    # red coin
                    reward1[i] += -2
                    self.info[3] += 1
                else:
                    # blue coin
                    self.info[4] += 1

        rewards = [reward1, reward2]
        state = self.generate_state()#.reshape(self.batch_size, -1)
        observations = [state, state]
        if self.n_steps == self.max_steps:
            self.info[0] += np.sum(1 - self.done)
            self.done = np.ones(self.batch_size, dtype = bool)
        return observations, rewards, self.done
