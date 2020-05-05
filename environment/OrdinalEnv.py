import gym
import numpy as np
import environment.BaseEnv as pcb

opposite_actions = {1: 5,
                    2: 6,
                    3: 7,
                    4: 8,
                    5: 1,
                    6: 2,
                    7: 3,
                    8: 4, }
grid_matrix = [[-1,  1, 2,  3,  4,  5,  6,  7,  8,  9 ],
               [ 1, -1, 10, 11, 12, 13, 14, 15, 16, 17],
               [ 2, 10, -1, 18, 19, 20, 21, 22, 23, 24],
               [ 3, 11, 18, -1, 25, 26, 27, 28, 29, 30],
               [ 4, 12, 19, 25, -1, 31, 32, 33, 34, 35],
               [ 5, 13, 20, 26, 31, -1, 36, 37, 38, 39],
               [ 6, 14, 21, 27, 32, 36, -1, 40, 41, 42],
               [ 7, 15, 22, 28, 33, 37, 40, -1, 43, 44],
               [ 8, 16, 23, 29, 34, 38, 41, 43, -1, 45],
               [ 9, 17, 24, 30, 35, 39, 42, 44, 45, -1],]

grid_matrix_uniform = [[-1, 1, 2,  3,  4,  5,  6,  7,  8,  9],
                       [1, -1, 11, 12, 13, 14, 15, 16, 17, 18],
                       [2, 11, -1, 21, 22, 23, 24, 25, 26, 27],
                       [3, 12, 21, -1, 31, 32, 33, 34, 35, 36],
                       [4, 13, 22, 31, -1, 41, 42, 43, 44, 45],
                       [5, 14, 23, 32, 41, -1, 51, 52, 53, 54],
                       [6, 15, 24, 33, 42, 51, -1, 61, 62, 63],
                       [7, 16, 25, 34, 43, 52, 61, -1, 71, 72],
                       [8, 17, 26, 35, 44, 53, 62, 71, -1, 81],
                       [9, 18, 27, 36, 45, 54, 63, 72, 81, -1],]
MAX_VALUE = max(max(grid_matrix)) + 1


class OrdinalEnv(pcb.BaseEnv):
    def __init__(self,
                 min_rows,
                 min_cols,
                 max_rows,
                 max_cols,
                 min_nets,
                 max_nets,
                 rand_nets=True,
                 filename=None,
                 padded=True):
        super().__init__(min_rows,
                         min_cols,
                         max_rows,
                         max_cols,
                         min_nets=min_nets,
                         max_nets=max_nets,
                         blank_value=MAX_VALUE,
                         obstacle_value=0.0,
                         rand_nets=rand_nets,
                         filename=filename)
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=0,
                            high=1,
                            shape=(1, self.obs_rows, self.obs_cols),
                            dtype=np.float32),
             gym.spaces.Box(low=0,
                            high=1,
                            shape=(2,),
                            dtype=np.float32)))

    def get_observation_size(self):
        return self.MAX_OBS_ROWS * self.MAX_OBS_COLS + 2

    def initialise_grid(self):
        for _, net in self.nets.items():
            row_s, col_s = net.start
            row_e, col_e = net.end
            self.grid[row_s, col_s] = 9
            self.grid[row_e, col_e] = 9

    def step(self, action: int):
        self.current_step += 1

        if self.agent.done:
            pass
        else:
            if not self.agent.invalid_move:
                self.agent.prev_prev_action = self.agent.prev_action
            self.agent.prev_action = action

            if self.is_valid_move(action):
                self.agent.invalid_move = False
                r, c = self.agent.location
                r_prime, c_prime = self.get_next_pos(self.agent.location, action)

                self.grid[r, c] = \
                    grid_matrix[self.grid[r, c] % MAX_VALUE][action]

                self.grid[r_prime, c_prime] = \
                    grid_matrix[self.grid[r_prime, c_prime] % MAX_VALUE] \
                               [opposite_actions[action]]
                self.agent.location = (r_prime, c_prime)

                self.nets.get(self.agent.net_id).path.append(action)

                if self.is_net_complete():
                    self.agent.net_done = True
                    new_net = self.get_new_net()
                    if new_net is None:
                        self.agent.done = True
                    else:
                        self.agent.net_id = new_net.net_id
                        self.agent.location = new_net.start

            else:
                self.agent.invalid_move = True

        reward = self.get_reward()
        self.total_reward += reward
        done = self.board_complete()
        time_out = self.current_step > self.total_steps
        obs = self.observe()

        return obs, reward, done or time_out, {"completed": done}

    def is_valid_move(self, action):
        next_r, next_c = self.get_next_pos(self.agent.location, action)

        # Check if new position is inside bounds of grid
        if next_r < 0 or next_r >= self.rows or \
                next_c < 0 or next_c >= self.cols:
            return False

        # Check if new position is free/not the destination
        net = self.nets.get(self.agent.net_id)
        if self.grid[next_r, next_c] != MAX_VALUE and \
                ((next_r, next_c) != net.end):
            return False

        # Check if diagonal lines cross
        if action % 2 == 0:
            r, c = self.agent.location
            delta_r, delta_c = pcb.actions.get(action)

            a1 = ((action + 7 - (2 * delta_r * delta_c)) % 8) + 1
            a2 = ((action + 7 + (2 * delta_r * delta_c)) % 8) + 1

            if self.grid[r + delta_r, c] in grid_matrix[a1] or \
                    self.grid[r, c + delta_c] in grid_matrix[a2]:
                return False

        return True

    def observe(self):
        return self.__padded_observation()

    def __padded_observation(self):
        if self.agent.done:
            return np.full(shape=(self.obs_rows, self.obs_cols),
                           fill_value=self.obstacle_value, dtype=np.float32), \
                   np.zeros(shape=2, dtype=np.float)
        dest_r, dest_c = self.nets.get(self.agent.net_id).end
        centre_obs_row = self.obs_rows // 2
        centre_obs_col = self.obs_cols // 2
        r, c = self.agent.location
        grid_observation = np.full(shape=(self.obs_rows, self.obs_cols),
                                   fill_value=self.obstacle_value,
                                   dtype=np.float32)

        obs_start_row = max(0, centre_obs_row - r)
        obs_end_row = min(self.obs_rows, centre_obs_row + self.rows - r)
        obs_start_col = max(0, centre_obs_col - c)
        obs_end_col = min(self.obs_cols, centre_obs_col + self.cols - c)

        grid_start_row = max(0, r - centre_obs_row)
        grid_end_row = min(self.rows, r + self.obs_rows - centre_obs_row)
        grid_start_col = max(0, c - centre_obs_col)
        grid_end_col = min(self.cols, c + self.obs_cols - centre_obs_col)

        grid_observation[obs_start_row:obs_end_row, obs_start_col:obs_end_col] = \
            self.grid[grid_start_row:grid_end_row, grid_start_col:grid_end_col]
        grid_observation /= MAX_VALUE

        agent_info = np.zeros(shape=2, dtype=np.float32)
        agent_info[0] = (dest_r - r) / self.rows
        agent_info[1] = (dest_c - c) / self.cols

        return grid_observation, agent_info
