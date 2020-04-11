import gym
import numpy as np
import environment.PCBBoard as pcb
from environment.GridCells import GridCells

opposite_actions = {1: 5,
                    2: 6,
                    3: 7,
                    4: 8,
                    5: 1,
                    6: 2,
                    7: 3,
                    8: 4, }
grid_matrix = [[-1,  1, 2,  3,  4,  5,  6,  7,  8,  9 ],
                [1, -1, 10, 11, 12, 13, 14, 15, 16, 17],
                [2, 10, -1, 18, 19, 20, 21, 22, 23, 24],
                [3, 11, 18, -1, 25, 26, 27, 28, 29, 30],
                [4, 12, 19, 25, -1, 31, 32, 33, 34, 35],
                [5, 13, 20, 26, 31, -1, 36, 37, 38, 39],
                [6, 14, 21, 27, 32, 36, -1, 40, 41, 42],
                [7, 15, 22, 28, 33, 37, 40, -1, 43, 44],
                [8, 16, 23, 29, 34, 38, 41, 43, -1, 45],
                [9, 17, 24, 30, 35, 39, 42, 44, 45, -1],]
MAX_VALUE = max(max(grid_matrix)) + 1


class ZeroTwentyPCBBoard(pcb.PCBBoard):
    def __init__(self, rows, cols, rand_nets=True, min_nets=None, max_nets=None, filename=None, padded=True):
        self.padded = padded
        super().__init__(rows, cols, rows, cols, blank_value=MAX_VALUE, obstacle_value=0.0, rand_nets=rand_nets, min_nets=min_nets, max_nets=max_nets, filename=filename)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(1, self.obs_rows, self.obs_cols), dtype=np.float32),
                                                   gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)))

    def get_observation_size(self):
        if self.padded:
            return self.MAX_OBS_ROWS * self.MAX_OBS_COLS + 2
        else:
            return self.MAX_OBS_ROWS * self.MAX_OBS_COLS + 4

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

                self.grid[r, c] = grid_matrix[self.grid[r, c] % MAX_VALUE][action]

                self.grid[r_prime, c_prime] = grid_matrix[self.grid[r_prime, c_prime] % MAX_VALUE][opposite_actions[action]]
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

        return obs, reward, done or time_out, done

    def is_valid_move(self, action):
        next_r, next_c = self.get_next_pos(self.agent.location, action)

        # Check if new position is inside bounds of grid
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            return False

        # Check if new position is free/not the destination
        net = self.nets.get(self.agent.net_id)
        if self.grid[next_r, next_c] != MAX_VALUE and ((next_r, next_c) != net.end):
            return False

        # Check if diagonal lines cross
        if action % 2 == 0:
            r, c = self.agent.location
            delta_r, delta_c = pcb.actions.get(action)

            # TODO: weird and overcomplicated there's probably a much better way to do this
            a1 = ((action + 7 - (2 * delta_r * delta_c)) % 8) + 1
            a2 = ((action + 7 + (2 * delta_r * delta_c)) % 8) + 1

            if self.grid[r + delta_r, c] in grid_matrix[a1] or self.grid[r, c + delta_c] in grid_matrix[a2]:
                return False

        return True

    def observe(self):
        if self.padded:
            return self.__padded_observation()
        else:
            return self.__shifted_observation()

    def __shifted_observation(self):
        if self.agent.done:
            return np.full(shape=self.get_observation_size(), fill_value=GridCells.BLANK.value, dtype=np.float32)

        r, c = self.agent.location
        dest_r, dest_c = self.nets.get(self.agent.net_id).end
        row_start = r - self._observation_row_start_offset
        row_end = r + self._observation_row_end_offset
        row_delta = 0
        if row_start < 0:
            row_delta = -row_start
        elif row_end > self.rows:
            row_delta = self.rows - row_end
        row_start += row_delta
        row_end += row_delta

        col_start = c - self._observation_col_start_offset
        col_end = c + self._observation_col_end_offset
        col_delta = 0
        if col_start < 0:
            col_delta = -col_start
        elif col_end > self.cols:
            col_delta = self.cols - col_end
        col_start += col_delta
        col_end += col_delta

        observation = np.zeros(shape=self.obs_rows * self.obs_cols + 5, dtype=np.float32)
        observation[:self.obs_rows * self.obs_cols] = \
            self.grid[row_start:row_end, col_start:col_end].flatten() / len(GridCells)
        observation[-4] = -row_delta / self.obs_rows  # TODO: might have to account for larger grid
        observation[-3] = -col_delta / self.obs_cols  # TODO: might have to account for larger grid
        observation[-2] = (dest_r - r) / self.rows  # TODO: might have to account for larger grid
        observation[-1] = (dest_c - c) / self.cols  # TODO: might have to account for larger grid

        return observation

    def __padded_observation(self):
        if self.agent.done:
            return np.full(shape=(self.obs_rows, self.obs_cols), fill_value=self.obstacle_value, dtype=np.float32), np.zeros(shape=2, dtype=np.float)
        dest_r, dest_c = self.nets.get(self.agent.net_id).end
        centre_obs_row = self.obs_rows // 2
        centre_obs_col = self.obs_cols // 2
        r, c = self.agent.location
        grid_observation = np.full(shape=(self.obs_rows, self.obs_cols), fill_value=self.obstacle_value, dtype=np.float32)

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
