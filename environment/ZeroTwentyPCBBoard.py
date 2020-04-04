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


class ZeroTwentyPCBBoard(pcb.PCBBoard):
    def __init__(self, filename, padded=True):
        rows, cols, obs_rows, obs_cols, nets = pcb.read_file(filename)
        self.padded = padded
        super().__init__(rows, cols, rows, cols, obs_rows, obs_cols, nets)

    def get_observation_size(self):
        if self.padded:
            return self.obs_rows * self.obs_cols + 2
        else:
            return self.obs_rows * self.obs_cols + 4

    def initialise_grid(self):
        for _, net in self.nets.items():
            row_s, col_s = net.start
            row_e, col_e = net.end
            self.grid[row_s, col_s] = GridCells.VIA.value
            self.grid[row_e, col_e] = GridCells.VIA.value

        self.grid[self.agent.location] = GridCells.AGENT.value

    def step(self, action):
        self.current_step += 1

        if self.agent.done:
            pass
        elif action == 0:
            self.agent.moved = False
        elif not self.agent.invalid_move:
            self.agent.prev_prev_action = self.agent.prev_action
        else:
            self.agent.prev_action = action
            self.agent.moved = True

            if self.is_valid_move(action):
                self.agent.invalid_move = False
                r, c = self.agent.location
                r_prime, c_prime = self.get_next_pos(self.agent.location, action)

                # Start of net update treated differently
                if self.nets.get(self.agent.net_id).start == self.agent.location:
                    self.grid[r, c] = GridCells.VIA.value + action
                else:
                    self.grid[r, c] = action

                self.grid[r_prime, c_prime] = GridCells.AGENT.value
                self.agent.location = (r_prime, c_prime)

                self.nets.get(self.agent.net_id).path.append(action)

                if self.is_net_complete():
                    self.grid[r_prime, c_prime] = GridCells.VIA.value + opposite_actions.get(action)

                    self.agent.net_done = True
                    new_net = self.get_new_net()
                    if new_net is None:
                        self.agent.done = True
                    else:
                        self.agent.net_id = new_net.net_id
                        new_net.agent_id = self.agent.agent_id
                        self.agent.location = new_net.start
                        self.grid[self.agent.location] = GridCells.AGENT.value

            else:
                self.agent.invalid_move = True

        reward = self.get_reward()
        self.total_reward += reward
        done = self.board_complete() or self.current_step > self.total_steps
        obs = self.observe()

        return obs, reward, done, {}

    def is_valid_move(self, action):
        next_r, next_c = self.get_next_pos(self.agent.location, action)

        # Check if new position is inside bounds of grid
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            return False

        # Check if new position is free/not the destination
        net = self.nets.get(self.agent.net_id)
        if self.grid[next_r, next_c] != 0.0 and ((next_r, next_c) != net.end):
            return False

        # Check if diagonal lines cross
        if action % 2 == 0:
            r, c = self.agent.location
            delta_r, delta_c = pcb.actions.get(action)

            # TODO: weird and overcomplicated there's probably a much better way to do this
            a1 = ((action + 7 - (2 * delta_r * delta_c)) % 8) + 1
            a2 = ((action + 7 + (2 * delta_r * delta_c)) % 8) + 1

            if self.grid[r + delta_r, c] % GridCells.VIA.value == a1 or \
                    self.grid[r, c + delta_c] % GridCells.VIA.value == a2:
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
            return np.full(shape=self.get_observation_size(), fill_value=GridCells.BLANK.value, dtype=np.float32)

        r, c = self.agent.location
        dest_r, dest_c = self.nets.get(self.agent.net_id).end
        row_start = r - self._observation_row_start_offset
        row_end = r + self._observation_row_end_offset
        row_start_delta = 0
        row_end_delta = 0
        if row_start < 0:
            row_start_delta = -row_start
        elif row_end > self.rows:
            row_end_delta = self.rows - row_end
        row_start += row_start_delta
        row_end += row_end_delta

        col_start = c - self._observation_col_start_offset
        col_end = c + self._observation_col_end_offset
        col_start_delta = 0
        col_end_delta = 0
        if col_start < 0:
            col_start_delta = -col_start
        elif col_end > self.cols:
            col_end_delta = self.cols - col_end
        col_start += col_start_delta
        col_end += col_end_delta

        grid_observation = np.full(shape=(self.obs_rows, self.obs_cols), fill_value=GridCells.OBSTACLE.value, dtype=np.float32)

        grid_observation[row_start_delta:self.obs_rows + row_end_delta, col_start_delta:self.obs_cols + col_end_delta] \
            = self.grid[row_start:row_end, col_start:col_end]
        grid_observation /= len(GridCells)
        agent_info = np.zeros(shape=2, dtype=np.float32)
        agent_info[0] = (dest_r - r) / self.rows
        agent_info[1] = (dest_c - c) / self.cols

        observation = np.concatenate((grid_observation.flatten(), agent_info))

        return observation
