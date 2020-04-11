import environment.PCBBoard as pcb
import numpy as np


class BinaryPCBBoard(pcb.PCBBoard):
    def __init__(self, rows, cols, rand_nets=True, min_nets=None, max_nets=None, filename=None, padded=True):
        self.padded = padded
        super().__init__(rows, cols, (3 * rows) - 2, (3 * cols) - 2, blank_value=0.0, obstacle_value=1.0, rand_nets=rand_nets, min_nets=min_nets, max_nets=max_nets, filename=filename)

    def get_observation_size(self):
        if self.padded:
            return (3 * self.obs_rows - 2) * (3 * self.obs_cols - 2) + 2
        else:
            return (3 * self.obs_rows - 2) * (3 * self.obs_cols - 2) + 4

    def initialise_grid(self):
        for _, net in self.nets.items():
            row_s, col_s = net.start
            row_e, col_e = net.end
            self.grid[3 * row_s, 3 * col_s] = 1.0
            self.grid[3 * row_e, 3 * col_e] = 1.0

    def step(self, action):
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
                r_delta, c_delta = pcb.actions.get(action)

                # Update map
                for i in range(1, 4):
                    self.grid[(3 * r) + (i * r_delta), (3 * c) + (i * c_delta)] = 1.0

                # Update agent params
                self.agent.location = self.get_next_pos(self.agent.location, action)
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
        if self.grid[3 * next_r, 3 * next_c] != 0.0 and ((next_r, next_c) != net.end):
            return False

        # Check if diagonal lines cross
        if action % 2 == 0:
            r, c = self.agent.location
            delta_r, delta_c = pcb.actions.get(action)
            diag_r = (3 * r) + (2 * delta_r)
            diag_c = (3 * c) + delta_c
            if self.grid[diag_r, diag_c] != 0.0:
                return False

        return True

    def observe(self):
        if self.padded:
            return self.__padded_observation()
        else:
            pass
            # return self.__shifted_observation()

    def __padded_observation(self):
        if self.agent.done:
            return np.full(shape=self.get_observation_size(), fill_value=self.obstacle_value, dtype=np.float32)
        dest_r, dest_c = self.nets.get(self.agent.net_id).end
        centre_obs_row = self.obs_rows // 2
        centre_obs_col = self.obs_cols // 2
        r, c = self.agent.location
        grid_observation = np.full(shape=((3 * self.obs_rows) - 2, (3 * self.obs_cols) - 2), fill_value=self.obstacle_value, dtype=np.float32)

        obs_start_row = 3 * max(0, centre_obs_row - r)
        obs_end_row = 3 * min(self.obs_rows, centre_obs_row + self.rows - r) - 2
        obs_start_col = 3 * max(0, centre_obs_col - c)
        obs_end_col = 3 * min(self.obs_cols, centre_obs_col + self.cols - c) - 2

        grid_start_row = 3 * max(0, r - centre_obs_row)
        grid_end_row = 3 * min(self.rows, r + self.obs_rows - centre_obs_row) - 2
        grid_start_col = 3 * max(0, c - centre_obs_col)
        grid_end_col = 3 * min(self.cols, c + self.obs_cols - centre_obs_col) - 2

        grid_observation[obs_start_row:obs_end_row, obs_start_col:obs_end_col] = \
            self.grid[grid_start_row:grid_end_row, grid_start_col:grid_end_col]

        agent_info = np.zeros(shape=2, dtype=np.float32)
        agent_info[0] = (dest_r - r) / self.rows
        agent_info[1] = (dest_c - c) / self.cols

        observation = np.concatenate((grid_observation.flatten(), agent_info))

        return observation
