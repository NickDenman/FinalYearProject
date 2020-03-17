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
    def __init__(self, filename, num_agents=1, padded=True):
        rows, cols, obs_rows, obs_cols, nets = pcb.read_file(filename)
        self.padded = padded
        super().__init__(rows, cols, rows, cols, obs_rows, obs_cols, nets, num_agents)

    def get_observation_size(self):
        if self.padded:
            return self.obs_rows * self.obs_cols + 3
        else:
            return self.obs_rows * self.obs_cols + 5

    def initialise_grid(self):
        for _, net in self.nets.items():
            row_s, col_s = net.start
            row_e, col_e = net.end
            self.grid[row_s, col_s] = GridCells.VIA.value
            self.grid[row_e, col_e] = GridCells.VIA.value

        for _, agent in self.agents.items():
            self.grid[agent.location] = GridCells.AGENT.value

    def step(self, agent_actions):
        reset = False

        for agent_id, action in agent_actions.items():
            agent = self.agents.get(agent_id)
            if agent.done:
                continue

            if action == 0:
                agent.moved = False
                continue

            if not agent.invalid_move:
                agent.prev_prev_action = agent.prev_action

            agent.prev_action = action
            agent.moved = True

            if self.is_valid_move(agent, action):
                agent.invalid_move = False
                r, c = agent.location
                r_prime, c_prime = self.get_next_pos(agent.location, action)

                # Start of net update treated differently
                if self.nets.get(agent.net_id).start == agent.location:
                    self.grid[r, c] = GridCells.VIA.value + action
                else:
                    self.grid[r, c] = action

                self.grid[r_prime, c_prime] = GridCells.AGENT.value
                agent.location = (r_prime, c_prime)

                self.nets.get(agent.net_id).path.append(action)

                if self.is_net_complete(agent):
                    self.grid[r_prime, c_prime] = GridCells.VIA.value + opposite_actions.get(action)

                    agent.net_done = True
                    new_net = self.get_new_net()
                    if new_net is None:
                        agent.done = True
                    else:
                        agent.net_id = new_net.net_id
                        new_net.agent_id = agent.agent_id
                        agent.location = new_net.start
                        self.grid[agent.location] = GridCells.AGENT.value

            else:
                # Allow all agents to move, then reset the env.
                agent.invalid_move = True
                reset = True

        reward = self.get_reward()
        done = self.board_complete()
        if done:
            obs = self.reset()
        else:
            obs = [self.observe(i) for i in range(self.num_agents)]

        agent_status = [a.done for _, a in self.agents.items()]

        return obs, reward, done, agent_status

    def is_valid_move(self, agent, action):
        next_r, next_c = self.get_next_pos(agent.location, action)

        # Check if new position is inside bounds of grid
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            return False

        # Check if new position is free/not the destination
        net = self.nets.get(agent.net_id)
        if self.grid[next_r, next_c] != 0.0 and ((next_r, next_c) != net.end):
            return False

        # Check if diagonal lines cross
        if action % 2 == 0:
            r, c = agent.location
            delta_r, delta_c = pcb.actions.get(action)

            # TODO: weird and overcomplicated there's probably a much better way to do this
            a1 = ((action + 7 - (2 * delta_r * delta_c)) % 8) + 1
            a2 = ((action + 7 + (2 * delta_r * delta_c)) % 8) + 1

            if self.grid[r + delta_r, c] % GridCells.VIA.value == a1 or \
                    self.grid[r, c + delta_c] % GridCells.VIA.value == a2:
                return False

        return True

    def observe(self, agent_id):
        if self.padded:
            return self.__padded_observation(agent_id)
        else:
            return self.__shifted_observation(agent_id)

    def __shifted_observation(self, agent_id):
        agent = self.agents.get(agent_id)
        if agent.done:
            return None

        r, c = agent.location
        dest_r, dest_c = self.nets.get(agent.net_id).end
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
        observation[-5] = -row_delta / self.obs_rows  # TODO: might have to account for larger grid
        observation[-4] = -col_delta / self.obs_cols  # TODO: might have to account for larger grid
        observation[-3] = (dest_r - r) / self.rows  # TODO: might have to account for larger grid
        observation[-2] = (dest_c - c) / self.cols  # TODO: might have to account for larger grid
        observation[-1] = agent_id / self.num_agents

        return observation

    def __padded_observation(self, agent_id):
        agent = self.agents.get(agent_id)
        if agent.done:
            return np.full(shape=(self.obs_rows * self.obs_cols + 3), fill_value=GridCells.BLANK.value, dtype=np.float32)

        r, c = agent.location
        dest_r, dest_c = self.nets.get(agent.net_id).end
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
        agent_info = np.zeros(shape=3, dtype=np.float32)
        agent_info[0] = (dest_r - r) / self.rows
        agent_info[1] = (dest_c - c) / self.cols
        agent_info[2] = agent_id

        observation = np.concatenate((grid_observation.flatten(), agent_info))

        return observation
