import environment.env_comparison.PCBBoard as pcb
import numpy as np


class BinaryPCBBoard(pcb.PCBBoard):
    def __init__(self, filename, num_agents=1):
        rows, cols, obs_rows, obs_cols, nets = pcb.read_file(filename)
        super().__init__(rows, cols, (2 * rows) - 1, (2 * cols) - 1, obs_rows, obs_cols, nets, num_agents)
        self.__observation_size = (2 * self.obs_rows - 1) * (2 * self.obs_cols - 1) + 5

    def initialise_grid(self):
        for _, net in self.nets.items():
            row_s, col_s = net.start
            row_e, col_e = net.end
            self.grid[2 * row_s, 2 * col_s] = 1.0
            self.grid[2 * row_e, 2 * col_e] = 1.0

    def step(self, agent_actions):
        reset = False

        for agent_id, action in agent_actions.items():
            if action == 0:
                continue

            agent = self.agents.get(agent_id)
            agent.prev_prev_action = agent.prev_action
            agent.prev_action = action

            if self.is_valid_move(agent, action):
                r, c = agent.location
                r_prime, c_prime = self.get_next_pos(agent.location, action)
                r_delta, c_delta = pcb.actions.get(action)

                # Update map
                self.grid[(2 * r) + r_delta, (2 * c) + c_delta] = 1.0
                self.grid[(2 * r_prime), (2 * c_prime)] = 1.0

                # Update agent params
                agent.location = (r_prime, c_prime)
                agent.moved = True

                self.nets.get(agent.net_id).path.append(action)

                if self.is_net_complete(agent):
                    agent.net_done = True
                    new_net = self.get_new_net()
                    if new_net is None:
                        agent.done = True
                    else:
                        agent.net_id = new_net.net_id
                        new_net.agent_id = agent.agent_id
                        agent.location = new_net.start

            else:
                # Allow all agents to move, then reset the env.
                reset = True

        if reset:
            self.reset()

        return self.get_reward(), self.board_complete()

    def is_valid_move(self, agent, action):
        next_r, next_c = self.get_next_pos(agent.location, action)

        # Check if new position is inside bounds of grid
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            return False

        # Check if new position is free/not the destination
        net = self.nets.get(agent.net_id)
        if self.grid[2 * next_r, 2 * next_c] != 0.0 and ((next_r, next_c) != net.end):
            return False

        # Check if diagonal lines cross
        if action % 2 == 0:
            r, c = agent.location
            delta_r, delta_c = pcb.actions.get(action)
            diag_r = (2 * r) + delta_r
            diag_c = (2 * c) + delta_c
            if self.grid[diag_r, diag_c] != 0.0:
                return False

        return True

    def observe(self, agent_id):
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

        observation = np.zeros(shape=self.__observation_size, dtype=np.float32)
        observation[:self.__observation_size - 5] = \
            self.grid[2 * row_start:(2 * row_end) - 1, 2 * col_start:(2 * col_end) - 1].flatten()
        observation[-5] = -row_delta / self.obs_rows  # TODO: might have to account for larger grid
        observation[-4] = -col_delta / self.obs_cols  # TODO: might have to account for larger grid
        observation[-3] = (dest_r - r) / self.rows    # TODO: might have to account for larger grid
        observation[-2] = (dest_c - c) / self.cols    # TODO: might have to account for larger grid
        observation[-1] = agent_id / self.num_agents

        return observation

    def padded_observation(self, agent_id):
        agent = self.agents.get(agent_id)
        if agent.done:
            return None

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

        row_start *= 2
        row_end = (row_end * 2) - 1
        row_start_delta *= 2
        row_end_delta *= 2

        col_start *= 2
        col_end = (col_end * 2) - 1
        col_start_delta *= 2
        col_end_delta *= 2

        grid_observation = np.ones(shape=((2 * self.obs_rows - 1), (2 * self.obs_cols - 1)), dtype=np.float32)

        grid_observation[row_start_delta:(2 * self.obs_rows - 1) + row_end_delta, col_start_delta:(2 * self.obs_cols - 1) + col_end_delta] \
            = self.grid[row_start:row_end, col_start:col_end]
        agent_info = np.zeros(shape=3, dtype=np.float32)
        agent_info[0] = (dest_r - r) / self.rows
        agent_info[1] = (dest_c - c) / self.cols
        agent_info[2] = agent_id

        observation = np.concatenate((grid_observation.flatten(), agent_info))

        return observation
