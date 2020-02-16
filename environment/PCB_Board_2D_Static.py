import numpy as np
import copy
import environment.PCBRenderer as renderer
import environment.GridCells as gc

FINAL_REWARD = 25.0
NET_REWARD = 0.0
STEP_REWARD = -1.0
INVALID_ACTION_REWARD = -4.0
DIAGONAL_REWARD = -0.4  # diagonal lines are of length √2 ~= 1.4

action_symbols = {0: ".",
                  1: "|",
                  2: "/",
                  3: "-",
                  4: "\\",
                  5: "|",
                  6: "/",
                  7: "-",
                  8: "\\",
                  9: "X",
                  10: "O"}


class PCBBoard:
    def __init__(self, rows=4, cols=4, min_net_dist=2.0, observation_rows=4, observation_cols=4, num_nets=10,
                 num_agents=1):
        self.grid = np.zeros(((2 * rows) - 1, (2 * cols) - 1), dtype=int)
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.observation_rows = observation_rows if observation_rows < rows else rows
        self.observation_cols = observation_cols if observation_cols < cols else cols
        self.num_nets = num_nets
        self.min_net_dist = min_net_dist
        self.agents = {}
        self.nets = {}
        self.total_nets = 0
        self.cur_net_id = 0

        self.actions = {1: [-1, 0],
                        2: [-1, 1],
                        3: [0, 1],
                        4: [1, 1],
                        5: [1, 0],
                        6: [1, -1],
                        7: [0, -1],
                        8: [-1, -1]}

        self.initialise_grid()
        self.initialise_agent_data()

    def reset_env(self):
        self.grid.fill(0)
        self.nets.clear()
        self.total_nets = 0
        self.cur_net_id = 0
        self.agents.clear()

        self.initialise_grid()
        self.initialise_agent_data()

    def add_net(self, start_row, start_col, end_row, end_col):
        self.grid[2 * start_row, 2 * start_col] = gc.GridCells.VIA.value
        self.grid[2 * end_row, 2 * end_col] = gc.GridCells.VIA.value
        self.nets[self.total_nets] = Net(Position(start_row, start_col), Position(end_row, end_col))
        self.total_nets += 1

    def initialise_grid(self):
        self.add_net(4, 0, 4, 5)
        self.add_net(5, 0, 1, 4)
        self.add_net(2, 2, 4, 4)
        self.add_net(5, 1, 5, 5)

    def get_new_net(self, agent_id):
        if self.cur_net_id == self.total_nets:
            return None

        # TODO: this needs to be thread safe for when multiple agents are running...
        net = self.nets.get(self.cur_net_id)
        net.agent_id = agent_id
        self.cur_net_id += 1

        return net

    def initialise_agent_data(self):
        agent_id = 0

        while agent_id < self.num_agents and len(self.nets) > 0:
            agent = AgentData()
            agent.agent_id = agent_id
            agent.cur_net = self.get_new_net(agent_id)
            agent.cur_loc = copy.deepcopy(agent.cur_net.start)
            self.grid[2 * agent.cur_loc.row, 2 * agent.cur_loc.col] = gc.GridCells.AGENT.value
            agent.dir = gc.GridCells.VIA.value
            agent.new_net = True
            self.agents[agent_id] = agent
            agent_id += 1

    def is_game_over(self):
        game_over = True
        for _, agent in self.agents.items():
            game_over &= agent.complete

        return game_over

    def get_next_pos(self, pos, direction):
        grid_deltas = self.actions.get(direction)

        return Position(pos.row + grid_deltas[0], pos.col + grid_deltas[1])

    # return grid, current_location relative to the observation grid and destination coordinates relative to agent
    def observe(self, agent_id):
        if self.agents.get(agent_id).complete:
            return None, None, None

        agent_loc = self.agents.get(agent_id).cur_loc
        row_start = agent_loc.row - (self.observation_rows // 2)
        row_end = agent_loc.row + (self.observation_rows // 2)
        row_delta = 0

        if row_start < 0:
            row_delta = -row_start
        elif row_end > self.rows:
            row_delta = self.rows - row_end

        row_start += row_delta
        row_end += row_delta

        col_start = agent_loc.col - (self.observation_cols // 2)
        col_end = agent_loc.col + (self.observation_cols // 2)
        col_delta = 0

        if col_start < 0:
            col_delta = -col_start
        elif col_end > self.cols:
            col_delta = self.cols - col_end

        col_start += col_delta
        col_end += col_delta

        observation = np.array(self.grid[2 * row_start:2 * row_end:2, 2 * col_start:2 * col_end: 2], copy=True)

        for row in range(row_end - row_start):
            for col in range(col_end - col_start):
                if observation[row, col] > gc.GridCells.VIA.value:
                    observation[row, col] %= 10

        return observation, \
               Position(-row_delta, -col_delta), \
               self.agents.get(agent_id).cur_net.end - agent_loc

    def joint_act(self, actions):
        for id, action in actions.items():
            if not self.agents.get(id).complete:
                self.__act(id, action)

        return self.get_reward()

    def __act(self, agent_id, direction):
        if direction == 9:
            print("shit...")

        agent_loc = self.agents.get(agent_id).cur_loc

        if self.is_valid_move(agent_loc, direction, agent_id):
            self.agents.get(agent_id).valid_action = True
            next_pos = self.get_next_pos(agent_loc, direction)

            grid_deltas = self.actions.get(direction)

            if self.agents.get(agent_id).new_net:
                self.agents.get(agent_id).new_net = False
                self.grid[2 * agent_loc.row, 2 * agent_loc.col] = gc.GridCells.VIA.value + direction
            else:
                self.grid[2 * agent_loc.row, 2 * agent_loc.col] = direction

            self.grid[(2 * agent_loc.row) + grid_deltas[0], (2 * agent_loc.col) + grid_deltas[1]] = direction
            self.agents.get(agent_id).prev_loc = agent_loc
            self.agents.get(agent_id).cur_loc = next_pos
            self.agents.get(agent_id).prev_dir = self.agents.get(agent_id).dir
            self.agents.get(agent_id).dir = direction

            if self.is_net_complete(agent_id):
                # print("Net complete...")
                self.agents.get(agent_id).prev_net = self.agents.get(agent_id).cur_net
                self.agents.get(agent_id).prev_loc = self.agents.get(agent_id).cur_loc

                new_net = self.get_new_net(agent_id)
                if new_net is not None:
                    self.agents.get(agent_id).cur_net = new_net
                    self.agents.get(agent_id).cur_loc = self.agents.get(agent_id).cur_net.start
                    self.agents.get(agent_id).new_net = True

                else:
                    self.agents.get(agent_id).cur_net = None
                    self.agents.get(agent_id).cur_loc = None
                    self.agents.get(agent_id).complete = True

            else:
                self.grid[2 * next_pos.row, 2 * next_pos.col] = gc.GridCells.AGENT.value

        else:
            self.agents.get(agent_id).valid_action = False
            # print("invalid move: " + str(direction) + " in position " + str(agent_loc) + " for agent " + str(agent_id))

    def step_reward_function(self, direction, prev_dir):
        if prev_dir < gc.GridCells.VIA.value:
            cur_dir_deltas = self.actions[direction]
            prev_dir_deltas = self.actions[prev_dir]

            dir_delta = abs(cur_dir_deltas[0] - prev_dir_deltas[0]) + abs(cur_dir_deltas[1] - prev_dir_deltas[1])
            if dir_delta <= 1:
                return 0

            return -dir_delta
        return 0

    def get_reward(self):
        reward = 0.0

        if self.is_game_over():
            reward += FINAL_REWARD

        reward += STEP_REWARD
        for _, agent in self.agents.items():
            if agent.complete:
                continue

            if agent.dir % 2 == 0:
                reward += DIAGONAL_REWARD
            if agent.valid_action and agent.prev_loc != agent.cur_net.start:
                reward += self.step_reward_function(agent.dir, agent.prev_dir)
            elif not agent.valid_action:
                reward += INVALID_ACTION_REWARD

        return reward

    def is_net_complete(self, agent_id):
        return self.agents.get(agent_id).cur_loc == self.agents.get(agent_id).cur_net.end

    def is_valid_move(self, agent_loc, direction, agent_id):
        # check if in boundary
        next_pos = self.get_next_pos(agent_loc, direction)

        if next_pos.row < 0 or next_pos.row >= self.rows:
            return False
        elif next_pos.col < 0 or next_pos.col >= self.cols:
            return False

        # check if next grid is free or the net destination
        if self.grid[2 * next_pos.row, 2 * next_pos.col] != 0 and next_pos != self.agents.get(agent_id).cur_net.end:
            return False

        # check diagonals do not cross
        elif direction % 2 == 0 and not self.is_valid_diagonal(direction, agent_loc):
            return False

        return True

    def is_valid_diagonal(self, direction, location):
        position_deltas = self.actions.get(direction)
        return self.grid[2 * location.row + position_deltas[0], 2 * location.col + position_deltas[1]] == 0

    def print_grid(self):
        result = np.full(fill_value=" ", shape=((self.rows * 2) - 1, (self.cols * 2) - 1), dtype=str)
        for row in range((self.rows * 2) - 1):
            for col in range((self.cols * 2) - 1):
                if self.grid[row, col] < gc.GridCells.VIA.value:
                    result[row, col] = action_symbols.get(self.grid[row, col])
                else:
                    result[row, col] = action_symbols.get(gc.GridCells.VIA.value)

        for _row in range((self.rows * 2) - 1):
            for _col in range((self.cols * 2) - 1):
                print(result[_row, _col], end='')
            print()


class Net:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.agent_id = -1

    def __eq__(self, other):
        if isinstance(other, Net):
            return self.start == other.start and self.end == other.end

        return False

    def __str__(self):
        return str(self.start) + " --> " + str(self.end)

    def __hash__(self):
        return hash((self.start, self.end))


class Position:
    def __init__(self, row=-1, col=-1):
        self.row = row
        self.col = col

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.row == other.row and self.col == other.col
        return False

    def __str__(self):
        return "(" + str(self.row) + ", " + str(self.col) + ")"

    def __hash__(self):
        return hash((self.row, self.col))

    def __sub__(self, other):
        return Position(self.row - other.row, self.col - other.col)


class AgentData:
    def __init__(self):
        self.agent_id = -1
        self.complete = False
        self.waiting = False
        self.new_net = True
        self.valid_action = True
        self.prev_net = None
        self.cur_net = None
        self.prev_loc = None
        self.cur_loc = None
        self.prev_dir = None
        self.dir = None


if __name__ == '__main__':
    # init_fail_run()
    # init_fail_run()
    board = PCBBoard(rows=6, cols=6, num_agents=1)
    for i, agent in board.agents.items():
        print(str(i) + ": " + str(agent.cur_net))

    print(board.grid)
    print()
    print()
    board.print_grid()

    reward = 0
    reward += board.joint_act({0: 1})
    reward += board.joint_act({0: 1})
    reward += board.joint_act({0: 2})
    reward += board.joint_act({0: 2})
    reward += board.joint_act({0: 3})
    reward += board.joint_act({0: 3})
    reward += board.joint_act({0: 4})
    reward += board.joint_act({0: 5})
    reward += board.joint_act({0: 5})
    reward += board.joint_act({0: 5})

    print()
    board.print_grid()
    reward += board.joint_act({0: 2})
    reward += board.joint_act({0: 1})
    reward += board.joint_act({0: 1})
    reward += board.joint_act({0: 2})
    reward += board.joint_act({0: 3})
    reward += board.joint_act({0: 3})

    print()
    board.print_grid()
    reward += board.joint_act({0: 4})
    reward += board.joint_act({0: 4})

    print()
    board.print_grid()
    reward += board.joint_act({0: 3})
    reward += board.joint_act({0: 3})
    reward += board.joint_act({0: 3})
    reward += board.joint_act({0: 3})

    print()
    print(reward)
    renderer.render_board(board.rows, board.cols, board.nets, board.grid[::2, ::2])

