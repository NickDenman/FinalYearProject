from enum import Enum

import numpy as np
import copy

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
        self.nets = set([])
        self.completed_nets = set([])

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
        self.agents.clear()
        self.completed_nets.clear()

        self.initialise_grid()
        self.initialise_agent_data()

    def add_via(self, start_row, start_col, end_row, end_col):
        self.grid[2 * start_row, 2 * start_col] = GridCells.VIA.value
        self.grid[2 * end_row, 2 * end_col] = GridCells.VIA.value
        self.nets.add(Net(Position(start_row, start_col), Position(end_row, end_col)))

    def initialise_grid(self):
        self.add_via(0, 0, 0, 3)
        self.add_via(1, 1, 2, 3)
        self.add_via(1, 0, 3, 0)
        self.add_via(2, 1, 3, 3)

    def initialise_agent_data(self):
        i = 0

        while i < self.num_agents and len(self.nets) > 0:
            agent = AgentData()
            agent.agent_id = i
            agent.cur_net = self.nets.pop()
            agent.cur_loc = copy.deepcopy(agent.cur_net.start)
            self.grid[2 * agent.cur_loc.row, 2 * agent.cur_loc.col] = GridCells.AGENT.value
            agent.dir = GridCells.VIA.value
            agent.new_net = True
            self.agents[i] = agent
            i += 1

    def is_game_over(self):
        game_over = True
        for _, agent in self.agents.items():
            game_over &= agent.complete

        return game_over

    def get_next_pos(self, pos, direction):
        grid_deltas = self.actions.get(direction)

        return Position(pos.row + grid_deltas[0], pos.col + grid_deltas[1])

    # return grid, current_location relative to the observation grid and destination coordinates relative to agent
    # TODO check if agent is done because they've got nada to observe
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

        # # insert agents location into observation map if not the current agent
        # for a_id, agent in self.agents.items():
        #     if not agent.complete and \
        #             a_id != agent_id and \
        #             row_start <= agent.cur_loc.row < row_end and \
        #             col_start <= agent.cur_loc.col < col_end:
        #         observation[agent.cur_loc.row, agent.cur_loc.col] = GridCells.AGENT.value

        for row in range(row_end - row_start):
            for col in range(col_end - col_start):
                if observation[row, col] > GridCells.VIA.value:
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
            self.agents.get(agent_id).cur_net.path.append(direction)
            self.agents.get(agent_id).valid_action = True
            next_pos = self.get_next_pos(agent_loc, direction)

            grid_deltas = self.actions.get(direction)

            if self.agents.get(agent_id).new_net:
                self.agents.get(agent_id).new_net = False
                self.grid[2 * agent_loc.row, 2 * agent_loc.col] = GridCells.VIA.value + direction
            else:
                self.grid[2 * agent_loc.row, 2 * agent_loc.col] = direction

            self.grid[(2 * agent_loc.row) + grid_deltas[0], (2 * agent_loc.col) + grid_deltas[1]] = direction
            self.agents.get(agent_id).prev_loc = agent_loc
            self.agents.get(agent_id).cur_loc = next_pos
            self.agents.get(agent_id).prev_dir = self.agents.get(agent_id).dir
            self.agents.get(agent_id).dir = direction

            if self.is_net_complete(agent_id):
                # print("Net complete...")
                self.completed_nets.add(self.agents.get(agent_id).cur_net)
                self.agents.get(agent_id).prev_net = self.agents.get(agent_id).cur_net
                self.agents.get(agent_id).prev_loc = self.agents.get(agent_id).cur_loc

                if len(self.nets) > 0:
                    self.agents.get(agent_id).cur_net = self.nets.pop()
                    self.agents.get(agent_id).cur_loc = self.agents.get(agent_id).cur_net.start
                    self.agents.get(agent_id).new_net = True

                else:
                    self.agents.get(agent_id).cur_net = None
                    self.agents.get(agent_id).cur_loc = None
                    self.agents.get(agent_id).complete = True

            else:
                self.grid[2 * next_pos.row, 2 * next_pos.col] = GridCells.AGENT.value

        else:
            self.agents.get(agent_id).valid_action = False
            # print("invalid move: " + str(direction) + " in position " + str(agent_loc) + " for agent " + str(agent_id))

    def step_reward_function(self, direction, prev_dir):
        if prev_dir < GridCells.VIA.value:
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

        else:
            reward += STEP_REWARD
            for _, agent in self.agents.items():
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
                if self.grid[row, col] < GridCells.VIA.value:
                    result[row, col] = action_symbols.get(self.grid[row, col])
                else:
                    result[row, col] = action_symbols.get(GridCells.VIA.value)

        for _row in range((self.rows * 2) - 1):
            for _col in range((self.cols * 2) - 1):
                print(result[_row, _col], end='')
            print()


class Net:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.agent = -1
        self.path = []

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


class GridCells(Enum):
    AGENT = 9
    N = 1
    NE = 2
    E = 3
    SE = 4
    S = 5
    SW = 6
    W = 7
    NW = 8
    VIA = 10
    VIA_N = 11
    VIA_NE = 12
    VIA_E = 13
    VIA_SE = 14
    VIA_S = 15
    VIA_SW = 16
    VIA_W = 17
    VIA_NW = 18


if __name__ == '__main__':
    # init_fail_run()
    board = PCBBoard(num_agents=1)
    for i, agent in board.agents.items():
        print(str(i) + ": " + str(agent.cur_net))

    print(board.grid)
    print()
    print()

    grid, location, destination = board.observe(0)
    print(grid)
    print("location: " + str(location))
    print("destination: " + str(destination))


    reward = board.joint_act({0:6})
    print("reward0: " + str(reward))

    reward = board.joint_act({0: 6})
    print("reward1: " + str(reward))

    reward = board.joint_act({0: 6})
    print("reward2: " + str(reward))

    board.print_grid()
