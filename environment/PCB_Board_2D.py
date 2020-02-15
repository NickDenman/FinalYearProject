import numpy as np
import random
import math
import copy

FINAL_REWARD = 10.0
NET_REWARD = 1.0
STEP_REWARD = -1.0
INVALID_ACTION_REWARD = -2.0


def test_act(action, agent_id):
    print(action)
    board.act(agent_id, action)
    reward = board.get_reward()
    print(reward)
    print(board.grid)
    print()
    print(board.action_grid)
    print()
    print()


def get_dist(start_row, start_col, end_row, end_col):
    return math.hypot(end_row - start_row, end_col - start_col)


class PCBBoard:
    def __init__(self, rows, cols, min_net_dist, observation_rows, observation_cols, num_nets=10, num_agents=1):
        self.grid = np.zeros((rows, cols), dtype=int)
        self.action_grid = np.zeros((rows, cols), dtype=int)
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.observation_rows = observation_rows if observation_rows < rows else rows
        self.observation_cols = observation_cols if observation_cols < cols else cols
        self.num_nets = num_nets
        self.min_net_dist = min_net_dist
        self.agents = {}
        self.nets = set([])

        self.actions = {1: [-1, 0],
                        2: [-1, 1],
                        3: [0, 1],
                        4: [1, 1],
                        5: [1, 0],
                        6: [1, -1],
                        7: [0, -1],
                        8: [-1, -1]}

    def reset_env(self):
        self.grid.fill(0)
        self.action_grid.fill(0)
        self.nets.clear()
        self.agents.clear()

        self.initialise_grid(self.rows, self.cols, self.num_nets, self.min_net_dist)
        self.initialise_agent_data()

    def initialise_grid(self, rows, cols, num_nets, min_net_dist):
        while len(self.nets) < num_nets:
            start_row = random.randrange(rows)
            start_col = random.randrange(cols)
            while self.grid[start_row, start_col] != 0:
                start_row = random.randrange(rows)
                start_col = random.randrange(cols)

            end_row = random.randrange(rows)
            end_col = random.randrange(cols)
            while self.grid[end_row, end_col] != 0 or get_dist(start_row, start_col, end_row, end_col) < min_net_dist:
                end_row = random.randrange(rows)
                end_col = random.randrange(cols)

            self.grid[start_row, start_col] = 1
            self.grid[end_row, end_col] = 1
            self.action_grid[start_row, start_col] = 9
            self.action_grid[end_row, end_col] = 9
            self.nets.add(Net(Position(start_row, start_col), Position(end_row, end_col)))

    def initialise_agent_data(self):
        i = 0

        while i < self.num_agents and len(self.nets) > 0:
            agent = AgentData()
            agent.agent_id = i
            agent.cur_net = self.nets.pop()
            agent.cur_loc = copy.deepcopy(agent.cur_net.start)
            agent.dir = 9
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
    def observe(self, agent_id):
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

        return self.grid[row_start:row_end, col_start:col_end], \
               Position(-row_delta, -col_delta), \
               self.agents.get(agent_id).cur_net.end - agent_loc

    def act(self, agent_id, direction):
        if direction == 9:
            print("shit...")

        agent_loc = self.agents.get(agent_id).cur_loc

        if self.is_valid_move(agent_loc, direction, agent_id):
            self.agents.get(agent_id).valid_action = True

            next_pos = self.get_next_pos(agent_loc, direction)
            self.grid[next_pos.row, next_pos.col] = 1
            self.action_grid[agent_loc.row, agent_loc.col] += direction
            self.agents.get(agent_id).prev_loc = agent_loc
            self.agents.get(agent_id).cur_loc = next_pos
            self.agents.get(agent_id).prev_dir = self.agents.get(agent_id).dir
            self.agents.get(agent_id).dir = direction

            if self.is_net_complete(agent_id):
                print("Net complete...")
                # update the final destination circle to have the incoming direction for rendering
                self.action_grid[next_pos.row, next_pos.col] += \
                    ((self.action_grid[agent_loc.row, agent_loc.col] + 3) % 8) + 1
                self.agents.get(agent_id).prev_net = self.agents.get(agent_id).cur_net
                self.agents.get(agent_id).prev_loc = self.agents.get(agent_id).cur_loc

                if len(self.nets) > 0:
                    self.agents.get(agent_id).cur_net = self.nets.pop()
                    self.agents.get(agent_id).cur_loc = self.agents.get(agent_id).cur_net.start
                    self.agents.get(agent_id).prev_dir = self.agents.get(agent_id).dir
                    self.agents.get(agent_id).dir = 9

                else:
                    self.agents.get(agent_id).cur_net = None
                    self.agents.get(agent_id).cur_loc = None
                    self.agents.get(agent_id).complete = True

        else:
            self.agents.get(agent_id).valid_action = False
            # print("invalid move: " + str(direction) + " in position " + str(agent_loc))

        return self.get_reward()

    def step_reward_function(self, direction, prev_dir):
        if prev_dir < 9:
            cur_dir_deltas = self.actions[direction]
            prev_dir_deltas = self.actions[prev_dir]

            dir_delta = abs(cur_dir_deltas[0] - prev_dir_deltas[0]) + abs(cur_dir_deltas[1] - prev_dir_deltas[1])
            if dir_delta <= 1:
                return STEP_REWARD
            return STEP_REWARD - (dir_delta * 1.0)

        return STEP_REWARD

    def get_reward(self):
        reward = 0.0

        if self.is_game_over():
            reward += FINAL_REWARD

        for _, agent in self.agents.items():
            if not agent.waiting and agent.prev_net is not None and agent.prev_net.end == agent.prev_loc:
                reward += NET_REWARD
                agent.waiting = agent.complete
            elif not agent.valid_action:
                reward += INVALID_ACTION_REWARD
            else:
                reward += self.step_reward_function(agent.dir, agent.prev_dir)

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
        if self.grid[next_pos.row, next_pos.col] != 0 and next_pos != self.agents.get(agent_id).cur_net.end:
            return False

        # check diagonals do not cross
        elif direction % 2 == 0 and not self.is_valid_diagonal(direction, agent_loc):
            return False

        return True

    def is_valid_diagonal(self, direction, location):
        if direction == 2 and \
                (self.action_grid[location.row - 1, location.col] % 9 == 4 or
                 self.action_grid[location.row, location.col + 1] % 9 == 8):
            return False
        elif direction == 4 and \
                (self.action_grid[location.row, location.col + 1] % 9 == 6 or
                 self.action_grid[location.row + 1, location.col] % 9 == 2):
            return False
        elif direction == 6 and \
                (self.action_grid[location.row, location.col - 1] % 9 == 4 or
                 self.action_grid[location.row + 1, location.col] % 9 == 8):
            return False
        elif direction == 8 and \
                (self.action_grid[location.row, location.col - 1] % 9 == 2 or
                 self.action_grid[location.row - 1, location.col] % 9 == 6):
            return False
        return True

    def print_grid(self):
        result = np.full(fill_value=" ", shape=(self.rows * 3, self.cols * 3), dtype=str)
        for row in range(self.rows):
            for col in range(self.cols):
                _row = (row * 3) + 1
                _col = (col * 3) + 1
                if self.action_grid[row, col] % 9 == 1:
                    result[_row, _col] = "|"
                    result[_row - 1, _col] = "|"
                    result[_row - 2, _col] = "|"

                elif self.action_grid[row, col] % 9 == 2:
                    result[_row, _col] = "/"
                    result[_row - 1, _col + 1] = "/"
                    result[_row - 2, _col + 2] = "/"

                elif self.action_grid[row, col] % 9 == 3:
                    result[_row, _col] = "-"
                    result[_row, _col + 1] = "-"
                    result[_row, _col + 2] = "-"

                elif self.action_grid[row, col] % 9 == 4:
                    result[_row, _col] = "\\"
                    result[_row + 1, _col + 1] = "\\"
                    result[_row + 2, _col + 2] = "\\"

                elif self.action_grid[row, col] % 9 == 5:
                    result[_row, _col] = "|"
                    result[_row + 1, _col] = "|"
                    result[_row + 2, _col] = "|"

                elif self.action_grid[row, col] % 9 == 6:
                    result[_row, _col] = "/"
                    result[_row + 1, _col - 1] = "/"
                    result[_row + 2, _col - 2] = "/"

                elif self.action_grid[row, col] % 9 == 7:
                    result[_row, _col] = "-"
                    result[_row, _col - 1] = "-"
                    result[_row, _col - 2] = "-"

                elif self.action_grid[row, col] % 9 == 8:
                    result[_row, _col] = "\\"
                    result[_row - 1, _col - 1] = "\\\\"
                    result[_row - 2, _col - 2] = "\\\\"

                if self.action_grid[row, col] > 8:
                    result[_row, _col] = "O"

        for row in range(self.rows * 3):
            for col in range(self.cols * 3):
                print(result[row, col], end='')
            print()


class Net:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.agent = -1

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
        self.valid_action = True
        self.prev_net = None
        self.cur_net = None
        self.prev_loc = None
        self.cur_loc = None
        self.prev_dir = None
        self.dir = None


if __name__ == '__main__':
    # init_fail_run()
    board = PCBBoard(10, 10, 4.0, 10, 10, 4, 1)
    for i, agent in board.agents.items():
        print(str(i) + ": " + str(agent.cur_net))

    print(board.grid)
    print()
    print(board.action_grid)
    print()
    print()

    grid, location, destination = board.observe(0)
    print(grid)
    print("location: " + str(location))
    print("destination: " + str(destination))

    # test_act(2, 0)
    # test_act(3, 0)
    # test_act(6, 0)
    # test_act(1, 0)
    # test_act(3, 0)
    # test_act(8, 0)
    #
    # board.print_grid()
