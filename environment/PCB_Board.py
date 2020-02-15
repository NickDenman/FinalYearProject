import numpy as np
import random
import math
import copy


def get_dist(start_row, start_col, end_row, end_col):
    return math.hypot(end_row - start_row, end_col - start_col)


def get_next_pos(pos, direction, layer):
    next_row = pos.row
    next_col = pos.col

    if direction == 1 or direction == 2 or direction == 8:
        next_row = pos.row - 1

    if direction == 4 or direction == 5 or direction == 6:
        next_row = pos.row + 1

    if direction == 2 or direction == 3 or direction == 4:
        next_col = pos.col + 1

    if direction == 6 or direction == 7 or direction == 8:
        next_col = pos.col - 1

    return Position(next_row, next_col, layer)


def is_destination(cur_loc, dest):
    return cur_loc.row == dest.row and cur_loc.col == dest.col


class PCBBoard:
    def __init__(self, rows, cols, layers, min_net_dist, num_nets=10, num_agents=1):
        self.nets = self.generate_nets(rows, cols, num_nets, min_net_dist)
        self.grid = np.zeros((rows, cols, layers), dtype=int)
        self.action_grid = np.zeros((2 * rows, 2 * cols, layers), dtype=int)
        self.rows = rows
        self.cols = cols
        self.layers = layers
        self.num_agents = num_agents
        self.agents = {}

        self.initialise_grid()
        self.initialise_agent_data()

    def initialise_grid(self):
        for net in self.nets:
            # loop through layers
            for i in range(self.layers):
                self.set_via(net.start)
                self.set_via(net.end)

    def set_via(self, location):
        action_row = (location.row * 2)
        action_col = (location.col * 2)

        for i in range(self.layers):
            self.grid[location.row, location.col, i] = 1
            self.action_grid[action_row, action_col, i] = 31
            self.action_grid[action_row, action_col + 1, i] = 7
            self.action_grid[action_row + 1, action_col, i] = 14
            self.action_grid[action_row + 1, action_col + 1, i] = 24

    def generate_nets(self, rows, cols, num_nets, min_net_dist):
        nets = set([])

        while len(nets) < num_nets:
            start_row = random.randrange(rows)
            start_col = random.randrange(cols)
            end_row = random.randrange(rows)
            end_col = random.randrange(cols)

            while (start_row == end_row & start_col == end_col) \
                    or get_dist(start_row, start_col, end_row, end_col) < min_net_dist:
                end_row = random.randrange(rows)
                end_col = random.randrange(cols)

            nets.add(Net(Position(start_row, start_col, 0), Position(end_row, end_col, 0)))

        return nets

    def initialise_agent_data(self):
        i = 0

        while i < self.num_agents and len(self.nets) > 0:
            agent = AgentData()
            agent.agent_id = i
            agent.cur_net = self.nets.pop()
            agent.cur_loc = copy.deepcopy(agent.cur_net.start)
            agent.prev_dir = 9
            self.agents[i] = agent
            i += 1

    def update_action_grid(self, direction, agent_loc, layer):
        row_delta = 0
        col_delta = 0

        if direction == 1 or direction == 2 or direction == 8:
            row_delta = -1
        elif direction == 4 or direction == 5 or direction == 6:
            row_delta = 1

        if direction == 2 or direction == 3 or direction == 4:
            col_delta = -1
        elif direction == 6 or direction == 7 or direction == 8:
            col_delta = 1

        action_row = (agent_loc.row * 2) + row_delta
        action_col = (agent_loc.col * 2) + col_delta

        if direction == 1 or direction == 5:
            self.action_grid[action_row, action_col, layer] += 1
            self.action_grid[action_row, action_col + 1, layer] += 2
            self.action_grid[action_row + 1, action_col, layer] += 1
            self.action_grid[action_row + 1, action_col + 1, layer] += 2

        elif direction == 2 or direction == 6:
            self.action_grid[action_row, action_col + 1, layer] += 3
            self.action_grid[action_row + 1, action_col, layer] += 3

        elif direction == 3 or direction == 7:
            self.action_grid[action_row, action_col, layer] += 4
            self.action_grid[action_row, action_col + 1, layer] += 4
            self.action_grid[action_row + 1, action_col, layer] += 5
            self.action_grid[action_row + 1, action_col + 1, layer] += 5

        elif direction == 4 or direction == 8:
            self.action_grid[action_row, action_col, layer] += 6
            self.action_grid[action_row + 1, action_col + 1, layer] += 6

    def blind_act(self, agent_id, direction, layer):
        agent_loc = self.agents.get(agent_id).cur_loc
        if self.is_valid_move(agent_loc, direction, layer, agent_id):
            if direction == 9:
                self.set_via(agent_loc)
            else:
                self.update_action_grid(direction, agent_loc, layer)
                next_pos = get_next_pos(agent_loc, direction, layer)
                self.grid[next_pos.row, next_pos.col, next_pos.layer] = 1
                self.agents.get(agent_id).prev_loc = agent_loc
                self.agents.get(agent_id).cur_loc = next_pos

            self.agents.get(agent_id).prev_dir = direction

            if self.is_net_complete(agent_id):
                self.agents.get(agent_id).prev_net = self.agents.get(agent_id).cur_net

                if len(self.nets) > 0:
                    self.agents.get(agent_id).cur_net = self.nets.pop()
                    self.agents.get(agent_id).cur_loc = self.agents.get(agent_id).cur_net.start
                    self.agents.get(agent_id).prev_dir = 9

                else:
                    self.agents.get(agent_id).cur_net = None
                    self.agents.get(agent_id).cur_loc = None
                    self.agents.get(agent_id).complete = True

        else:
            print("invalid move: " + str(direction) + " for layer: " + str(layer) + " in position " + str(agent_loc))

    # def act(self, agent_id, direction, layer):
    #     agent_loc = self.agents.get(agent_id).cur_loc
    #     if self.is_valid_move(agent_loc, direction, layer, agent_id):
    #         if direction == 9:
    #             for i in range(self.layers):
    #                 self.grid[agent_loc.row, agent_loc.col, i] = 1
    #                 self.action_grid[agent_loc.row, agent_loc.col, i] = 9
    #
    #         else:
    #             next_pos = get_next_pos(agent_loc, direction, layer)
    #             self.grid[next_pos.row, next_pos.col, next_pos.layer] = 1
    #             # you should be updating the previous location, not the one you're going to...
    #             self.action_grid[next_pos.row, next_pos.col, next_pos.layer] = direction
    #             self.agents.get(agent_id).prev_loc = agent_loc
    #             self.agents.get(agent_id).cur_loc = next_pos
    #
    #         if self.is_net_complete(agent_id):
    #             self.agents.get(agent_id).prev_net = self.agents.get(agent_id).cur_net
    #
    #             if len(self.nets) > 0:
    #                 self.agents.get(agent_id).cur_net = self.nets.pop()
    #                 self.agents.get(agent_id).cur_loc = self.agents.get(agent_id).cur_net.start
    #
    #             else:
    #                 self.agents.get(agent_id).cur_net = None
    #                 self.agents.get(agent_id).cur_loc = None
    #                 self.agents.get(agent_id).complete = True
    #
    #     else:
    #         print("invalid move: " + str(direction) + " for layer: " + str(layer) + " in position " + str(agent_loc))

    def is_net_complete(self, agent_id):
        return self.agents.get(agent_id).cur_loc == self.agents.get(agent_id).cur_net.end

    def is_via(self, loc, layer):
        row = loc.row * 2
        col = loc.col * 2

        return self.action_grid[row, col, layer] == 31 \
            and self.action_grid[row, col + 1, layer] == 7 \
            and self.action_grid[row + 1, col, layer] == 14 \
            and self.action_grid[row + 1, col + 1, layer] == 24

    def is_valid_move(self, agent_loc, direction, layer, agent_id):
        if layer >= self.layers:
            return False

        # do via checks separately
        if direction == 9:
            for i in range(self.layers):
                if i == agent_loc.layer:
                    continue
                if self.grid[agent_loc.row, agent_loc.col, i] != 0:
                    return False

            return True

        else:
            # check if a via exists on given layer
            if layer != agent_loc.layer and not self.is_via(agent_loc, layer):
                return False

            # check if in boundary
            if agent_loc.row == 0 and (direction == 1 or direction == 2 or direction == 8):
                return False
            elif agent_loc.row == self.rows - 1 and (direction == 4 or direction == 5 or direction == 6):
                return False
            elif agent_loc.col == 0 and (direction == 6 or direction == 7 or direction == 8):
                return False
            elif agent_loc.col == self.cols - 1 and (direction == 2 or direction == 3 or direction == 4):
                return False

            prev_dir = self.agents.get(agent_id).prev_dir
            if prev_dir == 9 or prev_dir == direction - 1 or prev_dir == direction or prev_dir == direction + 1 \
                    or (prev_dir == 1 and direction == 8) or (prev_dir == 8 and direction == 1):
                # if next square is free go there
                next_pos = get_next_pos(agent_loc, direction, layer)

                # check if next grid is free or the net destination
                if self.grid[next_pos.row, next_pos.col, next_pos.layer] != 0 and \
                        is_destination(next_pos,self.agents.get(agent_id).cur_net.end):
                    return False

                return True


class Net:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.agent = -1

    def __eq__(self, other):
        if isinstance(other, Net):
            return self.start == other.start \
                   & self.end == other.end

        return False

    def __str__(self):
        return str(self.start) + " --> " + str(self.end)

    def __hash__(self):
        return hash((self.start, self.end))


class Position:
    def __init__(self, row=-1, col=-1, layer=-1):
        self.row = row
        self.col = col
        self.layer = layer

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.row == other.row \
                   & self.col == other.col \
                   & self.layer == other.layer

        return False

    def __str__(self):
        return "(" + str(self.row) + ", " + str(self.col) + ", " + str(self.layer) + ")"

    def __hash__(self):
        return hash((self.row, self.col, self.layer))


class AgentData:
    def __init__(self):
        self.agent_id = -1
        self.complete = False
        self.waiting = False
        self.prev_net = None
        self.cur_net = None
        self.prev_loc = None
        self.cur_loc = None
        self.prev_dir = None


if __name__ == '__main__':
    board = PCBBoard(10, 10, 3, 4.0, 5, 3)
    for net in board.nets:
        print(str(net))

    print(board.grid[:, :, 0])
    print
    print(board.action_grid[:, :, 0])
    print

    for aid, agent in board.agents.items():
        print(str(agent.agent_id) + ": " + str(agent.cur_net))
        print(str(agent.agent_id) + ": " + str(agent.prev_net))
        print(str(agent.agent_id) + ": " + str(agent.cur_loc))

    print
    for aid, agent in board.agents.items():
        agent.prev_net = agent.cur_net
        agent.cur_net = None
        agent.cur_loc.row = 69

    print
    for aid, agent in board.agents.items():
        print(str(agent.agent_id) + ": " + str(agent.cur_net))
        print(str(agent.agent_id) + ": " + str(agent.prev_net))
        print(str(agent.agent_id) + ": " + str(agent.cur_loc))
