import math

from environment import ordinal_env, base_env
from environment.ordinal_env import OrdinalEnv
import environment.ordinal_env as BaseOE


class Node:
    def __init__(self, parent=None, position=None, net=None):
        self.g = 0
        self.h = 0
        self.parent = parent
        self.position = position
        self.action = -1
        self.net = net

    def __str__(self):
        return str(self.position) + " :: " + str(self.net)

    def get_f(self):
        return self.g + self.h

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position + (self.net, ))


class AStarNet:
    def __init__(self, start, end, h):
        self.start = start
        self.end = end
        self.h = h
        self.dist = self.__dist()

    def __dist(self):
        x1, y1 = self.start
        x2, y2 = self.end

        return math.hypot(x2 - x1, y2 - y1)


def orgainse_nets(nets):
    sorted_nets = sorted(nets.values(), key=lambda x: x.dist, reverse=True)
    a_star_nets = []

    remaining_dist = 0
    for sorted_net in sorted_nets:
        a_star_nets.append(AStarNet(sorted_net.start, sorted_net.end, remaining_dist))
        remaining_dist += sorted_net.dist

    a_star_nets.reverse()
    return {k: v for k, v in enumerate(a_star_nets)}


def is_valid_move(board, location, action, net, node):
    next_r, next_c = board.get_next_pos(location, action)

    # Check if new position is inside bounds of grid
    if next_r < 0 or next_r >= board.rows or \
            next_c < 0 or next_c >= board.cols:
        return False

    # Check if new position is free/not the destination
    if board.grid[next_r, next_c] != BaseOE.MAX_VALUE and \
            ((next_r, next_c) != net.end):
        return False
    delta_r = next_r - location[0]
    delta_c = next_c - location[1]

    while node is not None:
        if node.position == (next_r, next_c):
            return False
        if node.parent is not None and \
            (((location[0] + delta_r, location[1]) == node.position and
            (location[0], location[1] + delta_c) == node.parent.position) or
            ((location[0] + delta_r, location[1]) == node.parent.position and
             (location[0], location[1] + delta_c) == node.position)):
            return False

        node = node.parent

    # Check if diagonal lines cross
    if action % 2 == 0:
        r, c = location
        delta_r, delta_c = base_env.actions.get(action)

        a1 = ((action + 7 - (2 * delta_r * delta_c)) % 8) + 1
        a2 = ((action + 7 + (2 * delta_r * delta_c)) % 8) + 1

        if board.grid[r + delta_r, c] in BaseOE.grid_matrix[a1] or \
                board.grid[r, c + delta_c] in BaseOE.grid_matrix[a2]:
            return False

    return True


def get_next_pos(location, action):
    r, c = location
    r_delta, c_delta = base_env.actions.get(action)

    return r + r_delta, c + c_delta


def calculate_paths(current_node, nets):
    path = []
    paths = []
    current = current_node
    while current is not None:
        path.append(current.position)
        if current.position == nets[current.net].start:
            paths.append(path[::-1])
            path = []
        current = current.parent

    return paths[::-1]


def calculate_optimal_length(paths):
    total = 0
    for path in paths:
        for i in range(len(path) - 1):
            total += math.hypot(path[i][0] - path[i + 1][0], path[i][1] - path[i + 1][1])

    return total


def astar(board):
    nets = orgainse_nets(board.nets)

    start_node = Node(position=nets[0].start, net=0)
    end_node = Node(position=nets[len(nets) - 1].end, net=len(nets) - 1)
    open_list = {start_node}
    closed_list = set()

    while len(open_list) > 0:
        current_node = None
        min_f = math.inf
        for index, item in enumerate(open_list):
            if item.get_f() < min_f:
                min_f = item.get_f()
                current_node = item

        # Pop current off open list, add to closed list
        open_list.remove(current_node)
        closed_list.add(current_node)

        if current_node == end_node:
            paths = calculate_paths(current_node, nets)
            optimal_length = calculate_optimal_length(paths)

            return paths, optimal_length

        children = []
        for action in range(1, 9):  # Adjacent squares
            cur_net_id = current_node.net
            if not is_valid_move(board, current_node.position, action, nets[current_node.net], current_node):
                continue

            next_pos = get_next_pos(current_node.position, action)
            if next_pos == nets[cur_net_id].end and next_pos != end_node.position:
                new_node = Node(current_node, next_pos, cur_net_id)
                new_node.g = current_node.g + math.hypot(base_env.actions[action][0], base_env.actions[action][1])
                new_node.h = nets[cur_net_id].h
                current_node = new_node

                action = 0
                cur_net_id += 1
                next_pos = nets[cur_net_id].start

            # Create new node
            new_node = Node(current_node, next_pos, cur_net_id)
            new_node.g = current_node.g + math.hypot(base_env.actions[action][0], base_env.actions[action][1])
            new_node.h = math.hypot(nets[cur_net_id].end[0] - next_pos[0], nets[cur_net_id].end[1] - next_pos[1]) + nets[cur_net_id].h

            # Append
            children.append(new_node)

        for child in children:
            if child in closed_list:
                continue

            for open_node in open_list:
                if child == open_node and child.get_f() > open_node.get_f():
                    continue

                # Add the child to the open list
            open_list.add(child)


if __name__ == "__main__":
    env = OrdinalEnv(10, 15, 10, 15, 2, 4)
    for i in range(100):
        env.reset()
        env.render_board()
        path, optimal_length = astar(env)
        print(path)
        print(optimal_length)
        print()


