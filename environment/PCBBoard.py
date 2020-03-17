import functools
import abc
import math

import gym
import numpy as np
import cairocffi as cairo

FINAL_REWARD = 50.0
NET_REWARD = 2.0
STEP_REWARD = -1.0
INVALID_ACTION_REWARD = -6.0
DIAGONAL_REWARD = -0.4  # diagonal lines are of length √2 ~= 1.4
DIRECTION_CHANGE_FACTOR = 1.5
NO_MOVE = -4.0

actions = {0: (0, 0),
           1: (-1, 0),
           2: (-1, 1),
           3: (0, 1),
           4: (1, 1),
           5: (1, 0),
           6: (1, -1),
           7: (0, -1),
           8: (-1, -1), }

colours = {0: (0.2588, 0.5294, 0.9608),  # nice blue
           1: (1.0, 0.5, 0.0),  # orange
           2: (1.0, 1.0, 0.0),  # yellow
           3: (1.0, 0.2, 0.8),  # pink
           4: (1.0, 0.0, 0.0),  # red
           5: (0.0, 1.0, 0.0),  # green
           6: (0.0, 0.0, 1.0),  # very blue
           7: (0.8, 0.0, 1.0),  # purple
           8: (0.0, 1.0, 1.0), }  # turquoise


def read_file(filename):
    file_content = []
    with open(filename) as file:
        for idx, line in enumerate(file):
            file_content.append(line)

    nets = {}
    net_id = 0
    for net in range(4, len(file_content)):
        x = file_content[net].split(", ")
        nets[net_id] = (Net(net_id, (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))))
        net_id += 1

    return int(file_content[0]), int(file_content[1]), int(file_content[2]), int(file_content[3]), nets


class PCBBoard(abc.ABC, gym.Env):
    def __init__(self, rows, cols, grid_rows, grid_cols, obs_rows, obs_cols, nets, num_agents):
        super(PCBBoard, self).__init__()
        self.grid = np.zeros(shape=(grid_rows, grid_cols), dtype=np.float32)
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.obs_rows = obs_rows if obs_rows < rows else rows
        self.obs_cols = obs_cols if obs_cols < cols else cols
        self._observation_row_start_offset = math.floor(self.obs_rows / 2)
        self._observation_row_end_offset = math.ceil(self.obs_rows / 2)
        self._observation_col_start_offset = math.floor(self.obs_cols / 2)
        self._observation_col_end_offset = math.ceil(self.obs_cols / 2)
        self.agents = {}
        self.nets = nets
        self.total_nets = len(nets)
        self.cur_net_id = 0

        self.observation_space = gym.spaces.Box(low=0, high=20, shape=(self.num_agents, self.get_observation_size(),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.get_action_size())

        self.initialise_agents()
        self.initialise_grid()

    @abc.abstractmethod
    def get_observation_size(self):
        pass

    @abc.abstractmethod
    def initialise_grid(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def observe(self, agent_id):
        pass

    def get_action_size(self):
        return len(actions)

    def get_reward(self):
        r = STEP_REWARD

        if self.board_complete():
            r += FINAL_REWARD

        for _, agent in self.agents.items():
            if agent.moved:
                r += self.dir_change_reward(agent)

                if agent.net_done:
                    # print("net")
                    # self.render_board()
                    agent.net_done = False
                    r += NET_REWARD
                    agent.prev_action = 0

                if agent.prev_action % 2 == 0:
                    r += DIAGONAL_REWARD
            elif agent.done:
                pass
            else:
                r += NO_MOVE

            if agent.invalid_move:
                r += INVALID_ACTION_REWARD

        return r

    def board_complete(self):
        return functools.reduce(lambda a, b: a & b, [c.done for _, c in self.agents.items()])

    def initialise_agents(self):
        agent_id = 0

        while agent_id < self.num_agents and self.cur_net_id < len(self.nets):
            agent = Agent(agent_id)
            net = self.get_new_net()
            agent.net_id = net.net_id
            net.agent_id = agent.agent_id
            agent.location = net.start
            self.agents[agent_id] = agent

            agent_id += 1

    def get_new_net(self):
        if self.cur_net_id >= self.total_nets:
            return None

        net = self.nets.get(self.cur_net_id)
        self.cur_net_id += 1

        return net

    def is_net_complete(self, agent):
        return agent.location == self.nets.get(agent.net_id).end

    def get_next_pos(self, location, action):
        r, c = location
        r_delta, c_delta = actions.get(action)

        return r + r_delta, c + c_delta

    def reset(self):
        self.grid.fill(0)
        self.cur_net_id = 0
        for _, net in self.nets.items():
            net.reset()
        for _, agent in self.agents.items():
            agent.reset()
        self.initialise_agents()
        self.initialise_grid()

        return [self.observe(i) for i in range(self.num_agents)]

    def dir_change_reward(self, agent):
        if agent.prev_prev_action == 0 or agent.prev_action == 0:
            return 0

        prev_r, prev_c = actions.get(agent.prev_action)
        prev_prev_r, prev_prev_c = actions.get(agent.prev_prev_action)
        dir_delta = abs(prev_r - prev_prev_r) + abs(prev_c - prev_prev_c)
        if dir_delta <= 1:
            return 0

        return -dir_delta * DIRECTION_CHANGE_FACTOR

    def get_agent_statuses(self):
        return [a.done for _, a in self.agents.items()]

    def close(self):
        pass

    def render_board(self, filename="pcb_render.png", width=1024, height=1024):
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        cr = cairo.Context(ims)

        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)

        pixel_height = height // (2 * self.rows)
        pixel_width = width // (2 * self.cols)

        line_width = min(pixel_height / 8, pixel_width / 8)
        cr.set_line_width(line_width)

        radius = min(pixel_height / 4, pixel_width / 4)

        for net_id, net in self.nets.items():
            row, col = self.global_to_render(net.start, pixel_height, pixel_width)
            self.render_net_endpoints(cr, row, col, radius, net.agent_id)

            row, col = self.global_to_render(net.end, pixel_height, pixel_width)
            self.render_net_endpoints(cr, row, col, radius, net.agent_id)

            if net.path:
                cr.new_path()
                (r, g, b) = self.get_colour(net.agent_id)
                cr.set_source_rgb(r, g, b)
                render_row, render_col = self.global_to_render(net.start, pixel_height, pixel_width)

                r_delta, c_delta = actions.get(net.path[0])
                h = math.hypot(r_delta, c_delta)
                start_row = render_row + radius * r_delta / h
                start_col = render_col + radius * c_delta / h

                cr.move_to(start_col, start_row)
                row, col = net.start

                for dir in net.path:
                    r_delta, c_delta = actions.get(dir)
                    row += r_delta
                    col += c_delta
                    render_row, render_col = self.global_to_render((row, col), pixel_height, pixel_width)

                    if (row, col) == net.end:
                        h = math.hypot(r_delta, c_delta)
                        render_row -= radius * r_delta / h
                        render_col -= radius * c_delta / h

                    cr.line_to(render_col, render_row)
                cr.stroke()
            ims.write_to_png(filename)

    def global_to_render(self, pos, pixel_height, pixel_width):
        r, c = pos
        return ((r * 2) + 1) * pixel_height, ((c * 2) + 1) * pixel_width

    def render_net_endpoints(self, cr, row, col, radius, agent_id):
        (r, g, b) = self.get_colour(agent_id)
        cr.set_source_rgb(r, g, b)

        cr.arc(col, row, radius, 0, 2 * math.pi)
        cr.stroke()

    def get_colour(self, colour_id):
        return colours.get(colour_id, (1.0, 1.0, 1.0))


class Net:
    def __init__(self, net_id, start, end):
        self.net_id = net_id
        self.start = start
        self.end = end
        self.path = []
        self.agent_id = -1

    def reset(self):
        self.agent_id = -1
        self.path.clear()


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.location = (-1, -1)
        self.net_id = -1
        self.moved = False
        self.done = False
        self.net_done = False
        self.prev_action = 0
        self.prev_prev_action = 0
        self.invalid_move = False

    def reset(self):
        self.location = (-1, -1)
        self.net_id = -1
        self.moved = False
        self.done = False
        self.net_done = False
        self.prev_action = 0
        self.prev_prev_action = 0
        self.invalid_move = False
