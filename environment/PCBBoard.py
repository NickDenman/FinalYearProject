import abc
import math
import random

import gym
import numpy as np
import cairocffi as cairo

from environment.Net import Net
from environment.net_gen import NetGen

FINAL_REWARD = 10.0
NET_REWARD = 2.0
STEP_REWARD = -1.0
DIAGONAL_REWARD = -0.4  # diagonal lines are of length √2 ~= 1.4
DIRECTION_CHANGE_FACTOR = 0.0
STEP_FACTOR = 2

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
           1: (1.0,    0.5,    0.0),  # orange
           2: (1.0,    1.0,    0.0),  # yellow
           3: (1.0,    0.2,    0.8),  # pink
           4: (1.0,    0.0,    0.0),  # red
           5: (0.0,    1.0,    0.0),  # green
           6: (0.0,    0.0,    1.0),  # very blue
           7: (0.8,    0.0,    1.0),  # purple
           8: (0.0,    1.0,    1.0), }  # turquoise
blue = colours.get(0)


def read_file(filename):
    nets = {}
    with open(filename) as file:
        for idx, line in enumerate(file):
            x = line.split(", ")
            nets[idx] = Net(idx, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])))

    return nets


class PCBBoard(abc.ABC, gym.Env):
    def __init__(self, min_rows, min_cols, max_rows, max_cols, rand_nets, min_nets, max_nets, obstacle_value, blank_value, filename=None):
        super(PCBBoard, self).__init__()
        self.obstacle_value = obstacle_value
        self.blank_value = blank_value
        self.MAX_OBS_ROWS = 9
        self.MAX_OBS_COLS = 9

        self.min_nets = min_nets
        self.max_nets = max_nets

        self.middle_obs_row = self.MAX_OBS_ROWS // 2
        self.middle_obs_col = self.MAX_OBS_COLS // 2
        self.grid = None
        self.rows = min_rows
        self.cols = min_cols
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.obs_rows = self.MAX_OBS_ROWS
        self.obs_cols = self.MAX_OBS_COLS
        self._observation_row_start_offset = math.floor(self.obs_rows / 2)
        self._observation_row_end_offset = math.ceil(self.obs_rows / 2)
        self._observation_col_start_offset = math.floor(self.obs_cols / 2)
        self._observation_col_end_offset = math.ceil(self.obs_cols / 2)
        self.agent = None
        self.rand_nets = rand_nets
        if not rand_nets:
            self.nets = read_file(filename)
            self.total_nets = len(self.nets)

        n = int((self.rows * self.cols) ** .5)
        self.cur_net_id = 0
        self.total_reward = 0.0

        self.observation_space = gym.spaces.Box(low=0, high=20, shape=(self.get_observation_size(),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.get_action_size())

        self.total_steps = -1
        self.routed_nets = set()

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
    def observe(self):
        pass

    def increase_env_size(self, min_nets, max_nets):
        self.min_nets = min_nets
        self.max_nets = max_nets

    def get_action_size(self):
        return len(actions)

    def get_reward(self):
        r = STEP_REWARD

        if self.board_complete():
            r += FINAL_REWARD

        r += self.dir_change_reward(self.agent)

        if self.agent.net_done:
            self.agent.net_done = False
            r += NET_REWARD
            self.agent.prev_action = 0

        if self.agent.prev_action % 2 == 0:
            r += DIAGONAL_REWARD

        return r

    def board_complete(self):
        return self.agent.done

    def initialise_agent(self):
        agent = Agent()
        net = self.get_new_net()
        agent.net_id = net.net_id
        agent.location = net.start
        self.agent = agent

    def get_new_net(self):
        if len(self.routed_nets) == self.total_nets:
            return None

        shortest_net = None
        shortest_dist = math.inf
        for net_id, net in self.nets.items():
            if net_id not in self.routed_nets and net.dist < shortest_dist:
                shortest_net = net
                shortest_dist = net.dist

        self.routed_nets.add(shortest_net.net_id)
        return shortest_net

    def is_net_complete(self):
        return self.agent.location == self.nets.get(self.agent.net_id).end

    def get_next_pos(self, location, action):
        r, c = location
        r_delta, c_delta = actions.get(action)

        return r + r_delta, c + c_delta

    def reset(self):
        self.cur_net_id = 0
        self.current_step = 0
        if self.rand_nets:
            self.rows = random.randint(self.min_rows, self.max_rows)
            self.cols = random.randint(self.min_cols, self.max_cols)
            self.nets = NetGen.generate_board(self.rows, self.cols, self.min_nets, self.max_nets)
            self.total_nets = len(self.nets)
            self.total_steps = self.rows * self.cols * STEP_FACTOR
        else:
            for _, net in self.nets.items():
                net.reset()

        self.grid = np.full(shape=(self.rows, self.cols), fill_value=self.blank_value, dtype=np.int)
        self.routed_nets = set()
        self.initialise_agent()
        self.initialise_grid()
        self.total_reward = 0.0

        return self.observe()

    # TODO: bug here because it doesn't learn at all when this is used?
    def dir_change_reward(self, agent):
        if agent.prev_prev_action == 0 or agent.prev_action == 0:
            return 0

        prev_r, prev_c = actions.get(agent.prev_action)
        prev_prev_r, prev_prev_c = actions.get(agent.prev_prev_action)
        dir_delta = abs(prev_r - prev_prev_r) + abs(prev_c - prev_prev_c)
        if dir_delta <= 1:
            return 0

        return dir_delta * DIRECTION_CHANGE_FACTOR


    def close(self):
        pass

    def render_board(self, filename="pcb_render.png", width=None, height=None):
        width = int(1024 * (self.cols / max(self.rows, self.cols))) if width is None else width
        height = int(1024 * (self.rows / max(self.rows, self.cols))) if height is None else height

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
            self.render_net_endpoints(cr, row, col, radius, net_id)

            row, col = self.global_to_render(net.end, pixel_height, pixel_width)
            self.render_net_endpoints(cr, row, col, radius, net_id)

            if net.path:
                cr.new_path()
                (r, g, b) = self.get_colour(net_id)
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

    def render_net_endpoints(self, cr, row, col, radius, net_id):
        (r, g, b) = self.get_colour(net_id)
        cr.set_source_rgb(r, g, b)

        cr.arc(col, row, radius, 0, 2 * math.pi)
        cr.stroke()

    def get_colour(self, net_id):
        return colours.get(net_id)


class Agent:
    def __init__(self):
        self.location = (-1, -1)
        self.net_id = -1
        self.done = False
        self.net_done = False
        self.prev_action = 0
        self.prev_prev_action = 0
        self.invalid_move = False

    def reset(self):
        self.location = (-1, -1)
        self.net_id = -1
        self.done = False
        self.net_done = False
        self.prev_action = 0
        self.prev_prev_action = 0
        self.invalid_move = False
