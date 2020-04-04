import random

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.ptr = 0

    def append(self, s, a, r, s_prime, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (s, a, r, s_prime, done)
        self.ptr = (self.ptr + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, d = map(np.array, zip(*batch))

        return s, a, r, s_prime, d

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        pass

    def load(self, path):
        pass