import math


class Net:
    def __init__(self, net_id, start, end):
        self.net_id = net_id
        self.start = start
        self.end = end
        self.path = []
        self.dist = self.__dist()

    def reset(self):
        self.path.clear()

    def __dist(self):
        x1, y1 = self.start
        x2, y2 = self.end

        return math.hypot(x2 - x1, y2 - y1)

    def __str__(self):
        return str(self.net_id) + ": " + str(self.start) + " -> " + str(self.end)

