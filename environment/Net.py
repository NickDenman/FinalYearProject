class Net:
    def __init__(self, net_id, start, end):
        self.net_id = net_id
        self.start = start
        self.end = end
        self.path = []

    def reset(self):
        self.path.clear()

    def __str__(self):
        return str(self.net_id) + ": " + str(self.start) + " -> " + str(self.end)

