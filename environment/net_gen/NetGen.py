import random
from collections import defaultdict

from environment.Net import Net
from environment.net_gen.grid import Grid
from environment.net_gen.mitm import Mitm

LOOP_TRIES = 1000


def has_loops(grid, uf):
    """ Check whether the puzzle has loops not attached to an endpoint. """
    groups = len({uf.find((x, y)) for y in range(grid.h) for x in range(grid.w)})
    ends = sum(bool(grid[x, y] in 'v^<>') for y in range(grid.h) for x in range(grid.w))
    return ends != 2 * groups


def has_pair(tg, uf):
    """ Check for a pair of endpoints next to each other. """
    for y in range(tg.h):
        for x in range(tg.w):
            for dx, dy in ((1, 0), (0, 1)):
                x1, y1 = x + dx, y + dy
                if x1 < tg.w and y1 < tg.h:
                    if tg[x, y] == tg[x1, y1] == 'x' \
                            and uf.find( (x, y)) == uf.find( (x1, y1)):
                        return True
    return False


def has_tripple(tg, uf):
    """ Check whether a path has a point with three same-colored neighbours.
        This would mean a path is touching itself, which is generally not
        allowed in pseudo-unique puzzles.
        (Note, this also captures squares.) """
    for y in range(tg.h):
        for x in range(tg.w):
            r = uf.find( (x, y))
            nbs = 0
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                x1, y1 = x + dx, y + dy
                if 0 <= x1 < tg.w and 0 <= y1 < tg.h and uf.find( (x1, y1)) == r:
                    nbs += 1
            if nbs >= 3:
                return True
    return False


def make(w, h, mitm, min_numbers=0, max_numbers=1000):
    """ Creates a grid of size  w x h  without any loops or squares.
        The mitm table should be genearted outside of make() to give
        the best performance.
        """

    def test_ready(grid):
        # Test if grid is ready to be returned.
        sg = grid.shrink()
        stg, uf = sg.make_tubes()
        numbers = list(stg.values()).count('x') // 2
        return min_numbers <= numbers <= max_numbers \
                and not has_loops(sg, uf) \
                and not has_pair(stg, uf) \
                and not has_tripple(stg, uf) \

    # Internally we work on a double size grid to handle crossings
    grid = Grid(2 * w + 1, 2 * h + 1)

    gtries = 0
    while True:
        # Previous tries may have drawn stuff on the grid
        grid.clear()

        # Add left side path
        path = mitm.rand_path2(h, h, 0, -1)
        if not grid.test_path(path, 0, 0):
            continue
        grid.draw_path(path, 0, 0)
        # Draw_path doesn't know what to put in the first and last squares
        grid[0, 0], grid[0, 2 * h] = '\\', '/'

        # Add right side path
        path2 = mitm.rand_path2(h, h, 0, -1)
        if not grid.test_path(path2, 2 * w, 2 * h, 0, -1):
            continue
        grid.draw_path(path2, 2 * w, 2 * h, 0, -1)
        grid[2 * w, 0], grid[2 * w, 2 * h] = '/', '\\'

        # The puzzle might already be ready to return
        if test_ready(grid):
            return grid.shrink()

        # Add loops in the middle
        # Tube version of full grid, using for tracking orientations.
        # This doesn't make so much sense in terms of normal numberlink tubes.
        tg, _ = grid.make_tubes()
        # Maximum number of tries before retrying main loop
        for tries in range(LOOP_TRIES):
            x, y = 2 * random.randrange(w), 2 * random.randrange(h)

            # If the square square doen't have an orientation, it's a corner
            # or endpoint, so there's no point trying to add a loop there.
            if tg[x, y] not in '-|':
                continue

            path = mitm.rand_loop(clock=1 if tg[x, y] == '-' else -1)
            if grid.test_path(path, x, y):
                # A loop may not overlap with anything, and may even have
                # the right orientation, but if it 'traps' something inside it, that
                # might now have the wrong orientation.
                # Hence we clear the insides.
                grid.clear_path(path, x, y)

                # Add path and recompute orientations
                grid.draw_path(path, x, y, loop=True)
                tg, _ = grid.make_tubes()

                # Run tests to see if the puzzle is nice
                sg = grid.shrink()
                stg, uf = sg.make_tubes()
                numbers = list(stg.values()).count('x') // 2
                if numbers > max_numbers:
                    break
                if test_ready(grid):
                    return sg


def get_nets(grid):
    nets = defaultdict(list)
    tube_grid, uf = grid.make_tubes()
    ids = defaultdict(lambda: len(ids))
    for x in range(tube_grid.w):
        for y in range(tube_grid.h):
            if tube_grid[x, y] == 'x':
                net_id = ids[uf.find((x, y))]
                nets[net_id].append((y, x))

    net_list = {}
    for i, endpoints in nets.items():
        net = Net(i, endpoints[0], endpoints[1])
        net_list[i] = net

    return net_list


def generate_board(w, h, min_nets, max_nets):
    if w < 4 or h < 4:
        print('Please choose width and height at least 4.')
        return
    mitm = Mitm(lr_price=2, t_price=1)
    mitm.prepare(min(20, max(h, 6)))

    grid = make(w, h, mitm, min_nets, max_nets)

    return get_nets(grid)
