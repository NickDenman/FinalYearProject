from environment.BooleanEnv import BinaryPCBBoard
from environment.OrdinalEnv import OrdinalEnv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = OrdinalEnv(5, 5, 5, 5, 3, 3, rand_nets=False, filename="envs/invalid_1.txt", padded=True)
    env.reset()
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(5)
    env.step(6)
    env.render_board(filename="invalid_render1.png")

    env = OrdinalEnv(5, 5, 5, 5, 3, 3, rand_nets=False, filename="envs/invalid_2.txt", padded=True)
    env.reset()
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(1)
    env.render_board(filename="invalid_render2.png")

    for i in range(8):
        env = OrdinalEnv(5, 5, 5, 5, 3, 3, rand_nets=False, filename="envs/small/5x5_" + str(i) + ".txt", padded=True)
        env.render_board(filename="results/5x5_" + str(i) + ".png")

    board = OrdinalEnv(7, 7, 7, 7, 3, 3, rand_nets=True, padded=True)

    obs = board.reset()

    for i, net in board.nets.items():
        print(str(net))
    board.render_board()

    reward = 0.0

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 7
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 7
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r


    a = 6
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r


    a = 7
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 7
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 8
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    a = 5
    obs, r, d, _ = board.step(a)
    print(str(a) + ": " + str(r))
    reward += r

    print(d)
    print(reward)

    board.render_board()
