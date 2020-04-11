from environment.BinaryPCBBoard import BinaryPCBBoard
from environment.ZeroTwentyPCBBoard import ZeroTwentyPCBBoard

if __name__ == "__main__":
    board = ZeroTwentyPCBBoard(5, 5, rand_nets=False, filename="envs/small/5x5_5.txt", min_nets=4, max_nets=8, padded=True)
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
