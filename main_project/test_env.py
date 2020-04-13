from environment.BinaryPCBBoard import BinaryPCBBoard
from environment.ZeroTwentyPCBBoard import ZeroTwentyPCBBoard

if __name__ == "__main__":
    board = ZeroTwentyPCBBoard(5, 5, 7, 7, rand_nets=True, padded=True)
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
