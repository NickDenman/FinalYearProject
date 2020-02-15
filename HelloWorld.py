def recursive_fib(n = 7):
    if n < 0:
        return -1
    if n < 2:
        return n

    return recursive_fib(n - 1) + recursive_fib(n - 2)


def iterative_fib(n = 7):
    a, b = 0, 1
    for i in range(0, n):
        a, b = b, a + b
    return a


if __name__ == '__main__':
    print("Hello World!")
    print("Recursive Default: " + str(recursive_fib()))
    print("Recursive Fib 5: " + str(recursive_fib(5)))
    print("Recursive Fib 10: " + str(recursive_fib(10)))

    print("Iterative Default: " + str(iterative_fib()))
    print("Iterative Fib 5: " + str(iterative_fib(5)))
    print("Iterative Fib 10: " + str(iterative_fib(10)))
