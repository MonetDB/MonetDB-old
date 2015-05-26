def fibonacci(nmbr):
    if (nmbr == 0): return 0
    if (nmbr == 1): return 1
    a = 0
    b = 1
    for i in range(0, nmbr - 1):
        c = a + b
        a = b
        b = c
    return b
