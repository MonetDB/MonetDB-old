for j in range(1,15):
    f = open('workfile' + str(j) + '.sql', 'w')
    import random
    random.seed()
    f.write("START TRANSACTION;\n")
    for i in range(1, 1000000):
        f.write("INSERT INTO rval VALUES ")
        x = random.randint(0,100)
        y = random.randint(0,100)
        f.write("(" + str(x) + "," + str(y) + ");\n")
    f.write("COMMIT;\n")
    f.close()
