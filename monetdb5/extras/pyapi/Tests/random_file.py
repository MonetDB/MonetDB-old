for j in range(1,2):
    f = open('workfile' + str(j) + '.sql', 'w')
    import random
    random.seed()
    f.write("START TRANSACTION;\n")
    for i in range(1, 50000):
        f.write("INSERT INTO rval VALUES ")
        x = random.randint(0,100)
        y = random.randint(0,100)
        f.write("(" + str(x) + "," + str(y) + ");\n")
    f.write("COMMIT;\n")
    f.close()


\<workfile1.sql
\<workfile2.sql
\<workfile3.sql
\<workfile4.sql
\<workfile5.sql
\<workfile6.sql
\<workfile7.sql
\<workfile8.sql
\<workfile9.sql
\<workfile10.sql
\<workfile11.sql
\<workfile12.sql
\<workfile13.sql
\<workfile14.sql
\<workfile15.sql
