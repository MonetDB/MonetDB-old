
f = open('workfile.sql', 'w')
import random
random.seed()
records = 15000000
f.write("START TRANSACTION;\n")
f.write("CREATE TABLE rval2 (a integer, b integer);\n");
f.write("copy " + str(records) + " records into rval2 from stdin;\n");
for i in range(0, records):
    x = random.randint(0,100)
    y = random.randint(0,100)
    f.write(str(x) + "|" + str(y) + "\n")

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
