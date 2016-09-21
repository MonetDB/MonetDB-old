try:
    from MonetDBtesting import process
except ImportError:
    import process

import sys, time, monetdb.sql, os


def connect(autocommit):
    return monetdb.sql.connect(database = os.getenv('TSTDB'),
                               hostname = 'localhost',
                               port = int(os.getenv('MAPIPORT')),
                               username = 'monetdb',
                               password = 'monetdb',
                               autocommit = autocommit)

def query(conn, sql):
    print(sql)
    cur = conn.cursor()
    cur.execute(sql)
    r = cur.fetchall()
    cur.close()
    print(r)

def run(conn, sql):
    print(sql)
    r = conn.execute(sql)
    print(r)


c1 = connect(False)
run(c1, 'INSERT INTO htmtest VALUES (4, 40)')

c2 = connect(False)
run(c2, 'INSERT INTO htmtest VALUES (5, 50)')

#c3 = connect(False)
#run(c3, 'INSERT INTO htmtest VALUES (6, 60)')

run(c2, 'CALL precommit(2)')
try:
    run(c1, 'CALL precommit(1)')
except:
    print "precommit failed\n"
#run(c3, 'CALL precommit(3)')

try:
    run(c1, 'CALL persistcommit(1)')
except:
    print "persistcommit failed\n"
run(c2, 'CALL persistcommit(2)')

try:
    query(c1, 'SELECT * FROM htmtest')
except:
    print "select fails on aborted transaction\n"
query(c2, 'SELECT * FROM htmtest')
