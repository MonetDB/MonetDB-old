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
run(c1, 'INSERT INTO htmtest VALUES (40, 99), (41, 99), (42, 99)')
run(c1, 'COMMIT')
query(c1, 'SELECT * FROM htmtest')

c2 = connect(False)
query(c2, 'SELECT * FROM htmtest')

run(c1, 'DELETE FROM htmtest WHERE id > 3')
run(c1, 'COMMIT')
query(c1, 'SELECT * FROM htmtest')

query(c2, 'SELECT * FROM htmtest')

c3 = connect(False)
query(c3, 'SELECT * FROM htmtest')
