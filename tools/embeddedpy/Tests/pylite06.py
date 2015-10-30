# Test multithreading in MonetDBLite

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy, threading, time

monetdblite.create('pylite06a', colnames=['i', 'j'], values=[numpy.arange(1000000), numpy.arange(1000000)])
monetdblite.create('pylite06b', colnames=['i', 'k'], values=[numpy.arange(1000000), numpy.arange(1000000)])
# commit the transaction, otherwise different clients do not see the table
# every connection has its own transaction state
monetdblite.sql('commit')

default_conn = monetdblite.connect()

def query(conn):
    # do a query that takes a reasonable amount of time in the database, as time in the database is the only thing that is parallelized
    monetdblite.sql('select j,k from pylite06a inner join pylite06b on pylite06a.i=pylite06b.i', conn)

# test threading with every thread using the same client (only one SQL transaction per client, so the threads don't do much)
start = time.time()
threads = []
for n in range(10):
    thread = threading.Thread(target=query, args=[default_conn])
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
end = time.time()

same_client_time = end - start

del default_conn

start = time.time()
# test threading with different clients (every client can have its own SQL transaction)
threads = []
for n in range(10):
    thread = threading.Thread(target=query, args=[monetdblite.connect()])
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
end = time.time()

different_client_time = end - start

if different_client_time > same_client_time:
    print "Same client is faster"
else: print "Different client is faster"

