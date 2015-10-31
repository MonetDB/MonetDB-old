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

#sequential

start = time.time()
for n in range(10):
    query(default_conn)
end = time.time()

sequential_time = end - start

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
# test threading with different clients (every client has its own SQL transaction)
threads = []
for n in range(10):
    thread = threading.Thread(target=query, args=[monetdblite.connect()])
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
end = time.time()

different_client_time = end - start

# Time-wise, we expect Different Clients > Same Client > Sequential
# Same Client and Sequential should be close, because the actual query is not parallelized when the same client is used
# Same Client is slightly faster because there is a little bit of work the other threads can do while the GIL is released during the query
#    but since the bulk of the work is in the query, this does not make a huge difference
# Different Clients completely parallelize the query as well, and since the bulk of the work is in the query this should be the fastest by far

# we print the ranking instead of actual times because Mtest.py requires the same output when run multiple times
# and measured times will differ for each run/for every machine, but rankings should not differ (unless run on a 1 core machine)
a = [('Threads + Different Clients', different_client_time), ('Threads + Same Client', same_client_time), ('Sequential (No Threads)', sequential_time)]
a = sorted(a, key=lambda x : x[1])
print 'Rankings (Fastest to Slowest)'
for i in range(len(a)):
    print '%d: %s' % (i + 1, a[i][0])

#print "Sequential", sequential_time
#print "Same", same_client_time
#print "Different", different_client_time

