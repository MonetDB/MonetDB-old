
# test inserting dictionaries

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy

# create table using dict
a = dict()
numpy.random.seed(33)
a['i'] = numpy.random.randint(0, 1000, 100000)
a['j'] = numpy.random.randint(0, 1000, 100000)
a['k'] = numpy.random.randint(0, 1000, 100000)
a['l'] = numpy.random.randint(0, 1000, 100000)
a['m'] = numpy.random.randint(0, 1000, 100000)

monetdblite.create('pylite07', a)
arr = monetdblite.sql('select * from pylite07')
print arr
print len(arr['i'])

# create empty table
monetdblite.create('pylite07a', colnames=['a', 'b', 'c', 'd', 'e' , 'f'], values=[[],[],[],[],[],[]])
# insert data using dictionary
b = dict()
b['a'] = numpy.random.randint(0, 1000, 100000)
b['b'] = numpy.random.randint(0, 1000, 100000)
b['c'] = numpy.random.randint(0, 1000, 100000)
b['d'] = numpy.random.randint(0, 1000, 100000)
b['e'] = numpy.random.randint(0, 1000, 100000)
b['f'] = numpy.random.randint(0, 1000, 100000)

monetdblite.insert('pylite07a', b)
arr = monetdblite.sql('select * from pylite07a')
print arr
print len(arr['a'])

