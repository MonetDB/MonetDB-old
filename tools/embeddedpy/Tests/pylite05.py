# Test connections in monetdblite

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy

conn = monetdblite.connect() # create the client
monetdblite.create('pylite05', colnames=['i'], values=numpy.arange(100000), conn=conn)
print len(monetdblite.sql('select * from pylite05', conn=conn)['i'])
monetdblite.insert('pylite05', values=numpy.arange(100000), conn=conn)
print len(monetdblite.sql('select * from pylite05', conn=conn)['i'])
del conn # client is automatically disconnected when connection object is deleted


