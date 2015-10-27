# Test monetdblite.insert statement

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy
monetdblite.create('pylite02', ['i'], numpy.arange(100000))
monetdblite.insert('pylite02', numpy.arange(100000))
res = monetdblite.sql('select * from pylite02')
print res
print 'Count', len(res['i'])

shutil.rmtree(dbfarm)

