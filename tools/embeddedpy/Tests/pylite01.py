# Test monetdblite.create statement

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy
monetdblite.create('pylite01', ['i'], numpy.arange(100000))
res = monetdblite.sql('select * from pylite01')
print res
print 'Count', len(res['i'])

shutil.rmtree(dbfarm)

