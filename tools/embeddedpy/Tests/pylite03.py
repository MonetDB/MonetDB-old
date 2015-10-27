# monetdblite.create statement with multiple columns

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy
monetdblite.create('pylite03', ['i', 'j', 'k', 'l', 'm'], numpy.arange(100000).reshape((5,20000)))
res = monetdblite.sql('select * from pylite03')
print res
print 'Count', len(res['i'])

shutil.rmtree(dbfarm)

