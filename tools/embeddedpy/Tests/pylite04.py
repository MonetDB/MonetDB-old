# Test SQL types in monetdblite

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

import numpy
monetdblite.sql('CREATE TABLE pylite04_decimal(d DECIMAL(18,3))')
monetdblite.insert('pylite04_decimal', numpy.arange(100000))
print monetdblite.sql('SELECT * FROM pylite04_decimal')['d'].astype(numpy.int32)

monetdblite.sql('CREATE TABLE pylite04_date(d DATE)')
monetdblite.insert('pylite04_date', ['2000-01-01'])
print monetdblite.sql('SELECT * FROM pylite04_date')['d']


