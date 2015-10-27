# Test basic monetdblite.sql statements

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)
monetdblite.init(dbfarm)

monetdblite.sql('CREATE TABLE pylite00 (i INTEGER)')
monetdblite.sql('INSERT INTO pylite00 VALUES (1), (2), (3), (4), (5)')
print monetdblite.sql('SELECT * FROM pylite00')

shutil.rmtree(dbfarm)

