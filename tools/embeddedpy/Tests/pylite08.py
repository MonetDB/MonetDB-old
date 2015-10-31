# test error statements

import monetdblite, shutil, os
dbfarm = '/tmp/pylite_dbfarm'
if os.path.isdir(dbfarm): shutil.rmtree(dbfarm)

import sys
print '\n--select before init--'
try: monetdblite.sql('select * from tables')
except: print str(sys.exc_info()[1])

print '\n--init with weird argument--'
try: monetdblite.init(33)
except: print str(sys.exc_info()[1])

print '\n--init in unwritable directory--'
try: monetdblite.init('/unwritabledir')
except: print str(sys.exc_info()[1])

print '\n--proper init--'
monetdblite.init(dbfarm)

print '\n--select from non-existent table--'
try: monetdblite.sql('select * from nonexistenttable')
except: print str(sys.exc_info()[1])

print '\n--invalid connection object--'
try: monetdblite.sql('select * from tables', conn=33)
except: print str(sys.exc_info()[1])

print '\n--no colnames with list--'
try: monetdblite.create('pylite08', values=[[]])
except: print str(sys.exc_info()[1])

print '\n--invalid colnames--'
try: monetdblite.create('pylite08', values=[[]], colnames=33)
except: print str(sys.exc_info()[1])

print '\n--empty colnames--'
try: monetdblite.create('pylite08', values=[[]], colnames=[])
except: print str(sys.exc_info()[1])

print '\n--too many colnames for values--'
try: monetdblite.create('pylite08', values=[[]], colnames=['a', 'b', 'c'])
except: print str(sys.exc_info()[1])

print '\n--too few colnames for values--'
try: monetdblite.create('pylite08', values=[[33], [44], [55]], colnames=['a'])
except: print str(sys.exc_info()[1])

print '\n--dictionary with invalid keys--'
d = dict()
d[33] = 44
try: monetdblite.create('pylite08', d)
except: print str(sys.exc_info()[1])

monetdblite.create('pylite08', dict(a=[],b=[],c=[]))

print '\n--missing dict key in insert--'
try: monetdblite.insert('pylite08', dict(a=33,b=44))
except: print str(sys.exc_info()[1])

print '\n--too few columns in insert--'
try: monetdblite.insert('pylite08', [[33],[44]])
except: print str(sys.exc_info()[1])
