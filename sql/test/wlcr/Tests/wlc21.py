from __future__ import print_function

try:
    from MonetDBtesting import process
except ImportError:
    import process
import os, sys

dbfarm = os.getenv('GDK_DBFARM')
tstdb = os.getenv('TSTDB')

if not tstdb or not dbfarm:
    print('No TSTDB or GDK_DBFARM in environment')
    sys.exit(1)

dbname = tstdb

s = None
try:
    s = process.server(dbname = dbname, stdin = process.PIPE, stdout = process.PIPE, stderr = process.PIPE)

    c = process.client('sql', server = s, stdin = process.PIPE, stdout = process.PIPE, stderr = process.PIPE)

    cout, cerr = c.communicate('''\
insert into tmp values(5,'red'),(6,'fox');
select * from tmp;
''')

    sout, serr = s.communicate()

    sys.stdout.write(sout)
    sys.stdout.write(cout)
    sys.stderr.write(serr)
    sys.stderr.write(cerr)
finally:
    if s is not None:
        s.terminate()

def listfiles(path):
    sys.stdout.write("#LISTING OF THE LOG FILES\n")
    for f in sorted(os.listdir(path)):
        if (f.find('wlc') >= 0 or f.find('wlr') >=0 ) and f != 'wlc_logs':
            file = path + os.path.sep + f
            sys.stdout.write('#' + file + "\n")
            try:
                x = open(file)
                s = x.read()
                lines = s.split('\n')
                for l in lines:
                    sys.stdout.write('#' + l + '\n')
                x.close()
            except IOError:
                sys.stderr.write('Failure to read file ' + file + '\n')

listfiles(os.path.join(dbfarm, tstdb))
listfiles(os.path.join(dbfarm, tstdb, 'wlc_logs'))
