from __future__ import print_function

import os, sys, socket, shutil, tempfile
try:
    from MonetDBtesting import process
except ImportError:
    import process


def freeport():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


tmpdir = tempfile.mkdtemp()
s1p = freeport()
s1n = 's1'
server1 = process.server(mapiport=s1p, dbname=s1n, dbfarm=os.path.join(tmpdir, s1n), stdin=process.PIPE, stdout=process.PIPE)
client1 = process.client('sql', server=server1, stdin=process.PIPE, stdout=process.PIPE, stderr=process.PIPE)
client3 = process.client('sql', server=server1, stdin=process.PIPE, stdout=process.PIPE, stderr=process.PIPE)

s2p = freeport()
s2n = 's2'
server2 = process.server(mapiport=s2p, dbname=s2n, dbfarm=os.path.join(tmpdir, s2n), stdin=process.PIPE, stdout=process.PIPE)
client2 = process.client('sql', server=server2, stdin=process.PIPE, stdout=process.PIPE, stderr=process.PIPE)

out, err = client1.communicate("""
CREATE TABLE rt1 (col1 int);
CREATE TABLE rt2 (col1 int, col2 clob);
""")
sys.stdout.write(out)
sys.stderr.write(err)

out, err = client2.communicate("""
CREATE REMOTE TABLE rt1 (col1 int) ON 'mapi:monetdb://localhost:%d/%s';
CREATE REMOTE TABLE rt2 (col1 int, col2 clob) ON 'mapi:monetdb://localhost:%d/%s';

INSERT INTO rt1 VALUES (1);
INSERT INTO rt1 VALUES (1), (NULL), (3);
INSERT INTO rt2 VALUES (-32, 'hello');
INSERT INTO rt2 VALUES (1, 'another'), (NULL, NULL), (3, 'attempt');

SELECT col1 FROM rt1;
SELECT col1, col2 FROM rt2;

DROP TABLE rt1;
DROP TABLE rt2;
""" % (s1p, s1n, s1p, s1n))
sys.stdout.write(out)
sys.stderr.write(err)

out, err = client3.communicate("""
DROP TABLE rt1;
DROP TABLE rt2;
""")
sys.stdout.write(out)
sys.stderr.write(err)

sout, serr = server1.communicate()
sys.stdout.write(sout)
if serr is not None:
    sys.stderr.write(serr)

sout, serr = server2.communicate()
sys.stdout.write(sout)
if serr is not None:
    sys.stderr.write(serr)

shutil.rmtree(tmpdir)
