--Tests for the ingestion route using a simple delta encoding on the timestamp

call timetrails.ingest('rooms',
'1434055562000000,L301,3,21.4\n'
'+15000,L301,3,21.4\n'
'+15000,L301,3,21.4\n'
'+15000,L301,3,21.5\n'
'+15000,L301,3,21.4\n'
'+15000,L301,3,21.4\n'
'+15000,L301,3,21.4\n'
'+15000,L301,3,21.5\n'
'+15000,L301,3,21.5\n'
'+15000,L301,3,21.5\n'
'+15000,L301,3,21.5\n'
'+15000,L301,3,21.5\n');
