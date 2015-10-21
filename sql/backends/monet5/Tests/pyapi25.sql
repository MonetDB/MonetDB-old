
# Loopback query tests for mapped functions
START TRANSACTION;

# Use data from a different table in computation
CREATE TABLE pyapi09table(i integer);
INSERT INTO pyapi09table VALUES (1), (2), (3), (4);
CREATE TABLE pyapi09multiplication(i integer);
INSERT INTO pyapi09multiplication VALUES (3);

CREATE FUNCTION pyapi09(i integer) returns integer
language PYTHON_MAP
{
    res = _conn.execute('SELECT i FROM pyapi09multiplication;')
    return res['i'] * i
};

SELECT pyapi09(i) FROM pyapi09table; #multiply by 3
UPDATE pyapi09multiplication SET i=10;
SELECT pyapi09(i) FROM pyapi09table; #multiply by 10

DROP FUNCTION pyapi09;
DROP TABLE pyapi09table;
DROP TABLE pyapi09multiplication;

# Update table?
CREATE TABLE pyapi09table(i integer);
INSERT INTO pyapi09table VALUES (1), (2), (3), (4);
CREATE TABLE pyapi09multiplication(i integer);
INSERT INTO pyapi09multiplication VALUES (3);

CREATE FUNCTION pyapi09(i integer) returns integer
language PYTHON
{
    _conn.execute('UPDATE pyapi09multiplication SET i=20;')
    return i
};
CREATE FUNCTION pyapi09map(i integer) returns integer
language PYTHON_MAP
{
    _conn.execute('UPDATE pyapi09multiplication SET i=10;')
    return i
};

SELECT * FROM pyapi09multiplication; # 3
SELECT pyapi09(i) FROM pyapi09table;
SELECT * FROM pyapi09multiplication; # 20
SELECT pyapi09map(i) FROM pyapi09table;
SELECT * FROM pyapi09multiplication; # 20, update in PYTHON_MAP does not work (we should probably disable it)

DROP FUNCTION pyapi09;
DROP FUNCTION pyapi09map;
DROP TABLE pyapi09table;
DROP TABLE pyapi09multiplication;

ROLLBACK;

