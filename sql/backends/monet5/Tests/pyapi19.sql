START TRANSACTION;

CREATE TABLE vals(i INTEGER);
INSERT INTO vals VALUES (1), (2), (3), (4), (5), (6), (7);

# Directly store input in the dictionary doesn't work (because we do zero copy)
CREATE FUNCTION pyapi19a(i integer) returns integer
language P
{
	_values['array'] = i;
	return(i * 2)
};
SELECT pyapi19a(i * 2) FROM vals; #so this throws an error
ROLLBACK;

START TRANSACTION;
CREATE TABLE vals(i INTEGER);
INSERT INTO vals VALUES (1), (2), (3), (4), (5), (6), (7);

# Correct way of doing it
CREATE FUNCTION pyapi19a(i integer) returns integer
language P
{
	_values['array'] = numpy.copy(i);
	return(i * 2)
};
CREATE FUNCTION pyapi19b(i integer) returns integer
language P
{
	return _values['array'] * 2
};

SELECT pyapi19a(i) FROM vals; #so this throws an error
SELECT pyapi19b(i) FROM vals;

DROP FUNCTION pyapi19a; DROP FUNCTION pyapi19b;
DROP TABLE vals;

ROLLBACK;
