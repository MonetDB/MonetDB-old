# Test the hidden _values dictionary that is used by the user to store Python objects to be used by later python functions
START TRANSACTION;

CREATE TABLE vals(i INTEGER);
INSERT INTO vals VALUES (1), (2), (3), (4), (5), (6), (7);

CREATE FUNCTION pyapi18a(i integer) returns integer
language P
{
	_values['number'] = 42
	return(i)
};

CREATE FUNCTION pyapi18b(i integer) returns integer
language P
{
	return _values['number'] * i
};
SELECT pyapi18b(pyapi18a(i)) FROM vals;
DROP FUNCTION pyapi18a; DROP FUNCTION pyapi18b;
DROP TABLE vals;

ROLLBACK;
