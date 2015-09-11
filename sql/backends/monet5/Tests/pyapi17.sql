START TRANSACTION;

# Return dictionary testing

# Standard case
CREATE FUNCTION pyapi17() returns TABLE (a STRING, b STRING, c INTEGER, d BOOLEAN)
language P
{
	retval = dict()
	retval['a'] = ['foo']
	retval['b'] = 'bar'
	retval['c'] = numpy.zeros(1, dtype=numpy.int32)
	retval['d'] = True
	return retval
};
SELECT * FROM pyapi17();
DROP FUNCTION pyapi17;

# Too many keys (prints warning if warnings are enabled)
CREATE FUNCTION pyapi17() returns TABLE (a STRING, b STRING, c INTEGER, d BOOLEAN)
language P
{
	retval = dict()
	retval['a'] = ['foo']
	retval['b'] = 'bar'
	retval['c'] = numpy.zeros(1, dtype=numpy.int32)
	retval['d'] = True
	retval['e'] = "Unused value"
	return retval
};
SELECT * FROM pyapi17();
DROP FUNCTION pyapi17;

# Keys are multidimensional arrays (cannot convert to single column, throws error)
CREATE FUNCTION pyapi17() returns TABLE (a STRING, b STRING, c INTEGER, d BOOLEAN)
language P
{
	retval = dict()
	retval['a'] = [['foo'], ['hello']]
	retval['b'] = 'bar'
	retval['c'] = numpy.zeros(1, dtype=numpy.int32)
	retval['d'] = True
	return retval
};
SELECT * FROM pyapi17();
ROLLBACK;

START TRANSACTION;
# Missing return value (throws error)
CREATE FUNCTION pyapi17() returns TABLE (a STRING, b STRING, c INTEGER, d BOOLEAN)
language P
{
	retval = dict()
	retval['a'] = ['foo']
	retval['b'] = 'bar'
	retval['d'] = True
	return retval
};
SELECT * FROM pyapi17();
ROLLBACK;

START TRANSACTION;
# Doesn't return a table, this means the return key must be 'result', so this throws an error
CREATE TABLE integers(i INTEGER);
INSERT INTO integers VALUES (3);
CREATE FUNCTION pyapi17(i integer) returns integer
language P
{
	retval = dict()
	retval['a'] = 33
	return retval
};
SELECT pyapi17(i) FROM integers;
ROLLBACK;

START TRANSACTION;
# Doesn't return a table, now with return key as 'result'
CREATE TABLE integers(i INTEGER);
INSERT INTO integers VALUES (3);
CREATE FUNCTION pyapi17(i integer) returns integer
language P
{
	retval = dict()
	retval['result'] = 33
	return retval
};
SELECT pyapi17(i) FROM integers;
DROP FUNCTION pyapi17;
ROLLBACK;
