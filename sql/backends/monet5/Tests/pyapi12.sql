START TRANSACTION;

CREATE TABLE rval(i integer);
INSERT INTO rval VALUES (1),(2),(3),(4),(-1),(0);

CREATE FUNCTION pyapi12(i integer,z integer) returns boolean language PYTHON_MAP
{
	return(numpy.greater(i,z))
};
SELECT * FROM rval WHERE pyapi12(i,2);
DROP FUNCTION pyapi12;
DROP TABLE rval;


ROLLBACK;
