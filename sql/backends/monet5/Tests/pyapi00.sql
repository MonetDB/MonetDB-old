START TRANSACTION;


CREATE FUNCTION pyapi00() returns table (d integer)
language P
{
	x = range(1,11)
	return(x)
};

SELECT * FROM pyapi00() AS R WHERE d > 5;
DROP FUNCTION pyapi00;

ROLLBACK;
