START TRANSACTION;

CREATE TABLE rval(i string,j string);
INSERT INTO rval VALUES ('1' ,'4'), ('2','3'), ('3','2'), ('4','1');

CREATE FUNCTION pyapi02() returns integer
language P 
{
	return([[0,0,0]])
};

SELECT pyapi02() FROM rval;
DROP FUNCTION pyapi02;
DROP TABLE rval;

ROLLBACK;
