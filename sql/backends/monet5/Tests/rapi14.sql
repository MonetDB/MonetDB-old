START TRANSACTION;

CREATE TABLE dval(i integer);
INSERT INTO dval VALUES (1),(2),(3),(4);

CREATE FUNCTION rapi14() returns boolean
language P
{
    return [True, True, True, True]
};

SELECT rapi14() FROM dval;
DROP FUNCTION rapi14;
DROP TABLE dval;

ROLLBACK;

