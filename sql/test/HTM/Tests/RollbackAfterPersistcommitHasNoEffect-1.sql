START TRANSACTION;
INSERT INTO htmtest VALUES (31, 99), (32, 99), (33, 99);
CALL precommit(8);
CALL persistcommit(8);

ROLLBACK;

SELECT * FROM htmtest;
