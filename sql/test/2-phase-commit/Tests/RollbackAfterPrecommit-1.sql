START TRANSACTION;
INSERT INTO htmtest VALUES (10, 99), (11, 99), (12, 99);
CALL precommit(5);
ROLLBACK;
CALL persistcommit(5);
SELECT * FROM htmtest;
