START TRANSACTION;
INSERT INTO htmtest VALUES (10, 99), (11, 99), (12, 99);
ROLLBACK;
SELECT * FROM htmtest;
CALL precommit(4);
CALL persistcommit(4);
SELECT * FROM htmtest;
