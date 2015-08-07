START TRANSACTION;
INSERT INTO htmtest VALUES (40, 99), (41, 99), (42, 99);
SELECT * FROM htmtest;
CALL persistcommit(10);
ROLLBACK;

START TRANSACTION;
SELECT * FROM htmtest;
INSERT INTO htmtest VALUES (43, 99), (44, 99), (45, 99);
SELECT * FROM htmtest;
CALL precommit(10);
CALL persistcommit(10);
SELECT * FROM htmtest;
