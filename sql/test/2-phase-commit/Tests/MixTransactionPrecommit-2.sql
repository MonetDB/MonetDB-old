START TRANSACTION;
INSERT INTO htmtest VALUES (7, 99), (8, 99), (9, 99);
SELECT * FROM htmtest;
CALL precommit(3);
CALL persistcommit(3);

START TRANSACTION;
INSERT INTO htmtest VALUES (7, 99), (8, 99), (9, 99);
SELECT * FROM htmtest;
CALL precommit(3);
CALL persistcommit(3);
