INSERT INTO htmtest VALUES (19, 99), (20, 99), (21, 99);
SELECT * FROM htmtest;

START TRANSACTION;
DELETE FROM htmtest WHERE id > 3;
CALL precommit(7);
CALL persistcommit(7);

SELECT * FROM htmtest;
