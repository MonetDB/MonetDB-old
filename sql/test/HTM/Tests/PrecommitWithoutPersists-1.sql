START TRANSACTION;
INSERT INTO htmtest VALUES (34, 99), (35, 99), (36, 99);
SELECT * FROM htmtest;
CALL precommit(8);
