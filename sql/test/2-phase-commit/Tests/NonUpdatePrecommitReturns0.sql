SELECT * FROM htmtest;
START TRANSACTION;
SELECT * FROM htmtest;
CALL precommit(2);
CALL persistcommit(2);
