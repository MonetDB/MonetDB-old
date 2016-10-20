-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

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
