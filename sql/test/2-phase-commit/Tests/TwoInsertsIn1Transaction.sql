-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (31, 99), (32, 99), (33, 99);
INSERT INTO htmtest VALUES (34, 99), (35, 99), (36, 99);
CALL precommit(7);
CALL persistcommit(7);
SELECT * FROM htmtest;

START TRANSACTION;
INSERT INTO htmtest VALUES (37, 99), (38, 99), (39, 99);
DELETE FROM htmtest WHERE id > 3;
CALL precommit(8);
CALL persistcommit(8);
SELECT * FROM htmtest;
