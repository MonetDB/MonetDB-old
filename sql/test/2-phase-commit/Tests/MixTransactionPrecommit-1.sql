-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (7, 99), (8, 99), (9, 99);
SELECT * FROM htmtest;
CALL precommit(3);
CALL persistcommit(3);
