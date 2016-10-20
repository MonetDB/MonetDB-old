-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (34, 99), (35, 99), (36, 99);
SELECT * FROM htmtest;
CALL precommit(8);
