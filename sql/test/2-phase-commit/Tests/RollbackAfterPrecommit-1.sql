-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (10, 99), (11, 99), (12, 99);
CALL precommit(5);
ROLLBACK;
CALL persistcommit(5);
SELECT * FROM htmtest;
