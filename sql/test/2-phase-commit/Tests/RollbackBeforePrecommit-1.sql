-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (10, 99), (11, 99), (12, 99);
ROLLBACK;
SELECT * FROM htmtest;
CALL precommit(4);
CALL persistcommit(4);
SELECT * FROM htmtest;
