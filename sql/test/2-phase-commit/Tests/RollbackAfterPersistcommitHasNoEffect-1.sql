-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (31, 99), (32, 99), (33, 99);
CALL precommit(8);
CALL persistcommit(8);

ROLLBACK;

SELECT * FROM htmtest;
