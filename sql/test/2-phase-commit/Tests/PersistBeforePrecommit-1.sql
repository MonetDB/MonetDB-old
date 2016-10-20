-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (37, 99), (38, 99), (39, 99);
SELECT * FROM htmtest;
CALL persistcommit(9);
SELECT * FROM htmtest;
CALL precommit(9);
SELECT * FROM htmtest;
