-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

SELECT * FROM htmtest;
START TRANSACTION;
SELECT * FROM htmtest;
CALL precommit(2);
CALL persistcommit(2);
