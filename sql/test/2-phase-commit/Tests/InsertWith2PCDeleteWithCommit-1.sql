-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

START TRANSACTION;
INSERT INTO htmtest VALUES (16, 99), (17, 99), (18, 99);
SELECT * FROM htmtest;
CALL precommit(6);
CALL persistcommit(6);
SELECT * FROM htmtest;

DELETE FROM htmtest WHERE id > 3;
SELECT * FROM htmtest;
