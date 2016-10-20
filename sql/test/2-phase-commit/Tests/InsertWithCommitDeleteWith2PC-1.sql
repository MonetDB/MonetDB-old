-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.

INSERT INTO htmtest VALUES (19, 99), (20, 99), (21, 99);
SELECT * FROM htmtest;

START TRANSACTION;
DELETE FROM htmtest WHERE id > 3;
CALL precommit(7);
CALL persistcommit(7);

SELECT * FROM htmtest;
