
START TRANSACTION;


CREATE TABLE vals(joinkey INTEGER, id DOUBLE, type INTEGER);
INSERT INTO vals VALUES (1, 1, 100), (0, 0, 50);

# alias tables over weighted sample
SELECT v1.id FROM vals AS v1 SAMPLE 1 USING WEIGHTS 1-v1.id;

SELECT * FROM vals AS v1 INNER JOIN vals AS v2 ON v1.id=1-v2.id SAMPLE 1 USING WEIGHTS v1.id;


ROLLBACK;
