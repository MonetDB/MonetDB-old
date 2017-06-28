# test various error conditions

# negative weights
START TRANSACTION;

CREATE TABLE vals(w DOUBLE, value INTEGER);
INSERT INTO vals VALUES (1, 100), (-1, 50);

SELECT * FROM vals SAMPLE 1 USING WEIGHTS w;


ROLLBACK;

# string weights
START TRANSACTION;

CREATE TABLE vals(w STRING, value INTEGER);
INSERT INTO vals VALUES ('#1 sample', 100), ('#2 sample', 50);

SELECT * FROM vals SAMPLE 1 USING WEIGHTS w;

ROLLBACK;
