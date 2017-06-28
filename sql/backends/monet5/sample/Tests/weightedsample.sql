set optimizer = 'sequential_pipe';
-- ADD FLAG TO DISALLOW PARALLELIZATION (MITOSIS) FOR weighted_sample
CREATE TABLE wsample (i INTEGER, weights DOUBLE);
INSERT INTO wsample VALUES (1, 1), (2, 1), (3, 1), (4, 1), (5, 1);


explain SELECT i FROM wsample WHERE weighted_sample(weights, 2);
SELECT i FROM wsample WHERE weighted_sample(weights, 2);

