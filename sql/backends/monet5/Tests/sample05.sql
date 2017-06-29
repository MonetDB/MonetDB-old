
START TRANSACTION;

# random stuff in weights

CREATE FUNCTION complex_function(i DOUBLE) RETURNS DOUBLE LANGUAGE PYTHON {
	return (numpy.sin(numpy.sqrt(i+1)-2)+3)-4+5;
};

CREATE TABLE vals(w DOUBLE, value INTEGER);
INSERT INTO vals VALUES (1, 100), (0, 50);

SELECT * FROM vals SAMPLE 1 USING WEIGHTS CASE WHEN value > 55 THEN complex_function(w+3)+3 ELSE 0 END;
SELECT * FROM vals SAMPLE 1 USING WEIGHTS cast(cast(w AS INTEGER) AS DOUBLE);

ROLLBACK;
