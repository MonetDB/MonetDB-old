CREATE SCHEMA iot;
SET SCHEMA iot;
SET OPTIMIZER = 'iot_pipe';

CREATE STREAM TABLE temperature (t TIMESTAMP, sensor INT, val DECIMAL);
CALL window('iot', 'temperature', 3);
CREATE TABLE results (minimum TIMESTAMP, tuples INT, average DECIMAL);

CREATE PROCEDURE testing()
BEGIN
	INSERT INTO results SELECT MIN(t), COUNT(*), AVG(val) FROM temperature;
END;

CALL query('iot', 'testing');

INSERT INTO temperature VALUES (now(), 1, 1), (now(), 2, 2), (now(), 3, 3), (now(), 4, 4);

SELECT * FROM results; /* should have only 1 row, where "tuples" column value is 3 */

CALL getwindow('iot','temperature'); /* should not hang! */

