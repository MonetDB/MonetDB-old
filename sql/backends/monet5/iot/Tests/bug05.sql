SET optimizer = 'iot_pipe';

CREATE STREAM TABLE testing (a timestamp, b int, c real);
CALL iot.window('sys', 'testing', 3);
CREATE TABLE testout (a timestamp, b int, c real);

CREATE PROCEDURE cquery()
BEGIN	
	INSERT INTO testout SELECT * FROM testing;
END;

CALL iot.query('sys', 'cquery');
CALL iot.resume();

INSERT INTO testing VALUES (now(), 1, 1);
INSERT INTO testing VALUES (now(), 2, 2);
INSERT INTO testing VALUES (now(), 3, 3);

CALL iot.show('sys', 'cquery');
CALL iot.stop();

DROP PROCEDURE cquery;
DROP TABLE testout;
DROP TABLE testing;

