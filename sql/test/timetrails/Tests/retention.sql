-- Remove some of the tuples

CREATE TABLE rooms_min(tick timestamp, room string, level integer, temp integer);

CREATE PROCEDURE retention()
BEGIN
   DELETE FROM rooms 
	WHERE now() - 60 > tick;
END;

CALL cquery.register('sys','retention');
CALL cquery.heartbeat('sys','retention',60);
CALL cquery.resume();
CALL cquery.wait(500);
CALL cquery.pause();

--SELECT * FROM cquery.status();
CALL cquery.deregister('sys','retention');

SELECT * FROM rooms_min;

DROP PROCEDURE retention;
DROP TABLE rooms_min;
