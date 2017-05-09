-- Remove some of the tuples

CREATE TABLE rooms_min(tick timestamp, room string, level integer, temp integer);

CREATE PROCEDURE retention(stride integer)
BEGIN
   DELETE FROM rooms 
	WHERE now() - stride >tick;
END;

CALL timetrails.register('sys','retention');
CALL timetrails.heartbeat('sys','retention',60);
CALL timetrails.resume();
CALL timetrails.wait(500);
CALL timetrails.pause();

SELECT * FROM timetrails.status();
CALL timetrails.deregister('sys','retention');

SELECT * FROM rooms_min;

DROP PROCEDURE retention;
DROP TABLE rooms_min;
