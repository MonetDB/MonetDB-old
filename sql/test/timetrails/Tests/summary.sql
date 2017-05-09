-- summarize the table at regular minute intervals

CREATE TABLE rooms_min(tick timestamp, room string, level integer, temp integer);

CREATE PROCEDURE summarize(stride integer)
BEGIN
   INSERT INTO rooms_min 
      WITH T(tick, period, room, level, temp)
	   AS ( SELECT min(tick), epoch(tick)/stride AS period, room, level, avg(temp) 
	   FROM rooms
	   GROUP BY period, room, level)
	  SELECT tick,room,level,temp FROM T;
END;

CALL timetrails.register('sys','summarize');
CALL timetrails.heartbeat('sys','summarize',60);
CALL timetrails.resume();
CALL timetrails.wait(500);
CALL timetrails.pause();

SELECT * FROM timetrails.status();
CALL timetrails.deregister('sys','summarize');

SELECT * FROM rooms_min;

DROP PROCEDURE summarize;
DROP TABLE rooms_min;
