-- summarize the table at regular minute intervals

CREATE TABLE rooms_min(tick timestamp, room string, level integer, temp integer);

CREATE PROCEDURE summarize()
BEGIN
   INSERT INTO rooms_min 
      WITH T(tick, period, room, level, temp)
	   AS ( SELECT min(tick), epoch(tick)/60 AS period, room, level, avg(temp) 
	   FROM rooms
	   GROUP BY period, room, level)
	  SELECT tick,room,level,temp FROM T;
END;

CALL cquery.register('sys','summarize');
CALL cquery.heartbeat('sys','summarize',60);
CALL cquery.resume();
CALL cquery.wait(100);
CALL cquery.pause();

--SELECT * FROM cquery.status();
CALL cquery.deregister('sys','summarize');

SELECT * FROM rooms_min;

DROP PROCEDURE summarize;
DROP TABLE rooms_min;
