--Extend s table with the period
CREATE FUNCTION room_period(stride integer)
RETURNS TABLE(period int,tick timestamp,room string,level int,temp float)
BEGIN
	RETURN SELECT epoch(tick)/60 AS period, min(tick), 
	room, level, avg( temp) FROM rooms GROUP BY period, room, level;
END;

SELECT * FROM room_period(60);
