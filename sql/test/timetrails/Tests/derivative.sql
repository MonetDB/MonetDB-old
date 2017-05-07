--derivative by time
CREATE FUNCTION rooms_derivative( stride bigint)
RETURNS TABLE(
    tick timestamp,
    room string,
    level integer,
    temp double)
BEGIN
   RETURN
       WITH bounds(first, last, period)
        AS (SELECT min(tick) AS first, max(tick) as last, epoch(tick)/stride AS period FROM rooms GROUP BY period)
        SELECT r2.tick, r2.room, r2.level, (r2.temp - r1.temp)/ (epoch(bounds.last) - epoch(bounds.first)) FROM bounds, rooms r1, rooms r2
       WHERE r1.tick = bounds.first and r2.tick = bounds.last and r1.room = r2.room and r1.level = r2.level;
END;

--TODO simplify using CREATE AGGREGATE derivative(tick,temp) RETURNS TABLE ( tick timestamp, result double) EXTERNAL NAME xyz.yy
--SELECT min(tick), derivative(tick, temp) FROM rooms WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' GROUP BY room, level;

