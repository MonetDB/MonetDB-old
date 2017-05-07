-- time rollup with a particular stride, which is the number seconds.

CREATE FUNCTION rooms_time(stride bigint )
RETURNS TABLE(
    tick timestamp,
    room string,
    level integer,
    temp double)
BEGIN
   RETURN
      SELECT bounds.tick, r.room, r.level, r.temp FROM
           (SELECT min(tick) AS tick, epoch(tick)/stride AS period FROM rooms GROUP BY period) AS bounds, rooms r
       WHERE epoch(r.tick)/stride = bounds.period;
END;

SELECT tick, room, level, avg(temp) FROM rooms_time(60) GROUP BY tick, room, level;
