--derivative by period
WITH T(rowid, tick, room, level, temp)
    AS (SELECT row_number() over() AS rowid, H.tick, H.room, H.level, H.temp
		FROM (SELECT max(tick) as tick, epoch(tick)/60 AS period, room, level, avg(temp) as temp
			FROM rooms
			WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59'
			GROUP BY period, room,level) AS H)
SELECT A.tick, A.room, A.level, (B.temp - A.temp) FROM T A, T B
WHERE B.rowid = A.rowid+1 and A.room = B.room and A.level = B.level;

