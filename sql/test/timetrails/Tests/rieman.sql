--TODO cumulative sum required in windows function : cumulative_sum(H.temp)
WITH T(tick, room, level, temp)
    AS (SELECT H.tick, H.room, H.level, H.temp
        FROM (SELECT max(tick) as tick, epoch(tick)/60 AS period, room, level, sum(temp) as temp
            FROM rooms
            WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59'
            GROUP BY period, room,level) AS H)
SELECT * FROM T ;
