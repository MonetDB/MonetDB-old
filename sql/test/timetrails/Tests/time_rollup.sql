--time rollup with a particular stride, which is the number seconds.

WITH T(period, tick, room, level, temp)
	AS (SELECT epoch(tick)/60, tick, room, level,temp FROM rooms 
	    WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59')
SELECT min(T.tick), T.room, T.level, avg( T.temp) FROM T GROUP BY T.period, T.room, T.level;

