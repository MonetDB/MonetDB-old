--count
SELECT min(tick), room, level, count(*) 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
GROUP BY room, level;

-- distinct
SELECT min(tick),  count(DISTINCT temp )
FROM rooms WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59'
GROUP BY room, level;

--mean
SELECT min(tick), CAST(mean(cast(tick AS int)) AS timestamp)
FROM rooms 
WHERE tick 
BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59'
GROUP BY room, level;

--median
SELECT min(tick), CAST(median(cast(tick AS int)) AS timestamp)
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59'
GROUP BY room, level;

--avg
SELECT min(tick), avg(temp) 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
GROUP BY room, level;

--sum
SELECT min(tick), sum(temp) 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
GROUP BY room, level;

-- spread
SELECT min(tick), max(temp)-min(temp) AS spread
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
GROUP BY room, level;

--difference
SELECT R.tick, R.tick-S.tick FROM rooms R, rooms S WHERE R.tick = S.tick and R.room=S.room and R.level = S.level;

-- min/max
SELECT min(tick), min(temp) AS spread FROM rooms WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' GROUP BY room, level;


-- tag
SELECT tick, room, temp 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59'  AND room = 'L302';

--alias
SELECT tick, room, log(temp) AS mylabel 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59';

