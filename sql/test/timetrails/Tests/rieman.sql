--TODO integral, calculate the Rieman integral. If the events are regularly spread it is equal to the sum
-- introduce the following CREATE AGGREGATE  integral(col1,col2) RETURNS double EXTERNAL NAME  xyz.yy;
SELECT min(tick), integral(tick, temp) 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
GROUP BY room, level;
