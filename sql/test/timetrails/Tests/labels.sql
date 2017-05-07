-- Who should produce the proper label for the compound tag?

SELECT min(tick), concat(room, concat('_', cast(level as string))), count(*) 
FROM rooms 
WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
GROUP BY room, level;

-- TODO, introduce the following CREATE AGGREGATE  label(*) RETURNS string EXTERNAL NAME  xyz.yy; or aggregate function concat(*)
--SELECT min(tick), label(room,'_',level), count(*) 
--FROM rooms 
--WHERE tick BETWEEN timestamp '2017/01/01 09:00:00' AND timestamp '2017/01/31 23:59:59' 
--GROUP BY room, level;
