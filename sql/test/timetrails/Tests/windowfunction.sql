-- first/last in group

SELECT FIRST_VALUE OVER ( PARTITION BY room, level) FROM rooms;
SELECT LAST_VALUE OVER ( PARTITION BY room, level) FROM rooms;
