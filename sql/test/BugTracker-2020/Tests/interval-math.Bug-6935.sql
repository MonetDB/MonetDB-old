SELECT INTERVAL '1' HOUR / 2, INTERVAL '1' HOUR * 1000 / 2000; --all output 1800.000
SELECT INTERVAL '1' HOUR / 2.0; -- 1800.000
SELECT INTERVAL '1' HOUR * 1000.0 / 2000.0; --1800.000
SELECT INTERVAL '1' HOUR * 1000 / 1800000; -- 2.000
SELECT INTERVAL '1' HOUR * CAST(1000 AS DOUBLE); --cannot multiply interval with floating-point
SELECT INTERVAL '1' MONTH * 1.2; -- 1
SELECT INTERVAL '1' MONTH / 2.0; -- 0
SELECT INTERVAL '1' MONTH / 1.0; -- 1
SELECT INTERVAL '1' SECOND * 1.2; --1.200
SELECT INTERVAL '1' HOUR / INTERVAL '1800' SECOND; --error on typing branch, cannot divide intervals
select mya + interval '2' second from (select interval '3' second * 1.2) as mya(mya); -- 5.600
