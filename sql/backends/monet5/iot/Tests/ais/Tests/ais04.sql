SET SCHEMA ais;
SET optimizer = 'iot_pipe';

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels4 (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status tinyint, sog real, rotais smallint);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'vessels4' AND sch.name = 'ais';

--Q4 Ship turning degree > 180 -- Stream only

CREATE TABLE ais04r (calc_time timestamp, mmsi int);

CREATE PROCEDURE ais04q()
BEGIN
	INSERT INTO ais04r
		WITH data_time AS (SELECT current_timestamp AS cur_time)
		SELECT cur_time, mmsi FROM vessels4 CROSS JOIN data_time WHERE sys.abs(rotais) > 180 AND (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels4 GROUP BY mmsi);
END;

CALL iot.query('ais', 'ais04q');
CALL iot.pause();
-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels4', 8000);
CALL iot.resume();

CALL iot.pause();
DELETE FROM iot.webserverstreams;
DROP PROCEDURE ais04q;
DROP TABLE vessels4;
DROP TABLE ais04r;

