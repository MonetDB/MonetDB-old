CREATE SCHEMA ais;
SET SCHEMA ais;
SET optimizer = 'iot_pipe';

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels1 (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status smallint, sog real, rotais smallint);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'vessels1' AND sch.name = 'ais';

--Q1 Calculate speed of ships (in knots) -- Stream only

CREATE STREAM TABLE ais01r (calc_time timestamp, mmsi int, sog real);

CREATE PROCEDURE ais01q()
BEGIN	
	INSERT INTO ais01r
		WITH data_time AS (SELECT current_timestamp AS cur_time)
		SELECT cur_time, mmsi, sog FROM vessels1 CROSS JOIN data_time WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels1 GROUP BY mmsi);
END;

CALL iot.query('ais', 'ais01q');
CALL iot.pause();
-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels1', 8000);
CALL iot.resume();

CALL iot.pause();
DELETE FROM iot.webserverstreams;
DROP PROCEDURE ais01q;
DROP TABLE vessels1;
DROP TABLE ais01r;

