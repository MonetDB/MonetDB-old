SET SCHEMA ais;
SET optimizer = 'iot_pipe';

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels2 (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status smallint, sog real, rotais smallint);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'vessels2' AND sch.name = 'ais';

--Q2 Number of distinct ship in the last 8 seconds -- Stream only

CREATE TABLE ais02r (calc_time timestamp, number_ships int);

CREATE PROCEDURE ais02q()
BEGIN
	INSERT INTO ais02r
		SELECT current_timestamp, count(DISTINCT mmsi) FROM vessels2;
END;

CALL iot.query('ais', 'ais02q');
CALL iot.pause();
-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels2', 8000);
CALL iot.resume();

CALL iot.pause();
DELETE FROM iot.webserverstreams;
DROP PROCEDURE ais02q;
DROP TABLE vessels2;
DROP TABLE ais02r;

