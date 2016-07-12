CREATE SCHEMA ais;
SET SCHEMA ais;
SET optimizer = 'iot_pipe';

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status tinyint, sog real, rotais smallint);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams
	SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'vessels' AND sch.name = 'ais';

--Q2 Number of ship per hour -- Stream only

CREATE TABLE ais02r (calc_time timestamp, number_ships int);

CREATE PROCEDURE ais02q()
BEGIN
	INSERT INTO ais02r
		SELECT current_timestamp, count(DISTINCT mmsi) FROM vessels;
END;

CALL iot.query('ais', 'ais02q');
CALL iot.pause();
-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels', 8000);
CALL iot.resume();

