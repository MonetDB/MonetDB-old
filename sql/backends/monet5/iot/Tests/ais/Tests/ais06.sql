SET SCHEMA ais;
SET optimizer = 'iot_pipe';

-- Calculate distance in kms between two coordinates: http://www.movable-type.co.uk/scripts/latlong.html
-- Therefore we don't need to create a geometry element when working only with stream data (the iot web server doesn't support geom types yet)
CREATE FUNCTION km_distance(lat1 FLOAT, lon1 FLOAT, lat2 FLOAT, lon2 FLOAT) RETURNS FLOAT
BEGIN
	DECLARE deg_to_rad FLOAT, deg_to_rad_div FLOAT, aux FLOAT;
	SET deg_to_rad = pi() / 180;
	SET deg_to_rad_div = deg_to_rad / 2;
	SET aux = sys.power(sys.sin((lat2 - lat1) * deg_to_rad_div), 2) + sys.cos(lat1 * deg_to_rad) * sys.cos(lat2 * deg_to_rad) * sys.power(sys.sin((lon2 - lon1) * deg_to_rad_div), 2);
	RETURN 12742 * sys.atan(sys.sqrt(aux), sys.sqrt(1 - aux));
END;

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status tinyint, sog real, rotais smallint);
-- Stations positions reports table based on AIS message type 4
CREATE STREAM TABLE stations (implicit_timestamp timestamp, mmsi int, lat real, lon real);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams
	SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'vessels' AND sch.name = 'ais';

INSERT INTO iot.webserverstreams
	SELECT tabl.id, 2 , 10, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'stations' AND sch.name = 'ais';

--Q6 For each station calulate ship within a radios of 3 km -- Stream join

CREATE TABLE ais06r (calc_time timestamp, smmsi int, vmmsi int, distance float);

CREATE PROCEDURE ais06q()
BEGIN
	INSERT INTO ais06r
		WITH data1 AS (SELECT mmsi, lat, lon FROM vessels WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels GROUP BY mmsi)), 
		data2 AS (SELECT mmsi, lat, lon FROM stations WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM stations GROUP BY mmsi)), 
		calculations AS (SELECT d1.mmsi AS smmsi, d2.mmsi AS vmmsi, km_distance(d1.lat, d1.lon, d2.lat, d2.lon) AS distance FROM data1 d1 CROSS JOIN data2 d2),
		data_time AS (SELECT current_timestamp AS cur_time)
		SELECT cur_time, smmsi, vmmsi, distance FROM calculations CROSS JOIN data_time WHERE distance < 3;
END;

CALL iot.query('ais', 'ais06q');
CALL iot.pause();
-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels', 8000);
CALL iot.heartbeat('ais', 'stations', 10000);
CALL iot.resume();

