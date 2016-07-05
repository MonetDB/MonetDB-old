CREATE SCHEMA ais;
SET SCHEMA ais;
SET optimizer = 'iot_pipe';

/* calculate distance in kms between two coordinates http://www.movable-type.co.uk/scripts/latlong.html
   so we don't need to create a geometry element when working only with stream data (the iot web server doesn't support geom types yet) */
CREATE FUNCTION km_distance(lat1 FLOAT, lon1 FLOAT, lat2 FLOAT, lon2 FLOAT) RETURNS FLOAT
BEGIN
	DECLARE deg_to_rad FLOAT, deg_to_rad_div FLOAT, aux FLOAT;
	SET deg_to_rad = pi() / 180;
	SET deg_to_rad_div = deg_to_rad / 2;
	SET aux = sys.power(sys.sin((lat2 - lat1) * deg_to_rad_div), 2) + sys.cos(lat1 * deg_to_rad) * sys.cos(lat2 * deg_to_rad) * sys.power(sys.sin((lon2 - lon1) * deg_to_rad_div), 2);
	RETURN 12742 * sys.atan(sys.sqrt(aux), sys.sqrt(1 - aux));
END;

-- returns a geometry point from latitude and longitude https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
CREATE FUNCTION geographical_to_cartesian(lat FLOAT, lon FLOAT) RETURNS Geometry
BEGIN
	DECLARE deg_to_rad FLOAT, lat_rad FLOAT, lon_rad FLOAT, aux1 FLOAT, aux2 FLOAT;
	SET deg_to_rad = pi() / 180;
	SET lat_rad = lat * deg_to_rad;
	SET lon_rad = lon * deg_to_rad;
	SET aux1 = sys.cos(lat_rad);
	SET aux2 = 6371 * aux1;
	RETURN sys.st_makepoint(aux2 * sys.cos(lon_rad), aux2 * sys.sin(lon_rad), 6371 * sys.sin(lat_rad));
END;

CREATE TABLE static_locations (loc Geometry); /* TODO Populate this table */ 

CREATE STREAM TABLE vessels (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status tinyint, sog real, rotais smallint);
CREATE STREAM TABLE stations (implicit_timestamp timestamp, mmsi int, lat real, lon real, pos_dev tinyint);

CREATE STREAM TABLE vessels1 (LIKE vessels);
CREATE STREAM TABLE vessels2 (LIKE vessels);
CREATE STREAM TABLE vessels3 (LIKE vessels);
CREATE STREAM TABLE vessels4 (LIKE vessels);
CREATE STREAM TABLE vessels5 (LIKE vessels);
CREATE STREAM TABLE vessels6 (LIKE vessels);
CREATE STREAM TABLE vessels7 (LIKE vessels);
CREATE STREAM TABLE vessels8 (LIKE vessels);
CREATE STREAM TABLE vessels9 (LIKE vessels);
CREATE STREAM TABLE vessels10 (LIKE vessels);
CREATE STREAM TABLE vessels11 (LIKE vessels);

CALL iot.heartbeat('ais','vessels',8000); /*Position reports are sent every 3-5 seconds so we can run the query for the tuples arrived in the last */

CREATE PROCEDURE ais00q() /*Provide data for each query*/
BEGIN
	INSERT INTO vessels1 SELECT * FROM vessels;
	INSERT INTO vessels2 SELECT * FROM vessels;
	INSERT INTO vessels3 SELECT * FROM vessels;
	INSERT INTO vessels4 SELECT * FROM vessels;
	INSERT INTO vessels5 SELECT * FROM vessels;
	INSERT INTO vessels6 SELECT * FROM vessels;
	INSERT INTO vessels7 SELECT * FROM vessels;
	INSERT INTO vessels8 SELECT * FROM vessels;
	INSERT INTO vessels9 SELECT * FROM vessels;
	INSERT INTO vessels10 SELECT * FROM vessels;
	INSERT INTO vessels11 SELECT * FROM vessels;
END;

--Q1 Calculate speed of ships per hour (in knots) -- Stream only

CREATE TABLE ais01r (calc_time timestamp, mmsi int, sog real);

CREATE PROCEDURE ais01q()
BEGIN
	INSERT INTO ais01r
		SELECT current_timestamp, mmsi, sog FROM vessels1 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels1 GROUP BY mmsi);
END;

--Q2 Number of ship per hour -- Stream only

CREATE TABLE ais02r (calc_time timestamp, number_ships int);

CREATE PROCEDURE ais02q()
BEGIN
	INSERT INTO ais02r
		SELECT current_timestamp, count(DISTINCT mmsi) FROM vessels2;
END;

--Q3 Currently anchorred ship -- Stream only

CREATE TABLE ais03r (calc_time timestamp, mmsi int);

CREATE PROCEDURE ais03q()
BEGIN
	INSERT INTO ais03r
		SELECT current_timestamp, mmsi FROM vessels3 WHERE nav_status = 1 AND (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels3 GROUP BY mmsi);
END;

--Q4 Ship turning degree > 180 -- Stream only

CREATE TABLE ais04r (calc_time timestamp, mmsi int);

CREATE PROCEDURE ais04q()
BEGIN
	INSERT INTO ais04r
		SELECT current_timestamp, mmsi FROM vessels4 WHERE sys.abs(rotais) > 180 AND (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels4 GROUP BY mmsi);
END;

--Q5 Closest ship to each other -- Stream only

CREATE TABLE ais05r (calc_time timestamp, mmsi1 int, mmsi2 int, distance float);

CREATE PROCEDURE ais05q()
BEGIN
	INSERT INTO ais05r 
		WITH data AS (SELECT mmsi, lat, lon FROM vessels5 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels5 GROUP BY mmsi)),
		distances AS (SELECT d1.mmsi AS mmsi1, d2.mmsi AS mmsi2, km_distance(d1.lat, d1.lon, d2.lat, d2.lon) AS distance FROM data d1 CROSS JOIN data d2 WHERE NOT d1.mmsi = d2.mmsi)
		SELECT current_timestamp, mmsi1, mmsi2, distance FROM distances WHERE (mmsi1, distance) IN (SELECT mmsi1, min(distance) FROM distances GROUP BY mmsi1);
END;

--Q6 For each station calulate ship within a radios of 3 km -- Stream join

CREATE TABLE ais06r (calc_time timestamp, smmsi int, vmmsi int, distance float);

CREATE PROCEDURE ais06q()
BEGIN
	INSERT INTO ais06r
		WITH data1 AS (SELECT mmsi, lat, lon FROM vessels6 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels6 GROUP BY mmsi)), 
		data2 AS (SELECT mmsi, lat, lon FROM stations WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM stations GROUP BY mmsi)), 
		calculations AS (SELECT d1.mmsi AS smmsi, d2.mmsi AS vmmsi, km_distance(d1.lat, d1.lon, d2.lat, d2.lon) AS distance FROM data1 d1 CROSS JOIN data2 d2)
		SELECT current_timestamp, smmsi, vmmsi, distance FROM calculations WHERE distance < 3;
END;

--Q7 Which ship are currently anchored at the harbors -- Stream + static

CREATE TABLE ais07r (calc_time timestamp, location Geometry, mmsi int);

CREATE PROCEDURE ais07q()
BEGIN
	INSERT INTO ais07r
		WITH data AS (SELECT mmsi, geographical_to_cartesian(lat, lon) AS calc_point FROM vessels7 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels7 WHERE nav_status = 1 GROUP BY mmsi))
		SELECT current_timestamp, loc, mmsi FROM data CROSS JOIN static_locations WHERE Contains(loc, calc_point);
END;

--Q8 Track the movements of a ship S -- Stream only

CREATE TABLE ais08r (calc_time timestamp, mmsi int, implicit_timestamp timestamp, latitude float, longitude float);

CREATE PROCEDURE ais08q()
BEGIN
	INSERT INTO ais08r
		SELECT current_timestamp, mmsi, implicit_timestamp, lat, lon FROM vessels8;
END;

--Q9 Notify when a ship S arrived at an harbor -- Stream + static

CREATE TABLE ais09r (calc_time timestamp, location Geometry, mmsi int, implicit_timestamp timestamp);

CREATE PROCEDURE ais09q()
BEGIN
	INSERT INTO ais09r
		WITH data AS (SELECT mmsi, implicit_timestamp, geographical_to_cartesian(lat, lon) AS calc_point FROM vessels9),
		SELECT current_timestamp, loc, mmsi, min(implicit_timestamp) FROM data CROSS JOIN static_locations WHERE (loc, mmsi) NOT IN (SELECT location, mmsi FROM ais09r) AND Contains(loc, calc_point) GROUP BY loc, mmsi;
END;

--Q10 Estimated time of arrival of ship S at harbor H -- Stream join + static

CREATE TABLE ais10r (calc_time timestamp, location Geometry, mmsi int, time_left float); /* in hours */

CREATE PROCEDURE ais10q()
BEGIN
	INSERT INTO ais10r
		WITH data AS (SELECT loc, mmsi, sog, Distance(loc, geographical_to_cartesian(lat, lon)) AS distance FROM vessels10 CROSS JOIN static_locations WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels10 GROUP BY mmsi)),
		SELECT current_timestamp, loc, mmsi, distance / sog * 1.852 FROM data WHERE distance > 0;
END;

--Q11 Calculate average speed per ship -- Stream only

CREATE TABLE ais11r (calc_time timestamp, mmsi int, speed_sum float, speed_count int);

CREATE VIEW ais11v AS SELECT calc_time, mmsi, speed_sum / speed_count AS average_speed FROM ais11r;

CREATE PROCEDURE ais11q()
BEGIN
	UPDATE ais11r
		SET calc_time = current_timestamp,
			speed_sum = speed_sum + (SELECT sum(sog) FROM vessels11 WHERE vessels11.mmsi = ais11r.mmsi),
			speed_count = speed_count + (SELECT count(*) FROM vessels11 WHERE vessels11.mmsi = ais11r.mmsi)
		FROM ais11r INNER JOIN vessels11 ON results.mmsi = vessels11.mmsi;
	DELETE FROM ais11r
		WHERE mmsi NOT IN (SELECT mmsi FROM vessels11);
	INSERT INTO ais11r
		SELECT current_timestamp, mmsi, sum(sog), count(*) FROM vessels11 GROUP BY mmsi HAVING mmsi NOT IN (SELECT mmsi FROM ais11r);
END;

CALL iot.query('ais','ais00q');
CALL iot.query('ais','ais01q');
CALL iot.query('ais','ais02q');
CALL iot.query('ais','ais03q');
CALL iot.query('ais','ais04q');
CALL iot.query('ais','ais05q');
CALL iot.query('ais','ais06q');
CALL iot.query('ais','ais07q');
CALL iot.query('ais','ais08q');
CALL iot.query('ais','ais09q');
CALL iot.query('ais','ais10q');
CALL iot.query('ais','ais11q');

