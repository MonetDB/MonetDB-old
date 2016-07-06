CREATE SCHEMA ais;
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

-- Returns a geometry point from latitude and longitude: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
CREATE FUNCTION geographic_to_cartesian(lat FLOAT, lon FLOAT) RETURNS POINT
BEGIN
	DECLARE deg_to_rad FLOAT, lat_rad FLOAT, lon_rad FLOAT, aux1 FLOAT, aux2 FLOAT;
	SET deg_to_rad = pi() / 180;
	SET lat_rad = lat * deg_to_rad;
	SET lon_rad = lon * deg_to_rad;
	SET aux1 = sys.cos(lat_rad);
	SET aux2 = 6371 * aux1;
	RETURN sys.st_makepoint(aux2 * sys.cos(lon_rad), aux2 * sys.sin(lon_rad), 6371 * sys.sin(lat_rad));
END;

CREATE TABLE static_locations (harbor CHAR(32), field POLYGON); /* All major harbors of Amsterdam: https://en.wikipedia.org/wiki/Port_of_Amsterdam#Geography */

INSERT INTO static_locations VALUES ('Amerikahaven', 'POLYGON( (3871.44456893 322.23523795 5049.52694825, 3873.75021533 323.064093023 5047.70540553, 3873.71867646 323.885373981 5047.67697859, 3873.5017397 324.872292495 5047.78003345, 3872.7938342 325.19853123 5048.30217331, 3872.60164177 324.785093693 5048.47622229, 3872.6741224 323.809744716 5048.48327629, 3871.54981295 323.35946261 5049.37438737) )'); /* Amerikahaven */
INSERT INTO static_locations VALUES ('Petroleumhaven', 'POLYGON( (3871.97792572 328.849674441 5048.69149725, 3872.33795265 328.725455306 5048.42345247, 3872.43401269 329.066629268 5048.32754195, 3872.11428671 329.287777613 5048.55835959, 3872.00301454 329.044638923 5048.65955289) )'); /* Petroleumhaven */
INSERT INTO static_locations VALUES ('Oostelijkhaven', 'POLYGON( (3874.76945837 332.896017406 5046.28406712, 3874.99350047 332.95621223 5046.1080579, 3875.1279382 334.058396133 5045.93197046, 3875.5579763 334.604372405 5045.56550708, 3875.35055079 335.984299981 5045.63312763, 3874.41663614 335.616780364 5046.3747489, 3874.6428613 333.721315622 5046.32676121) )'); /* Oostelijkhaven */
INSERT INTO static_locations VALUES ('Afrikahaven', 'POLYGON( (3871.30476187 320.817833925 5049.72438438, 3872.88902009 321.220747092 5048.4838189, 3872.96603 321.966084418 5048.37726116, 3872.74497591 322.035368331 5048.54242065, 3871.38317977 321.685638604 5049.60905668) )'); /* Afrikahaven */
INSERT INTO static_locations VALUES ('DeRuijterkadehaven', 'POLYGON( (3874.84907684 331.832795545 5046.29295895, 3874.5621738 332.100597943 5046.49563105, 3874.74221319 332.592560967 5046.32499644, 3875.14500709 332.378589729 5046.02979055) )'); /* DeRuijterkadehaven */
INSERT INTO static_locations VALUES ('Coenhaven', 'POLYGON( (3872.18320026 329.539035698 5048.48910938, 3872.7184903 329.096658095 5048.10735667, 3872.88840793 329.5493504 5047.9474646, 3872.80986545 329.954639206 5047.98124819, 3872.54376258 330.007944404 5048.18190674, 3872.3152549 329.754769754 5048.37373404) )'); /* Coenhaven */
INSERT INTO static_locations VALUES ('Houthaven', 'POLYGON( (3872.87117926 330.512107085 5047.89773826, 3872.96624169 330.37417078 5047.83383225, 3873.54601208 330.634050719 5047.37193169, 3873.94528714 331.083171623 5047.03604561, 3873.97858621 331.533215981 5046.98094313, 3873.54647189 331.388091275 5047.32212793, 3872.86905716 330.50899826 5047.89956994) )'); /* Houthaven */
INSERT INTO static_locations VALUES ('JanvanRiebeeckhaven', 'POLYGON( (3871.94120918 327.917683906 5048.780275, 3872.13342591 327.779180854 5048.64185109, 3872.08217261 327.044586025 5048.72879914, 3872.14824523 327.029747516 5048.67908578, 3872.20328537 327.665296944 5048.59566315, 3872.60946857 327.378362182 5048.30271596, 3872.70184991 327.573104981 5048.21921499, 3872.17072336 327.995591149 5048.59919, 3872.36759179 328.102800999 5048.44122339, 3872.6245223 327.885014862 5048.25828641, 3872.67235818 328.003023524 5048.21392403, 3872.48169232 328.150453288 5048.35060417, 3872.89890584 328.498434046 5048.00790847, 3872.68943537 328.702668473 5048.15531585, 3872.54528354 328.410031669 5048.28494422, 3871.93965888 328.154140123 5048.76610058) )'); /* JanvanRiebeeckhaven */
INSERT INTO static_locations VALUES ('Westhaven', 'POLYGON( (3871.6739966 324.806577833 5049.18628603, 3872.14078933 325.202088988 5048.80285899, 3872.12532899 325.551292003 5048.7922113, 3871.97172049 325.836261828 5048.89163344, 3873.7575901 326.565103468 5047.47445425, 3873.75724972 325.799548089 5047.52418763, 3874.0235621 325.991429546 5047.30740377, 3874.1699978 326.506372453 5047.16171891, 3874.16875642 327.01475571 5047.12975822, 3873.91395529 327.291366901 5047.30740377, 3873.58537382 327.672691046 5047.53483983, 3872.81792241 327.54936104 5048.13170945, 3872.87160844 326.694979039 5048.14588688, 3871.7994125 326.178096178 5049.00169924) )'); /* Westhaven */
INSERT INTO static_locations VALUES ('Mercuriushaven', 'POLYGON( (3872.72450963 330.277569199 5048.02561402, 3873.1986563 329.657593618 5047.70235253, 3872.92006997 329.040734675 5047.95635149, 3873.11386675 328.884805095 5047.81782161, 3873.41352071 329.00959555 5047.57975505, 3873.61782996 329.465350531 5047.39323713, 3873.33890872 330.429316655 5047.54427072, 3872.9540041 330.381842104 5047.84271947) )'); /* Mercuriushaven */

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels0 (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status tinyint, sog real, rotais smallint);
-- Stations positions reports table based on AIS message type 4
CREATE STREAM TABLE stations (implicit_timestamp timestamp, mmsi int, lat real, lon real);

-- Need to duplicate vessels data to several tables for each continuous query
CREATE STREAM TABLE vessels1 (LIKE vessels0);
CREATE STREAM TABLE vessels2 (LIKE vessels0);
CREATE STREAM TABLE vessels3 (LIKE vessels0);
CREATE STREAM TABLE vessels4 (LIKE vessels0);
CREATE STREAM TABLE vessels5 (LIKE vessels0);
CREATE STREAM TABLE vessels6 (LIKE vessels0);
CREATE STREAM TABLE vessels7 (LIKE vessels0);
CREATE STREAM TABLE vessels8 (LIKE vessels0);
CREATE STREAM TABLE vessels9 (LIKE vessels0);
CREATE STREAM TABLE vessels10 (LIKE vessels0);
CREATE STREAM TABLE vessels11 (LIKE vessels0);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams
	SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name LIKE 'vessels%' AND sch.name = 'ais';

INSERT INTO iot.webserverstreams
	SELECT tabl.id, 2 , 10, 's' FROM sys.tables INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE name = 'stations' AND sch.name = 'ais';

-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels0', 8000); 
CALL iot.heartbeat('ais', 'vessels1', 8000);
CALL iot.heartbeat('ais', 'vessels2', 8000);
CALL iot.heartbeat('ais', 'vessels3', 8000);
CALL iot.heartbeat('ais', 'vessels4', 8000);
CALL iot.heartbeat('ais', 'vessels5', 8000);
CALL iot.heartbeat('ais', 'vessels6', 8000);
CALL iot.heartbeat('ais', 'vessels7', 8000);
CALL iot.heartbeat('ais', 'vessels8', 8000);
CALL iot.heartbeat('ais', 'vessels9', 8000);
CALL iot.heartbeat('ais', 'vessels10', 8000);
CALL iot.heartbeat('ais', 'vessels11', 8000);

-- The stations send less often, therefore the heartbeat is 10 seconds
CALL iot.heartbeat('ais', 'stations', 10000);

CREATE PROCEDURE ais00q() /* Provide data for each query */
BEGIN
	INSERT INTO vessels1 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels2 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels3 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels4 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels5 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels6 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels7 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels8 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels9 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels10 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
	INSERT INTO vessels11 SELECT implicit_timestamp, mmsi, lat, lon, nav_status, sog, rotais FROM vessels0;
END;

--Q1 Calculate speed of ships per hour (in knots) -- Stream only

CREATE STREAM TABLE ais01r (calc_time timestamp, mmsi int, sog real);

CREATE PROCEDURE ais01q()
BEGIN
	INSERT INTO ais01r
		SELECT current_timestamp, mmsi, sog FROM vessels1 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels1 GROUP BY mmsi);
END;

--Q2 Number of ship per hour -- Stream only

CREATE STREAM TABLE ais02r (calc_time timestamp, number_ships int);

CREATE PROCEDURE ais02q()
BEGIN
	INSERT INTO ais02r
		SELECT current_timestamp, count(DISTINCT mmsi) FROM vessels2;
END;

--Q3 Currently anchorred ship -- Stream only

CREATE STREAM TABLE ais03r (calc_time timestamp, mmsi int);

CREATE PROCEDURE ais03q()
BEGIN
	INSERT INTO ais03r
		SELECT current_timestamp, mmsi FROM vessels3 WHERE nav_status = 1 AND (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels3 GROUP BY mmsi);
END;

--Q4 Ship turning degree > 180 -- Stream only

CREATE STREAM TABLE ais04r (calc_time timestamp, mmsi int);

CREATE PROCEDURE ais04q()
BEGIN
	INSERT INTO ais04r
		SELECT current_timestamp, mmsi FROM vessels4 WHERE sys.abs(rotais) > 180 AND (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels4 GROUP BY mmsi);
END;

--Q5 Closest ship to each other -- Stream only

CREATE STREAM TABLE ais05r (calc_time timestamp, mmsi1 int, mmsi2 int, distance float);

CREATE PROCEDURE ais05q()
BEGIN
	INSERT INTO ais05r 
		WITH data AS (SELECT mmsi, lat, lon FROM vessels5 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels5 GROUP BY mmsi)),
		distances AS (SELECT d1.mmsi AS mmsi1, d2.mmsi AS mmsi2, km_distance(d1.lat, d1.lon, d2.lat, d2.lon) AS distance FROM data d1 CROSS JOIN data d2 WHERE NOT d1.mmsi = d2.mmsi)
		SELECT current_timestamp, mmsi1, mmsi2, distance FROM distances WHERE (mmsi1, distance) IN (SELECT mmsi1, min(distance) FROM distances GROUP BY mmsi1);
END;

--Q6 For each station calulate ship within a radios of 3 km -- Stream join

CREATE STREAM TABLE ais06r (calc_time timestamp, smmsi int, vmmsi int, distance float);

CREATE PROCEDURE ais06q()
BEGIN
	INSERT INTO ais06r
		WITH data1 AS (SELECT mmsi, lat, lon FROM vessels6 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels6 GROUP BY mmsi)), 
		data2 AS (SELECT mmsi, lat, lon FROM stations WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM stations GROUP BY mmsi)), 
		calculations AS (SELECT d1.mmsi AS smmsi, d2.mmsi AS vmmsi, km_distance(d1.lat, d1.lon, d2.lat, d2.lon) AS distance FROM data1 d1 CROSS JOIN data2 d2)
		SELECT current_timestamp, smmsi, vmmsi, distance FROM calculations WHERE distance < 3;
END;

--Q7 Which ship are currently anchored at the harbors -- Stream + static

CREATE STREAM TABLE ais07r (calc_time timestamp, harbor char(32), mmsi int);

CREATE PROCEDURE ais07q()
BEGIN
	INSERT INTO ais07r
		WITH data AS (SELECT mmsi, geographic_to_cartesian(lat, lon) AS calc_point FROM vessels7 WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels7 WHERE nav_status = 1 GROUP BY mmsi))
		SELECT current_timestamp, harbor, mmsi FROM data CROSS JOIN static_locations WHERE sys.st_contains(field, calc_point);
END;

--Q8 Track the movements of a ship S -- Stream only

CREATE STREAM TABLE ais08r (calc_time timestamp, mmsi int, implicit_timestamp timestamp, latitude float, longitude float);

CREATE PROCEDURE ais08q()
BEGIN
	INSERT INTO ais08r
		SELECT current_timestamp, mmsi, implicit_timestamp, lat, lon FROM vessels8;
END;

--Q9 Notify when a ship S arrived at an harbor -- Stream + static

CREATE STREAM TABLE ais09r (calc_time timestamp, harbor char(32), mmsi int, implicit_timestamp timestamp);

CREATE PROCEDURE ais09q()
BEGIN
	INSERT INTO ais09r
		WITH data AS (SELECT mmsi, implicit_timestamp, geographic_to_cartesian(lat, lon) AS calc_point FROM vessels9)
		SELECT current_timestamp, harbor, mmsi, min(implicit_timestamp) FROM data CROSS JOIN static_locations WHERE sys.st_contains(field, calc_point) AND (harbor, mmsi) NOT IN (SELECT harbor, mmsi FROM ais09r) GROUP BY harbor, mmsi;
END;

--Q10 Estimated time of arrival of ship S at harbor H -- Stream join + static

CREATE STREAM TABLE ais10r (calc_time timestamp, harbor char(32), mmsi int, time_left float); /* in hours */

CREATE PROCEDURE ais10q()
BEGIN
	INSERT INTO ais10r
		WITH data AS (SELECT harbor, mmsi, sog, Distance(field, geographic_to_cartesian(lat, lon)) AS distance FROM vessels10 CROSS JOIN static_locations WHERE (implicit_timestamp, mmsi) IN (SELECT max(implicit_timestamp), mmsi FROM vessels10 GROUP BY mmsi))
		SELECT current_timestamp, harbor, mmsi, distance / sog * 1.852 FROM data WHERE distance > 0;
END;

--Q11 Calculate average speed per ship -- Stream only

CREATE STREAM TABLE ais11r (calc_time timestamp, mmsi int, speed_sum float, speed_count int);

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

CALL iot.query('ais', 'ais00q');
CALL iot.query('ais', 'ais01q');
CALL iot.query('ais', 'ais02q');
CALL iot.query('ais', 'ais03q');
CALL iot.query('ais', 'ais04q');
CALL iot.query('ais', 'ais05q');
CALL iot.query('ais', 'ais06q');
CALL iot.query('ais', 'ais07q');
CALL iot.query('ais', 'ais08q');
CALL iot.query('ais', 'ais09q');
CALL iot.query('ais', 'ais10q');
CALL iot.query('ais', 'ais11q');

