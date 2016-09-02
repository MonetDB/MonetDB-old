SET SCHEMA ais;
SET optimizer = 'iot_pipe';

-- Vessels positions reports table based on AIS messages types 1, 2 and 3
CREATE STREAM TABLE vessels11 (implicit_timestamp timestamp, mmsi int, lat real, lon real, nav_status smallint, sog real, rotais smallint);

-- Position reports are sent every 3-5 seconds so is resonable to consume the tuples arrived on the last 8 seconds
-- Inserts for iot web server (providing time based flush of 8 seconds)
INSERT INTO iot.webserverstreams SELECT tabl.id, 2 , 8, 's' FROM sys.tables tabl INNER JOIN sys.schemas sch ON tabl.schema_id = sch.id WHERE tabl.name = 'vessels11' AND sch.name = 'ais';

--Q11 Calculate average speed observations per ship -- Stream only

CREATE TABLE ais11r (calc_time timestamp, mmsi int, speed_sum real, speed_count int);

CREATE VIEW ais11v AS SELECT calc_time, mmsi, speed_sum / speed_count AS average_speed FROM ais11r;

CREATE PROCEDURE ais11q()
BEGIN
	UPDATE ais11r
		SET calc_time = current_timestamp,
			speed_sum = speed_sum + (SELECT COALESCE(SUM(sog), 0) FROM vessels11 INNER JOIN ais11r ON vessels11.mmsi = ais11r.mmsi),
			speed_count = speed_count + (SELECT COUNT(*) FROM vessels11 INNER JOIN ais11r ON vessels11.mmsi = ais11r.mmsi)
		FROM vessels11 WHERE ais11r.mmsi = vessels11.mmsi; /* Don't forget the join! */
	
	--DELETE FROM ais11r
		--WHERE mmsi NOT IN (SELECT mmsi FROM vessels11);

	INSERT INTO ais11r
		WITH data_time AS (SELECT current_timestamp AS cur_time),
		new_inserts AS (SELECT mmsi, SUM(sog) AS sum_sog, COUNT(*) AS count_tuples FROM vessels11 WHERE mmsi NOT IN (SELECT mmsi FROM ais11r) GROUP BY mmsi)
		SELECT cur_time, mmsi, sum_sog, count_tuples FROM new_inserts CROSS JOIN data_time;
END;

CALL iot.query('ais', 'ais11q');
CALL iot.pause();
-- We don't set the tumbling, so no tuple will be reused in the following window
CALL iot.heartbeat('ais', 'vessels11', 8000);
CALL iot.resume();

CALL iot.pause();
DELETE FROM iot.webserverstreams;
DROP PROCEDURE ais11q;
DROP TABLE vessels11;
DROP VIEW ais11v;
DROP TABLE ais11r;

CALL iot.stop();
SET SCHEMA sys;
DROP SCHEMA ais;

