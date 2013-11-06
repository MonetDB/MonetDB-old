-- Optimized schema for mSEED (meta-)data.
CREATE SCHEMA mseed;

CREATE TABLE "mseed"."files" (
--	"file_id"  INTEGER,
	"file_location" STRING,
-- 	"file_last_modified"	TIMESTAMP,
	"dataquality"	CHAR(1),
	"network"	VARCHAR(10),
	"station"	VARCHAR(10),
	"location"	VARCHAR(10),
	"channel"	VARCHAR(10),
	"encoding"	TINYINT,
	"byte_order"	BOOLEAN,
--	CONSTRAINT "files_pkey_file_id" PRIMARY KEY (file_id)
	CONSTRAINT "files_pkey_file_loc" PRIMARY KEY (file_location)
);


CREATE TABLE "mseed"."segments" (
--	"file_id" INTEGER,	
	"file_location" STRING,
	"segment_id"	INTEGER,
	"start_time"	TIMESTAMP,
	"end_time"	TIMESTAMP,
	"prev_gap"	DOUBLE,
	"frequency"	DOUBLE,
	"sample_count"   BIGINT,
	"sample_type"   CHAR(1),
	CONSTRAINT "segments_file_loc_segment_id_pkey" PRIMARY KEY (file_location, segment_id),
	CONSTRAINT "segments_fkey_files_file_loc" FOREIGN KEY (file_location) REFERENCES mseed.files(file_location)
);


CREATE TABLE "mseed"."data" (
--	"file_id"      INTEGER,
	"file_location" STRING,
	"segment_id"       INTEGER,
	"sample_time"  TIMESTAMP,
	"sample_value" INTEGER
--	CONSTRAINT "data_file_id_segment_id_sample_time_pkey" PRIMARY KEY (file_id, segment_id, sample_time),
--	CONSTRAINT "data_fkey_segments_file_id_segment_id" FOREIGN KEY (file_id, segment_id) REFERENCES mseed.segments(file_id, segment_id)
-- 	CONSTRAINT "data_file_loc_segment_id_sample_time_pkey" PRIMARY KEY (file_location, segment_id, sample_time),
-- 	CONSTRAINT "data_fkey_segments_file_loc_segment_id" FOREIGN KEY (file_location, segment_id) REFERENCES mseed.segments(file_location, segment_id)
);

CREATE VIEW mseed.metadataview AS
SELECT f.file_location, dataquality, network, station, location, channel, encoding, byte_order, segment_id, start_time, end_time, prev_gap, frequency, sample_count, sample_type
FROM mseed.files AS f
	JOIN mseed.segments AS s
		ON f.file_location = s.file_location;

CREATE VIEW mseed.dataview AS
SELECT f.file_location, dataquality, network, station, location, channel, encoding, byte_order, s.segment_id, start_time, end_time, prev_gap, frequency, sample_count, sample_type, sample_time, sample_value
FROM mseed.files AS f 
	JOIN mseed.segments AS s
		ON f.file_location = s.file_location
	JOIN mseed.data AS d 
		ON s.file_location = d.file_location AND s.segment_id = d.segment_id;

-- To dump information of the cached files
CREATE FUNCTION dumpcache()
RETURNS STRING EXTERNAL NAME recycle.dumpcache;

