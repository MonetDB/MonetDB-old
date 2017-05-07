CREATE TABLE rooms(
     tick timestamp,
     room string,
     level integer,
     temp double,
    PRIMARY KEY(tick, room,level) );
CREATE TABLE weather(
     tick timestamp,
     geoname string,
     degrees double,
     windspeed double,
    PRIMARY KEY(tick, geoname) );
CREATE TABLE birds(
     tick timestamp,
     birdname string,
     location string,
     speed integer,
    PRIMARY KEY(tick, birdname, location) );

INSERT INTO timetrails.metrics
VALUES('rooms','monetdb',0,'unknown','unknown',0,0,true,'unknown title','description');
INSERT INTO rooms VALUES
(timestamp '2017/01/01 09:00:00.000', 'L302', 3, 12.3),
(timestamp '2017/01/01 09:00:15.000', 'L302', 3, 13.3),
(timestamp '2017/01/01 09:00:30.000', 'L302', 3, 14.4),
(timestamp '2017/01/01 09:00:45.000', 'L302', 3, 15.5),
(timestamp '2017/01/01 10:00:00.000', 'L302', 3, 15.8),
(timestamp '2017/01/01 10:00:15.000', 'L302', 3, 18.4),
(timestamp '2017/01/01 10:00:30.000', 'L302', 3, 19.4),
(timestamp '2017/01/01 10:00:45.000', 'L302', 3, 21.5),
(timestamp '2017/01/01 11:00:00.000', 'L302', 3, 21.4),
(timestamp '2017/01/01 11:00:15.000', 'L302', 3, 21.4),
(timestamp '2017/01/01 11:00:30.000', 'L302', 3, 21.4),
(timestamp '2017/01/01 11:00:45.000', 'L302', 3, 21.5),
(timestamp '2017/01/01 12:00:00.000', 'L302', 3, 21.5),
(timestamp '2017/01/01 12:00:15.000', 'L302', 3, 21.5),
(timestamp '2017/01/01 12:00:30.000', 'L302', 3, 21.5),
(timestamp '2017/01/01 12:00:45.000', 'L302', 3, 21.5),
(timestamp '2017/01/01 13:00:00.000', 'L301', 3, 19.3),
(timestamp '2017/01/01 13:00:15.000', 'L301', 3, 18.3),
(timestamp '2017/01/01 13:00:30.000', 'L301', 3, 17.4),
(timestamp '2017/01/01 13:00:45.000', 'L301', 3, 13.5);
