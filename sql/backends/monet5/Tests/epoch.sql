CREATE TABLE justatest (justacolumn timestamp);

INSERT INTO justatest VALUES (timestamp '2017/01/01 09:00:00.000'),(timestamp '2016/04/09 08:21:22.000'),
(timestamp '1950/12/11 00:21:22.000'),(timestamp '1977/12/11 23:59:59.999');

EXPLAIN SELECT epoch(justacolumn) FROM justatest;
SELECT epoch(justacolumn) FROM justatest;

DROP TABLE justatest;
