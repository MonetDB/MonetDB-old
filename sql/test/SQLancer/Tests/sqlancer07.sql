START TRANSACTION;
CREATE TABLE "t1" ("c0" DECIMAL(18,3),"c1" BINARY LARGE OBJECT NOT NULL,"c2" DECIMAL(18,3),CONSTRAINT "t1_c1_unique" UNIQUE ("c1"));
PREPARE SELECT DISTINCT (SELECT DISTINCT r'|m<kv' FROM t1 WHERE ((t1.c0)<(?)) GROUP BY t1.c2, ?), ?, t1.c2 FROM t1 WHERE CAST(? AS BOOLEAN) LIMIT 2103332269785059850;
	-- Could not determine type for argument number 2
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0"("c0" INTERVAL MONTH,"c1" INTERVAL SECOND);
CREATE TABLE "t1"("c0" INTERVAL MONTH,"c1" INTERVAL SECOND);
CREATE TABLE "t2"("c0" INTERVAL MONTH,"c1" INTERVAL SECOND);

SELECT 1 FROM t1 JOIN t2 ON NOT (NOT (SELECT FALSE FROM t2));
	-- empty
SELECT ALL CAST(NOT ((DATE '1970-01-22') NOT IN (DATE '1970-01-04')) AS INT) as count FROM t0, t1 FULL OUTER JOIN t2 ON NOT (NOT ((SELECT DISTINCT FALSE FROM t2, t0, t1)));
	-- empty
ROLLBACK;

CREATE TABLE "t0" ("c0" DOUBLE PRECISION NOT NULL);
INSERT INTO "t0" VALUES (0.13492451886840173);
INSERT INTO "t0" VALUES (79004262);
INSERT INTO "t0" VALUES (1476461507);
CREATE TABLE "t1" ("c0" DOUBLE PRECISION);
INSERT INTO "t1" VALUES (973588428);
INSERT INTO "t1" VALUES (NULL);
INSERT INTO "t1" VALUES (0.39517295223772886);
INSERT INTO "t1" VALUES (NULL);

SELECT t0.c0 BETWEEN SYMMETRIC (SELECT t0.c0 FROM t0) AND t0.c0 FROM t0;
	--error, more than one row returned by a subquery used as an expression
SELECT CAST(NOT ((t0.c0) NOT BETWEEN SYMMETRIC ((SELECT DISTINCT t0.c0 FROM t0, t1)) AND (t0.c0)) AS INT) FROM t0;
	--error, more than one row returned by a subquery used as an expression
DROP TABLE t0;
DROP TABLE t1;

START TRANSACTION;
CREATE TABLE "t0" ("c0" DATE NOT NULL);
CREATE TABLE "t1" ("c1" INT);
CREATE TABLE "t2" ("c0" DATE NOT NULL);

SELECT 0 <= ANY(SELECT CASE INTERVAL '2' SECOND WHEN INTERVAL '6' SECOND THEN t1.c1 ELSE (SELECT t1.c1 FROM t1) END FROM t1) FROM t1;
	-- empty
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" BOOLEAN NOT NULL);
CREATE TABLE "t1" ("c0" DECIMAL(18,3));
CREATE TABLE "t2" ("c0" DECIMAL(18,3),"c2" DATE);
PREPARE (SELECT DISTINCT t0.c0, INTERVAL '1734780053' SECOND FROM t0, t1) UNION ALL (SELECT ?, ? FROM t2);
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" INTERVAL MONTH NOT NULL,CONSTRAINT "t0_c0_pkey" PRIMARY KEY ("c0"),CONSTRAINT "t0_c0_unique" UNIQUE ("c0"),CONSTRAINT "t0_c0_unique" UNIQUE ("c0"));
INSERT INTO "t0" VALUES (INTERVAL '2101098338' MONTH);
CREATE TABLE "t1" ("c0" INTERVAL MONTH,"c1" BOOLEAN);
CREATE TABLE "t2" ("c0" INTERVAL MONTH);

SELECT CAST(t1.c1 AS INT) FROM t1, t0 RIGHT OUTER JOIN t2 ON ((SELECT DISTINCT t1.c0 FROM t1, t0 WHERE FALSE)) NOT IN (t0.c0);
	-- empty
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" DATE,"c2" INTEGER);
CREATE TABLE "t1" ("c1" TIMESTAMP,"c2" INTEGER);
CREATE TABLE "t2" ("c0" DATE,"c1" TIMESTAMP,"c2" INTEGER);
PREPARE (SELECT ?, t1.c2 FROM t1, t0 WHERE (SELECT DISTINCT (t1.c2) BETWEEN ASYMMETRIC (?) AND (t1.c2) FROM t1 CROSS JOIN 
((SELECT DISTINCT 6.9089063E7, TRUE FROM t2 WHERE TRUE) EXCEPT (SELECT ALL 0.4, FALSE FROM t2, t1 INNER JOIN t0 ON FALSE)) AS sub0 WHERE FALSE)) INTERSECT DISTINCT (SELECT DISTINCT 0.2, ? FROM t0, t2 WHERE ?);
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" VARCHAR(156) NOT NULL);
CREATE TABLE "t1" ("c0" VARCHAR(156) NOT NULL);
select 1 from t0, t1 inner join t0 on true;
	--error table name "t0" specified more than once
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" CHAR(315));
INSERT INTO "t0" VALUES ('(y/l'), (''), ('*J');
CREATE TABLE "t1" ("c0" VARCHAR(512), "c1" DOUBLE PRECISION);
INSERT INTO "t1" VALUES ('0.9295919173154146', NULL);
CREATE TABLE "t2" ("c0" CHAR(315));
INSERT INTO "t2" VALUES ('1753268987.000'), ('(y/l'), ('(y/l'), ('');

SELECT '1' > COALESCE(((SELECT t1.c0 FROM t1) INTERSECT (SELECT '2')), t0.c0) FROM t0;
	-- True
	-- True
	-- True
SELECT CAST(((COALESCE(t0.c0, t0.c0, t0.c0))>(COALESCE(((SELECT DISTINCT t1.c0 FROM t1, t0, t2 WHERE FALSE) INTERSECT DISTINCT (SELECT ALL t0.c0 FROM t0 WHERE TRUE)), 
t0.c0, CASE INTERVAL '720497648' MONTH WHEN INTERVAL '1899785652' MONTH THEN t0.c0 ELSE t0.c0 END))) AS INT) FROM t0;
	-- 0
	-- 0
	-- 0
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" VARCHAR(156) NOT NULL);
INSERT INTO "t0" VALUES ('');
CREATE TABLE "t1" ("c0" VARCHAR(156) NOT NULL);
INSERT INTO "t1" VALUES ('i?#\t+U,s'), ('288044674'), ('b'), ('W?ykP7L+X'), ('34076821'), ('ah'), ('﹂9j0M4');

SELECT CASE '1' WHEN COALESCE((SELECT t0.c0 FROM t0), COALESCE(t1.c0, '2')) THEN 1 END FROM t1;
	-- 7 NULL rows
ROLLBACK;

CREATE TABLE "t0" ("c1" VARCHAR(388) NOT NULL,CONSTRAINT "t0_c1_pkey" PRIMARY KEY ("c1"),CONSTRAINT "t0_c1_unique" UNIQUE ("c1"));
INSERT INTO "t0" VALUES (''), ('Ral%}?U*A'), ('Dz '), ('P');
CREATE TABLE "t2" ("c1" VARCHAR(388) NOT NULL);
INSERT INTO "t2" VALUES ('4'),('4'),('3eSU8,'),(''),('5E~쟱'),('~'),('1386006226'),('0.19005213960704492'),('''{Mdd뒆VB'''),('\015␱%L%]'),('+'),(''),('㕚o+k');

select t0.c1 from t0 where (-5) in (case when t0.c1 then 1 end, (select -3 from t0));
	-- error while converting empty string to bit
select t0.c1 from t0 where (-5) in (case when t0.c1 then 1 else -2 end, (select -3 from t0, t2 where false));
	-- error while converting empty string to bit
DROP TABLE t0;
DROP TABLE t2;

CREATE TABLE t1 (c0 DOUBLE PRECISION NOT NULL);
INSERT INTO t1(c0) VALUES((0.6548429615298178*0.3050008736497528)/(0.5061323979270875)); --error, scale too large
INSERT INTO t1(c0) VALUES(((((COALESCE(0.6548429615298178, 0.20317629894456002))*(COALESCE(0.3050008736497528, 0.2277902039419617))))/(0.5061323979270875))); --error, scale too large
SELECT t1.c0 FROM t1;
	-- empty
DROP TABLE t1;

START TRANSACTION;
CREATE TABLE "t0" ("c0" DECIMAL(18,3) NOT NULL,"c1" DOUBLE PRECISION NOT NULL);
CREATE TABLE "t1" ("c1" DOUBLE PRECISION);
CREATE TABLE "t2" ("c0" DECIMAL(18,3),"c1" DOUBLE PRECISION);

SELECT (SELECT 1 FROM t1, t2, t0 JOIN (SELECT 1) AS sub0 ON (t1.c1) BETWEEN (t2.c1) AND (t1.c1)) FROM t2, t0 CROSS JOIN t1;
	-- empty
ROLLBACK;

START TRANSACTION;
CREATE TABLE "t0" ("c0" BIGINT NOT NULL,CONSTRAINT "t0_c0_pkey" PRIMARY KEY ("c0"),CONSTRAINT "t0_c0_unique" UNIQUE ("c0"));
INSERT INTO "t0" VALUES (0),(-1557127883),(-488477810);
CREATE TABLE "t1" ("c0" BIGINT NOT NULL);
INSERT INTO "t1" VALUES (1457011207),(98933083),(1259938486);
CREATE TABLE "t2" ("c0" BIGINT NOT NULL,CONSTRAINT "t2_c0_pkey" PRIMARY KEY ("c0"));
INSERT INTO "t2" VALUES (596983192), (-601428889), (1688368391);

SELECT t1.c0 FROM t2, t0 CROSS JOIN t1 WHERE ((((t0.c0)%((SELECT DISTINCT t0.c0 FROM t1, t0, t2 WHERE FALSE))))<=(t1.c0));
	-- empty
SELECT CAST(SUM(count) AS BIGINT) FROM (SELECT ALL CAST(((((t0.c0)%((SELECT DISTINCT t0.c0 FROM t1, t0, t2 WHERE FALSE))))<=(t1.c0)) AS INT) as count FROM t2, t0 CROSS JOIN t1) as res;
	-- NULL
ROLLBACK;
