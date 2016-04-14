CREATE TABLE points1(x BIGINT, y BIGINT);
CREATE TABLE points2(x BIGINT, y BIGINT);

INSERT INTO points1 VALUES (100, 100), (150, 100), (100, 150), (150,150);
INSERT INTO points2 VALUES (99, 101),  (149, 149), (101, 180), (149,151);

-- selection
SELECT * FROM points1 p1 WHERE [p1.x, p1.y] distance [ 99, 99, 5];
SELECT * FROM points1 p1 WHERE [p1.x, p1.y] distance [125,150,30];
SELECT * FROM points2 p2 WHERE [p2.x, p2.y] distance [150,150, 5];

-- join
--SELECT * FROM points1 p1, points2 p2 WHERE [p1.x, p1.y] distance [p2.x, p2.x, 5] ORDER BY p1.x, p1.y;

DROP TABLE points1;
DROP TABLE points2;
