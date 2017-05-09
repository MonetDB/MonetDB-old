-- on-off query problem. Find all the values that form a consecutive list

CREATE TABLE coil( tick integer, level integer);

INSERT INTO coil
VALUES
(1,0),(2,0),(3,0),(4,0),(5,1),(6,1),(7,1),(8,0),(9,2),(10,2),(11,2),(12,2),(13,0),(14,0),(15,3),(16,0),(17,0),(18,0),(19,4),(20,4),(21,4),(22,5),(23,5),(24,5),(25,0),(26,0),(27,6),(28,0),(29,6),(30,0),(31,7),(32,7),(33,7),(34,7),(35,8),(36,8),(37,8),(38,8),(39,8),(40,7),(41,7),(42,7),(43,7),(44,9),(45,9),(46,9),(47,0),(48,0),(49,0);

select A.level, count(*)from coil A, coil B where A.tick+1 = B.tick and A.level <> B.level group by A.level;
