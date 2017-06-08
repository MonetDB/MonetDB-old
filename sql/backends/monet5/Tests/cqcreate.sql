CREATE stream TABLE testing (a int);

CREATE TABLE results (b int);

CREATE CONTINUOUS QUERY stressing() BEGIN INSERT INTO results SELECT a FROM testing; END;
