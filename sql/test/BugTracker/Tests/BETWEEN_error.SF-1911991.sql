create table t1911991 (id int);
insert into t1911991 values (0);
insert into t1911991 values (1);
insert into t1911991 values (2);
insert into t1911991 values (3);
insert into t1911991 values (4);
insert into t1911991 values (5);
insert into t1911991 values (6);
insert into t1911991 values (7);
insert into t1911991 values (8);
insert into t1911991 values (9);
select * from t1911991;
select * from t1911991 WHERE (id - 5) BETWEEN 2 AND 7;
select * from t1911991 WHERE (4 - id) BETWEEN 2 AND 7;
drop table t1911991;
