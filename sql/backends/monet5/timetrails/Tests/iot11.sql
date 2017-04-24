-- check proper deletion from a stream table
set schema iot;
set optimizer='iot_pipe';

create stream table sdel11 (t timestamp, sensor integer, val decimal(8,2)) ;

insert into sdel11 values('2005-09-23 12:34:26.736',1,12.31);
insert into sdel11 values('2005-09-23 12:34:26.736',2,12.32);
insert into sdel11 values('2005-09-23 12:34:26.736',3,12.33);
insert into sdel11 values('2005-09-23 12:34:26.736',4,12.34);
insert into sdel11 values('2005-09-23 12:34:26.736',2,12.35);

-- don't remove tuples automatically
call iot.keep('iot','sdel11');

explain select * from sdel11;
explain delete from sdel11 where sensor = 2;

select * from sdel11;
delete from sdel11 where sensor = 2;
select * from sdel11;

drop table sdel11;
