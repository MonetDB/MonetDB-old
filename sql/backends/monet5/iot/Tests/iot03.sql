-- Setting stream parameters after iot00
set schema iot;
set optimizer='iot_pipe';

create stream table stmp3 (t timestamp, sensor integer, val decimal(8,2)) ;
-- Example of a window based action
create table result3(like stmp3);

create procedure cq03()
begin
	call iot.window('iot','stmp3',2);
	insert into result3 select min(t), count(*), avg(val) from stmp3;
end;
call iot.query('iot','cq03');
call iot.pause('iot','cq03');

insert into stmp3 values('2005-09-23 12:34:26.000',1,9.0);
insert into stmp3 values('2005-09-23 12:34:27.000',1,11.0);
insert into stmp3 values('2005-09-23 12:34:28.000',1,13.0);
insert into stmp3 values('2005-09-23 12:34:28.000',1,15.0);

call iot.resume('iot','cq03');
call iot.wait(4);

select * from stmp3;
select * from result3;

call iot.stop();
--select * from  iot.queries();
drop table stmp3;
drop table result3;
