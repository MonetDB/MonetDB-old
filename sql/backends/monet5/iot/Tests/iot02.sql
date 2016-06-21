-- Setting stream parameters after iot00
set schema iot;
set optimizer='iot_pipe';

-- Example of a stream splitter
create stream table stmp2 (t timestamp, sensor integer, val decimal(8,2)) ;
create table result1(like stmp2);
create table result2(like stmp2);

create procedure cq02()
begin
	call iot.tumble('iot','stmp2',1);
	insert into result1 select * from stmp2 where val <12;
	insert into result2 select * from stmp2 where val >12;
end;
call iot.query('iot','cq02');
call iot.pause('iot','cq02');

insert into stmp2 values('2005-09-23 12:34:26.000',1,11.0);
insert into stmp2 values('2005-09-23 12:34:27.000',1,11.0);
insert into stmp2 values('2005-09-23 12:34:28.000',1,13.0);
insert into stmp2 values('2005-09-23 12:34:28.000',1,13.0);

call iot.resume('iot','cq02');

-- wait for 1 cycle in the scheduler
call iot.wait(5);

select 'RESULT';
select * from stmp2;
select * from result1;
select * from result2;

call iot.stop();
select * from iot.errors();
drop procedure cq02;
drop table stmp2;
drop table result1;
drop table result2;
