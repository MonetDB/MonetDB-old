-- Introduce firing conditions to the queries themselves.
set schema iot;
set optimizer='iot_pipe';

create stream table tmp13 (t timestamp, sensor integer, val decimal(8,2)) ;
create table agenda13(i integer, msg string);

-- Queries can fire based on both the actual state of the streams and heartbeat
call iot.window('iot','tmp13',2);

-- every second inspect the basket regardless filling
call iot.heartbeat('iot','tmp13',1000);

create procedure cq13()
begin
	insert into agenda13 select count(*), 'full batch' from tmp13;
end;

call iot.query('iot','cq13');
call iot.pause();

insert into tmp13 values('2005-09-23 12:34:26.736',1,12.34);
insert into tmp13 values('2005-09-23 12:34:26.736',1,12.35);
insert into tmp13 values('2005-09-23 12:34:26.736',1,12.36);
--insert into tmp13 values('2005-09-23 12:34:26.736',1,12.37);
select * from tmp13;

-- reactivate all continuous queries
call iot.resume();
-- wait a few seconds
call iot.wait(5000);

select 'RESULT';
select * from agenda13;

call iot.stop();
select * from iot.errors();
drop procedure cq13;
drop table tmp13;
drop table agenda13;
