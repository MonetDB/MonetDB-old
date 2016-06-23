-- Introduce firing conditions to the queries themselves.
set schema iot;
set optimizer='iot_pipe';

create stream table tmp13 (t timestamp, sensor integer, val decimal(8,2)) ;
create table agenda(like tmp13);

-- Queries can fire based on the actual state of the streams and heartbeat
call iot.window('iot','tmp13',4);
-- at least one tuple in basket
call iot.window('iot','cq13b','iot','tmp13',1);

-- every 5 seconds inspect the basket regardless filling
call iot.heartbeat('iot','cq13a',5000);

create procedure cq13a()
begin
	insert into agenda select count(*), 'full batch' from tmp13;
end;

create procedure cq13b()
begin
	insert into agenda select count(*), 'partial batch' from tmp13;
end;

call iot.query('iot','cq13a');
call iot.query('iot','cq13b');
call iot.pause();

insert into tmp13 values('2005-09-23 12:34:26.736',1,12.34);
select * from tmp13;

-- reactivate all continuous queries
call iot.resume();
-- wait for 1 cycle in the scheduler
call iot.wait(1);

select 'RESULT';
select * from agenda;

select * from iot.errors();
call iot.stop();
drop procedure cq13a;
drop procedure cq13b;
drop table tmp13;
drop table agenda;
