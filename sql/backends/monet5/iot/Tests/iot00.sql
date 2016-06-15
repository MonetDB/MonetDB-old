-- A simple continuous query.
set schema iot;
set optimizer='iot_pipe';

create stream table stmp (t timestamp, sensor integer, val decimal(8,2)) ;
create table result(like stmp);

create procedure cq00()
begin
	insert into result select min(t), count(*), avg(val) from stmp;
end;

call iot.query('iot','cq00');
--call iot.query('insert into iot.result select min(t), count(*), avg(val) from iot.stmp;');

select * from  iot.baskets();
select * from  iot.queries();
select * from  iot.inputs();
select * from  iot.outputs();

-- stop all continuous queries and wait for it
call iot.pause();

insert into stmp values('2005-09-23 12:34:26.736',1,12.34);
select * from stmp;

-- reactivate all continuous queries
call iot.resume();
-- wait for 1 cycle in the scheduler
call iot.wait(1);

select 'RESULT';
select * from result;

--select * from  iot.baskets();
--select * from  iot.queries();
