-- A simple continuous query over non-stream relations
-- controlled by a heartbeat.
create table tmp.result(i integer);

create procedure cq_basic()
begin
	insert into tmp.result (select count(*) from tmp.result);
end;

-- register the CQ
call cquery.register('sys','cq_basic');

-- The scheduler executes this CQ every 50 milliseconds
call cquery.heartbeat('sys','cq_basic',50);

-- reactivate all continuous queries
call cquery.resume();
call cquery.wait(2000);
call cquery.pause();

select 'RESULT';
select * from tmp.result;

select * from cquery.summary();
select * from cquery.log();

-- ideally auto remove upon dropping the procedure
call cquery.deregister('sys','cq_basic');

drop procedure cq_basic;
drop table tmp.result;
