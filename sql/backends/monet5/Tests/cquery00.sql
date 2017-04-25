-- A simple continuous query over non-stream relations
-- controlled by a heartbeat.
create table result(i integer);

create procedure cq_basic()
begin
	insert into result values(select count(*) from result);
end;

-- The scheduler executes all CQ every 50 milliseconds
call cquery.heartbeat(50);

-- register the CQ
call cquery.new('iot','cq_basic');

-- reactivate all continuous queries
call cquery.resume();
call cquery.wait(2000);
call cquery.pause();

select 'RESULT';
select * from result;

select * from cquery.summary();
select * from cquery.log();

-- ideally auto remove upon dropping the procedure
call cquery.release('sys','cq_basic');

drop procedure cq_basic;
drop table result;
