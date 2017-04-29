-- A simple continuous query over non-stream relations
-- controlled by a cycle count
create table result(i integer);

create procedure cq_cycles()
begin
	insert into result values(select count(*) from result);
end;

-- The scheduler executes all CQ at most 5 rounds
call cquery.cycles(5);

-- register the CQ
call cquery.register('iot','cq_cycles');

-- reactivate all continuous queries
call cquery.resume();
call cquery.wait(2000);
call cquery.pause();

select 'RESULT';
select * from result;

select * from cquery.summary();
select * from cquery.log();

-- ideally auto remove upon dropping the procedure
call cquery.deregister('sys','cq_cycles');

drop procedure cq_cycles;
drop table result;
