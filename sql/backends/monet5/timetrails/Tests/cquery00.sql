-- A simple continuous query.
set optimizer='cquery_pipe';

create table stmp (t timestamp, sensor integer, val decimal(8,2)) ;
create table result(like stmp);

create procedure cq_basic()
begin
	call cquery.heartbeat(50);
	insert into result select min(t), count(*), avg(val) from stmp;
end;

call cquery.new('iot','cq00');

insert into stmp values('2005-09-23 12:34:26.736',1,12.34);
select * from stmp;

-- reactivate all continuous queries
call cquery.resume();
-- wait for 2 seconds 
call cquery.wait(2000);
call cquery.pause();

select 'RESULT';
select * from result;

select * from cquery.summary();
select * from cquery.log();

drop procedure cq_basic;
drop table stmp;
drop table result;
