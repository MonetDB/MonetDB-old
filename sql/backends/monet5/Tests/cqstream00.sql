-- Example of a stream splitter
create stream table stmp2 (t timestamp, sensor integer, val decimal(8,2)) ;
create table result1(like stmp2);
create table result2(like stmp2);

-- CREATE CONTINUOUS QUERY cq_splitter
create procedure cq_splitter()
begin
	-- If you use a stream table there should be a tumble option being set (by default all)
    call cquery.tumble('sys','stmp2',1); -- consume one tuple at a time
    insert into result1 select * from stmp2 where val <12;
    insert into result2 select * from stmp2 where val >12;
end;
call cquery.new('sys','cq_splitter');

-- The stream use in the CQ determines the activation and sets the scheduler heartbeat to -1
-- If set explictly it overrules the window based bounds
call cquery.heartbeat(-1);

insert into stmp2 values('2005-09-23 12:34:26.000',1,11.0);
insert into stmp2 values('2005-09-23 12:34:27.000',1,11.0);
insert into stmp2 values('2005-09-23 12:34:28.000',1,13.0);
insert into stmp2 values('2005-09-23 12:34:28.000',1,13.0);

-- START cq_splitter;
call cquery.resume('sys','cq_splitter');

-- wait for a few seconds for scheduler to do work
call cquery.wait(1000);

-- STOP cq_splitter;
call cquery.pause('sys','cq_splitter'));

select 'RESULT';
select * from stmp2;
select * from result1;
select * from result2;

select * from cquery.log('sys','cq_splitter');

-- ideally auto remove upon dropping the procedure
call cquery.release('sys','cq_splitter');

drop procedure cq_splitter;
drop table stmp2;
drop table result1;
drop table result2;
