-- Example of a window based action
create stream table stmp2 (t timestamp, sensor integer, val decimal(8,2)) ;
create table result2(like stmp2);

-- CREATE CONTINUOUS QUERY cq_window
create procedure cq_window()
begin
	-- The window ensures a maximal number of tuples to consider
	-- Could be considered a property of the stream table
    call cquery.window('sys','stmp2',2);
    insert into result2 select * from stmp2 where val >12;
end;
call cquery.register('sys','cq_window');

insert into stmp2 values('2005-09-23 12:34:26.000',1,9.0);
insert into stmp2 values('2005-09-23 12:34:27.000',1,11.0);
insert into stmp2 values('2005-09-23 12:34:28.000',1,13.0);
insert into stmp2 values('2005-09-23 12:34:28.000',1,15.0);

-- START cq_window;
call cquery.resume('sys','cq_window');

-- wait for a few seconds for scheduler to do work
call cquery.wait(1000);

-- STOP cq_window;
call cquery.pause('sys','cq_window'));

select 'RESULT';
select * from stmp2;
select * from result2;

select * from cquery.log('sys','cq_window');

-- ideally auto remove upon dropping the procedure
call cquery.deregister('sys','cq_window');

drop procedure cq_splitter;
drop table stmp2;
drop table result2;
