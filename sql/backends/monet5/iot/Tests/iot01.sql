-- Setting stream parameters after iot00
set schema iot;
set optimizer='iot_pipe';

-- Eating away the tuples from the stream
create procedure cq01()
begin
	insert into result select min(t), count(*), avg(val) from stmp;
	call iot.tumble('iot','stmp'); 
end;
call iot.query('iot','cq01');

insert into stmp values('2005-09-23 12:34:26.000',1,11.0);
insert into stmp values('2005-09-23 12:34:27.000',1,12.0);
insert into stmp values('2005-09-23 12:34:28.000',1,13.0);

-- deactivate all when streams are empty.
call iot.deactivate();

-- stream table should be empty now
select * from stmp;
-- and result delivered
select * from result;

select * from  iot.queries();
