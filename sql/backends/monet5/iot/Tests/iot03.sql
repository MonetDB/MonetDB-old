-- Setting stream parameters after iot00
set schema iot;
set optimizer='iot_pipe';

-- Example of a window based action
create table result3(like stmp);

create procedure cq03()
begin
	call iot.window('iot','stmp',2);
	insert into result3 select min(t), count(*), avg(val) from stmp;
	call iot.tumble('iot','stmp',2);
end;
call iot.query('iot','cq03');

insert into stmp values('2005-09-23 12:34:26.000',1,11.0);
insert into stmp values('2005-09-23 12:34:27.000',1,11.0);
insert into stmp values('2005-09-23 12:34:28.000',1,13.0);
insert into stmp values('2005-09-23 12:34:28.000',1,13.0);

call iot.deactivate('iot','cq03');

select * from stmp;
select * from result3;

select * from  iot.queries();
