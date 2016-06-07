-- Setting stream parameters after iot00
set schema iot;
set optimizer='iot_pipe';

-- Example of a stream splitter
create table result1(like stmp);
create table result2(like stmp);

create procedure cq02()
begin
	insert into result1 select * from stmp;
	insert into result2 select * from stmp;
	call iot.tumble('iot','stmp');
end;
call iot.query('iot','cq02');

insert into stmp values('2005-09-23 12:34:26.000',1,11.0);
insert into stmp values('2005-09-23 12:34:27.000',1,11.0);
insert into stmp values('2005-09-23 12:34:28.000',1,13.0);
insert into stmp values('2005-09-23 12:34:28.000',1,13.0);

call iot.deactivate('iot','cq02');

select * from stmp;
select * from result1;
select * from result2;

select * from  iot.queries();
