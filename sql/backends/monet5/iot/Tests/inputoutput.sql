-- Setting stream parameters after iot00
set schema iot;
set optimizer='iot_pipe';

-- simple input/output pipeline
create stream table input1 (t timestamp, sensor integer, val decimal(8,2)) ;
create stream table output1 (t timestamp, sensor integer, val decimal(8,2)) ;

create procedure inout()
begin
	insert into output1 select * from input1;
end;
call iot.query('iot','inout');
call iot.pause('iot','inout');

insert into input1 values('2005-09-23 12:34:26.000',1,11.00);
insert into input1 values('2005-09-23 12:34:27.000',1,11.00);
insert into input1 values('2005-09-23 12:34:28.000',1,13.00);
insert into input1 values('2005-09-23 12:34:28.000',1,13.00);

call iot.resume('iot','inout');

-- wait for 1 cycle in the scheduler
call iot.wait(1);

select 'RESULT';
select * from input1;
select * from output1;

