-- removal from a stream
set schema iot;

create stream table sdel (t timestamp, sensor integer, val decimal(8,2)) ;

insert into sdel values('2005-09-23 12:34:26.736',1,12.34);
insert into sdel values('2005-09-23 12:34:26.736',2,12.34);
insert into sdel values('2005-09-23 12:34:26.736',3,12.34);
insert into sdel values('2005-09-23 12:34:26.736',4,12.34);
select * from sdel;

delete from sdel where sensor = 2;
select * from sdel;

-- don't remove tuples automatically
call iot.tumble('iot','sdel',0);

create procedure sdel00()
begin
	delete from sdel where sensor = 3;
end;

call iot.query('iot','sdel00',3);
call iot.show('iot','sdel00');
call iot.pause();
select * from sdel;
insert into sdel values('2005-09-23 12:34:26.736',1,12.34);
insert into sdel values('2005-09-23 12:34:26.736',3,12.34);
insert into sdel values('2005-09-23 12:34:26.736',4,12.34);
insert into sdel values('2005-09-23 12:34:26.736',3,12.34);
select * from sdel;

call iot.resume();
-- wait a few seconds
call iot.wait(2000);
-- stop the scheduler because it remains enabled
call iot.stop();
select 'final state';
select * from sdel;

drop procedure sdel00;
drop table sdel;
