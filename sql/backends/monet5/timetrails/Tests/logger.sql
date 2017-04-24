-- A simple heartbeat continuous query triggered.
set schema iot;
set optimizer='iot_pipe';

create stream table log(t timestamp,b integer);

create procedure cqlogger()
begin
	insert into log values(now(), iot.getheartbeat('iot','log'));
end;

call iot.heartbeat('iot','log',1000);
call iot.query('iot','cqlogger');

-- wait for 2 seconds
call iot.wait(4000);

select 'RESULT';
select b from log;

select * from iot.errors();
call iot.stop();
drop procedure cqlogger;
drop table log;
