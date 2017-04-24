-- A simple continuous query.
set schema iot;
set optimizer='iot_pipe';

create stream table tmp06 (t timestamp, sensor integer, val decimal(8,2)) ;
create table result(like tmp06);

create procedure cq06()
begin
	insert into result select min(t), count(*), avg(val) from tmp06;
end;

call iot.query('iot','cq06');
call iot.pause();

call iot.show('iot','cq06');

select iot.gettumble('iot','tmp06');
call iot.tumble('iot','tmp06',5);
select iot.gettumble('iot','tmp06');

select iot.getwindow('iot','tmp06');
call iot.window('iot','tmp06',7);
select iot.getwindow('iot','tmp06');

select iot.getheartbeat('iot','tmp06');
call iot.heartbeat('iot','tmp06',8);
select iot.getheartbeat('iot','tmp06');

select * from iot.errors();

drop procedure cq06;
drop table tmp06;
drop table result;
