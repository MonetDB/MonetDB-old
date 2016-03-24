-- Clear the stream testing environment
set schema iot;
set optimizer='iot_pipe';

create stream table stream_tmp (t timestamp, sensor integer, val decimal(8,2)) ;
create table result(like stream_tmp);

create procedure cq00()
begin
	insert into result select min(t), count(*), avg(val) from stream_tmp;
end;

call iot.query('iot','cq00');
call iot.query('insert into result select min(t), count(*), avg(val) from stream_tmp;');

call iot.baskets();
call iot.petrinet();

call iot.dump();
call iot.drop();
call iot.dump();
