-- A simple continuous query.
set schema iot;

create stream table stream_tmp (t timestamp, sensor integer, val decimal(8,2)) ;
create table result(like stream_tmp);

create procedure cq00()
begin
	insert into result select min(t), count(*), avg(val) from stream_tmp;
end;

call iot.query('iot','cq00');
call iot.query('insert into result select min(t), count(*), avg(val) from stream_tmp;');

select * from  iot.baskets();
select * from  iot.queries();
select * from  iot.inputplaces();
select * from  iot.outputplaces();
