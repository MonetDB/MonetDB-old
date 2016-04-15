-- A simple continuous query.
set schema iot;

create stream table stream_tmp (t timestamp, sensor integer, val decimal(8,2)) ;
create table result(like stream_tmp);

create procedure cq00()
begin
	insert into result select min(t), count(*), avg(val) from stream_tmp;
end;

call iot.query('iot','cq00');
call iot.query('insert into iot.result select min(t), count(*), avg(val) from iot.stream_tmp;');

select * from  iot.baskets();
select * from  iot.queries();
select * from  iot.inputplaces();
select * from  iot.outputplaces();

-- stop all continuous queries
call iot.deactivate();

insert into stream_tmp values('2005-09-23 12:34:26.736',1,12.34);
select * from stream_tmp;

-- reactivate all continuous queries
call iot.activate();

select * from  iot.baskets();
select * from  iot.queries();
