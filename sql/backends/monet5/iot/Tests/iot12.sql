-- introduce a heartbeat query
set schema iot;
set optimizer='iot_pipe';

-- we don't have global variables
create stream table clocks(cnt integer,clk integer);
insert into clocks values(0,0);

create table clocklog( t timestamp, clk integer);

call iot.heartbeat('iot','clocks',2000);

create procedure clk()
begin
	update iot.clocks
		set clk = clk+1,
		cnt = cnt +1;
	insert into clocklog values (current_timestamp(),(select clk from iot.clocks));
end;

select * from clocks;

-- run a continuous query a limited number of times
call iot.query('iot','clk',5);

-- wait long enough to let the cycles run
call iot.wait(5000);
select * from clocks;
--select * from clocklog;
call iot.wait(5000);
select * from clocks;
--select * from clocklog;
call iot.wait(5000);
select * from clocks;
--select * from clocklog;
--select * from  iot.baskets();
--select * from  iot.queries();
select * from iot.errors();
drop procedure clk;
drop table clocks;
drop table clocklog;
