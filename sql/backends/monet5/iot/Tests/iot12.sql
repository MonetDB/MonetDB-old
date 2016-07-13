-- introduce a heartbeat query
set schema iot;
set optimizer='iot_pipe';

-- we don't have global variables
create stream table clocks(cnt integer,clk1 integer);
insert into clocks values(0,0);

call iot.heartbeat('iot','clocks',5);
create procedure clk1()
begin
	update iot.clocks
		set clk1 = clk1+1,
		cnt = cnt +1;
end;

-- alternative is a simple query
call iot.query('iot','clk1');
call iot.pause();

select * from clocks;
call iot.cycles('iot','clk1',5);
call iot.resume();
select * from clocks;

-- wait long enough to let the cycles run
call iot.wait(1000);
call iot.stop();
--select * from  iot.baskets();
select * from  iot.queries();
select * from iot.errors();
drop procedure clk1;
drop table clocks;

