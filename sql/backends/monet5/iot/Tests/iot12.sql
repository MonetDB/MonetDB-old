-- introduce a heartbeat query
set schema iot;
set optimizer='iot_pipe';

-- we don't have global variables
create stream table clocks(cnt integer,clk1 integer, clk2 integer);
insert into clocks values(0,0,0);

call iot.heartbeat('iot','clocks',1000);
create procedure clk1()
begin
	update iot.clocks
		set clk1 = clk1+1,
		cnt = cnt +1;
end;

create procedure clk3()
begin
	update clocks
		set clk1 = clk1+1,
			clk2 = clk2+2,
			cnt = cnt +1;
end;

-- alternative is a simple query
call iot.query('iot','clk1');
call iot.query('iot','clk3');
call iot.pause();

--select * from  iot.baskets();
--select * from  iot.queries();

select * from clocks;
call iot.resume();
call iot.wait(5);
select * from clocks;

call iot.stop();
select * from iot.errors();
drop procedure clk1;
drop procedure clk3;
drop table clocks;

