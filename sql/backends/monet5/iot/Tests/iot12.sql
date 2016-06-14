-- introduce a heartbeat query
set schema iot;
set optimizer='iot_pipe';

declare hbclk1 integer;
declare hbclk2 integer;
declare cnt integer;

set hbclk1 = 0;
set hbclk2 = 0;
set cnt = 0;

-- continuous queries should be encapsulated in procedures
-- this way their naming becomes easier, and mult-statement
-- actions are better supported.

--However, these queries won't run because the SQL context
--holding the variables is not generally known
create procedure clk1()
begin
	set hbclk1 = hbclk1+1;
end;

create procedure clk3()
begin
	set hbclk1 = hbclk1+1;
	set hbclk2 = hbclk2+2;
	--set cnt =(select count(*) from stmp);
end;

-- alternative is a simple query
call iot.query('iot','clk1');
call iot.query('iot','clk3');

select * from  iot.baskets();
select * from  iot.queries();

select hbclk1, hbclk2;
call iot.resume();
call iot.wait(2);
select hbclk1, hbclk2;


