-- test to read the baskets from a source and react to it
set schema iot;
set optimizer='iot_pipe';

declare basketdir string;
set basketdir= '/ufs/mk/baskets/measures/temperatures/';

create stream table temps( iotclk timestamp, room string , temperature real);
create table tempresult(like temps);

call iot.receptor('iot','temps', concat(basketdir,'1'));

create procedure collecttemps()
begin
	insert into tempresult select min(iotclk), count(*), avg(temperature) from temps;
end;

call iot.query('iot','collecttemps');
call iot.resume();
call iot.wait(2);
select * from temps;
select * from tempresult;

call iot.stop();
drop procedure collecttemps;
drop table tempresult;
drop table temps;
