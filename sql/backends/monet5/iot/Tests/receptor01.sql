-- test to read the baskets from a source
set schema iot;
set optimizer='iot_pipe';

create stream table temps( iotclk timestamp, room string , temperature real);

declare basketdir string;
set basketdir= '/ufs/mk/baskets/measures/temperatures/';

call iot.receptor('iot','temps', concat(basketdir,'1'));

call iot.wait(2);
select * from temps;

drop table temps;
