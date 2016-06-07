-- test to read the baskets from a source
set schema iot;
create stream table temps( iotclk timestamp, room string , temperature real);

declare basketdir string;
set basketdir= '/ufs/mk/baskets/measures/temperatures/';

call iot.receptor('iot','temps', concat(basketdir,'1'));

select * from temps;

drop table temps;
