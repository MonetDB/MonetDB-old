-- test to read the baskets from a source
set schema iot;

create stream table temps( iotclk timestamp, room string , temperature real);

declare basketdir string;
set basketdir= '/ufs/mk/baskets/measures/temperatures/';

explain call iot.import('iot','temps', concat(basketdir,'1'));
call iot.import('iot','temps', concat(basketdir,'1'));

select * from iot.errors();

select * from temps;
drop table temps;
