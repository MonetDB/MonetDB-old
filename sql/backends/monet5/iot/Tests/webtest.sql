set schema iot;
create stream table temps( iotclk timestamp, room string , temperature real);
-- remainder depends on location of the baskets root

declare baskets string;
set baskets= '/ufs/mk/baskets/measures/temperatures/';

call iot.basket('iot','temps', concat(baskets,'1'));
select * from temps;
call iot.basket('iot','temps', concat(baskets,'1'));
select * from temps;
