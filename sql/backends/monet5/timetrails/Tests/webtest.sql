set schema iot;
set optimizer='iot_pipe';
create stream table ntemps( iotclk timestamp, room string , temperature real);
create table atemps( iotclk timestamp, cnt int , temperature real);
-- remainder depends on location of the baskets root

declare baskets string;
set baskets= '/ufs/mk/baskets/measures/temperatures/';

call iot.import('iot','ntemps', concat(baskets,'1'));
select * from ntemps;

create procedure web00()
begin
    insert into atemps select min(iotclk), count(*), avg(temperature) from ntemps;
end;

call iot.query('iot','web00');
call iot.show('iot','web00');

call iot.stop();
--select * from  iot.baskets();
--select * from  iot.queries();

