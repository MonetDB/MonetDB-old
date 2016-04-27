set schema iot;
create stream table temps( iotclk timestamp, room string , temperature real);
create table atemps( iotclk timestamp, cnt int , temperature real);
-- remainder depends on location of the baskets root

declare baskets string;
set baskets= '/ufs/mk/baskets/measures/temperatures/';

call iot.basket('iot','temps', concat(baskets,'1'));
create procedure web00()
begin
    insert into atemps select min(iotclk), count(*), avg(temperature) from temps;
end;

call iot.query('iot','web00');

select * from  iot.baskets();
select * from  iot.queries();

