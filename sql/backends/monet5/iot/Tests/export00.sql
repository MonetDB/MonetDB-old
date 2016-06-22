-- A simple continuous query.
set schema iot;
set optimizer='iot_pipe';

create stream table tempsout (t timestamp, sensor integer, val decimal(8,2)) ;

insert into tempsout values('2005-09-23 12:34:26.736',1,12.34);
insert into tempsout values('2005-09-23 12:34:26.736',1,12.34);
insert into tempsout values('2005-09-23 12:34:26.736',1,12.34);
select * from tempsout;

declare basketdir string;
set basketdir= '/ufs/mk/baskets/measures/temperatures/';

call iot.export('iot','tempsout', concat(basketdir,'20'));

drop table tempsout;
