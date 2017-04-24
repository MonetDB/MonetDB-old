-- Avoid tuples to 'disappear' from stream tables.

set schema iot;
set optimizer='iot_pipe';

create stream table xyz (t timestamp, sensor integer, val decimal(8,2)) ;

call iot.keep('iot','xyz');
select winsize from  iot.baskets();
call iot.release('iot','xyz');
select winsize from  iot.baskets();

call iot.window('iot','xyz',3);
call iot.keep('iot','xyz');
select winsize from  iot.baskets();
call iot.release('iot','xyz');
select winsize from  iot.baskets();


select * from iot.errors();
drop table xyz;
