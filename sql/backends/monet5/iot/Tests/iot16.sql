-- introduce a heartbeat and windo directly on a stream
set schema iot;
set optimizer='iot_pipe';

create stream table winheart(cnt integer,clk1 integer);

call iot.window('iot','winheart',2);
call iot.heartbeat('iot','winheart',75);

select winsize, winstride, heartbeat from  iot.baskets();
select *from  iot.queries();

drop table winheart;

