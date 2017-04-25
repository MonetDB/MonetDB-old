-- Introduce firing conditions to the queries themselves.
create stream table tmp13 (t timestamp, sensor integer, val decimal(8,2)) ;
create table agenda13(i integer, msg string);

-- The window determines the input size, not necessareily the firing
-- which is overruled here by the hearbeat (order is important)
call sys.window('iot','tmp13',2);
call sys.heartbeat('iot','tmp13',1000);

create procedure cq_agenda()
begin
    declare b boolean;
    set b = (select count(*) > 0 from tmp13);
    if (b)
    then
        insert into agenda13 select count(*), 'full batch' from tmp13;
    end if;
end;

call cquery.new('sys','cq_agenda');

insert into tmp13 values('2005-09-23 12:34:26.736',1,12.34);
insert into tmp13 values('2005-09-23 12:34:26.736',1,12.35);
insert into tmp13 values('2005-09-23 12:34:26.736',1,12.36);

-- reactivate all continuous queries
call cquery.resume();
-- wait a few seconds
call cquery.wait(5000);

select 'RESULT';
select * from agenda13;

call cquery.stop();

select * from cquery.summary();

call cquery.release('sys','cq_agenda');
drop procedure cq_agenda;
drop table tmp13;
drop table agenda13;

