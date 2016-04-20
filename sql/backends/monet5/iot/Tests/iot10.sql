-- use accumulated aggregation
set schema iot;

create table tmp_aggregate(tmp_total decimal(8,2), tmp_count decimal(8,2));
insert into tmp_aggregate values(0.0,0.0);

create procedure collector()
begin
	update tmp_aggregate
		set tmp_total = tmp_total + (select sum(val) from iot.stmp),
			tmp_count = tmp_total + (select count(*) from iot.stmp);
	delete from iot.stmp;
end;

insert into stmp values('2005-09-23 12:34:26.736',1,12.34);
select * from stmp;

call iot.query('iot','collector');

select * from iot.baskets();
select * from iot.queries();

call iot.activate();
select * from tmp_aggregate;
