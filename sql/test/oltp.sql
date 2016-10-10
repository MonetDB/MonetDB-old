call oltp_enable();
create table tmp_oltp(i integer);

insert into tmp_oltp values(1);
insert into tmp_oltp values(2),(3);
select username,lockid,cnt,query from oltp_locks();

call oltp_disable();
insert into tmp_oltp values(4);

call oltp_enable();
insert into tmp_oltp values(5);
select * from tmp_oltp;
drop table tmp_oltp;
select username,lockid,cnt,query from oltp_locks();

call oltp_reset();
select username,lockid,cnt,query from oltp_locks();
