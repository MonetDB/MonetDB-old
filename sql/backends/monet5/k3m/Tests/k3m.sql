set optimizer='sequential_pipe';


start transaction;


create table catalog(id int, ra double, decl double);
insert into catalog values (1, 222.3, 79.5 ), (2, 122.3, 88.5), (3, 22.3, 79.5 ), (4, 88.0, 38.0);


create table catalog2(id int, ra double, decl double);
insert into catalog2 values (5, 122.3, 89.5 ), (6, 32.3, 98.5), (7, 32.3, 89.5 ), (8, 98.0, 48.0);


create table sourcelist(id int, ra double, decl double);
insert into sourcelist values (11, 22.305, 79.499 ), (12,122.305, 88.499), (13, 222.305, 79.499 );

select * from k3m_free();

select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog as s));
select * from k3m_query((select id, ra*PI()/180, decl*PI()/180, 0.01745329 from sourcelist));


select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog2 as s));
select * from k3m_query((select id, ra*PI()/180, decl*PI()/180, 0.01745329 from sourcelist)) order by idc;

select * from k3m_free();

create table catalog_union as select id, ra*PI()/180, decl*PI()/180 from catalog union all select id, ra*PI()/180, decl*PI()/180 from catalog2 with data; 

select * from k3m_build((select * from catalog_union as s));
select * from k3m_query((select id, ra*PI()/180, decl*PI()/180, 0.01745329 from sourcelist)) order by idc;
;


ROLLBACK;
