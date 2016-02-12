--start transaction;

DECLARE ds_deg, ds_rad, ds_rad_squared, isint2 DOUBLE;

--SET ds_deg = CAST(0.5 AS DOUBLE)/3600;
SET ds_deg = CAST(5 AS DOUBLE); /* units in degrees */
SET ds_rad = PI() * ds_deg / 180;
SET ds_rad_squared = ds_rad * ds_rad;
SET isint2 = 4 * SIN ( RADIANS (0.5 * ds_deg )) * SIN ( RADIANS (0.5 * ds_deg ));

SELECT ds_deg AS ds_deg
      ,3600 * ds_deg AS ds_arcsec
      ,ds_rad AS ds_rad
      ,ds_rad_squared AS "ds_rad_squared (input arg)"
      ,isint2 AS "4 sin^2 \theta"
;


drop table catalog;
drop table catalog2;
drop table catalog_union;
drop table sourcelist;

create table catalog(id int, ra double, decl double);
insert into catalog values (1, 222.3, 79.5 ), (2, 122.3, 88.5), (3, 22.3, 79.5 ), (4, 88.0, 38.0);


create table catalog2(id int, ra double, decl double);
insert into catalog2 values (5, 122.3, 89.5 ), (6, 32.3, 98.5), (7, 32.3, 89.5 ), (8, 98.0, 48.0);


create table sourcelist(id int, ra double, decl double);
insert into sourcelist values (11, 22.305, 79.499 ), (12,122.305, 88.499), (13, 222.305, 79.499 ), (14, 98.05, 47.99 );

-- select * from k3m_free();
-- select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog2 as s));
-- select *
--       ,sqrt(dist) as dist_rad
--       ,180*sqrt(dist)/pi() as dist_deg
--       ,3600*180*sqrt(dist)/pi() as dist_arcsec 
--   from k3m_query((select id, ra*PI()/180, decl*PI()/180, ds_rad_squared from sourcelist));


--select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog2 as s));
--select *,3600*180*sqrt(dist)/pi() as dist_arcsec from k3m_query((select id, ra*PI()/180, decl*PI()/180, 0.000304617419787 from sourcelist)) order by idc;

create table catalog_union as select id, ra, decl from catalog 
                              union all 
                              select id, ra, decl from catalog2 with data; 

-- select *
--       ,COS(RADIANS(decl)) * COS(RADIANS(ra)) as x
--       ,COS(RADIANS(decl)) * SIN(RADIANS(ra)) as y
--       ,SIN(RADIANS(decl)) as z
--   from catalog;
-- select *
--       ,COS(RADIANS(decl)) * COS(RADIANS(ra)) as x
--       ,COS(RADIANS(decl)) * SIN(RADIANS(ra)) as y
--       ,SIN(RADIANS(decl)) as z
--   from catalog2;
select *
      ,COS(RADIANS(decl)) * COS(RADIANS(ra)) as x
      ,COS(RADIANS(decl)) * SIN(RADIANS(ra)) as y
      ,SIN(RADIANS(decl)) as z
  from catalog_union;
select *
      ,COS(RADIANS(decl)) * COS(RADIANS(ra)) as x
      ,COS(RADIANS(decl)) * SIN(RADIANS(ra)) as y
      ,SIN(RADIANS(decl)) as z
from sourcelist;
-- we need the seq pipe for building
set optimizer='sequential_pipe';

select * from k3m_free();
explain select * from k3m_build((select id, pi()*ra/180, pi()*decl/180 from catalog_union as s));
select * from k3m_build((select id, pi()*ra/180, pi()*decl/180 from catalog_union as s));

set optimizer='default_pipe';

explain select *
      ,sqrt(dist) as dist_rad
      ,180*sqrt(dist)/pi() as dist_deg
      ,3600*180*sqrt(dist)/pi() as dist_arcsec 
  from k3m_query((select id, ra*PI()/180, decl*PI()/180, ds_rad_squared from sourcelist)) order by idc;

select *
      ,sqrt(dist) as dist_rad
      ,180*sqrt(dist)/pi() as dist_deg
      ,3600*180*sqrt(dist)/pi() as dist_arcsec 
  from k3m_query((select id, ra*PI()/180, decl*PI()/180, ds_rad_squared from sourcelist)) order by idc;

select cid
      ,sid
      ,power(dist_rad,2) as dist
      ,dist_rad
      ,degrees(dist_rad ) AS dist_deg
      ,3600*degrees(dist_rad ) AS dist_arcsec
  from (select c.id as cid
              ,s.id as sid
              ,2 * ASIN(SQRT( power( (COS(RADIANS(c.decl)) * COS(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * COS(RADIANS(s.ra))), 2)
                            + power( (COS(RADIANS(c.decl)) * SIN(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * SIN(RADIANS(s.ra))), 2)
                            + power( (SIN(RADIANS(c.decl)) - SIN(RADIANS(s.decl))), 2)
                            ) / 2) as dist_rad
          from catalog_union c
              ,sourcelist s 
         where   power( (COS(RADIANS(c.decl)) * COS(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * COS(RADIANS(s.ra))) , 2)
               + power( (COS(RADIANS(c.decl)) * SIN(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * SIN(RADIANS(s.ra))), 2)
               + power( (SIN(RADIANS(c.decl)) - SIN(RADIANS(s.decl))), 2) < isint2
       ) t order by cid
;

--ROLLBACK;
