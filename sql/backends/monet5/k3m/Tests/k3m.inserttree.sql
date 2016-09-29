start transaction;

DECLARE ds_deg, ds_rad, ds_rad_squared, isint2 DOUBLE;

SET ds_deg = CAST(1 AS DOUBLE); /* units in degrees */
SET ds_rad = PI() * ds_deg / 180;
SET ds_rad_squared = ds_rad * ds_rad;
SET isint2 = 4 * SIN ( RADIANS (0.5 * ds_deg )) * SIN ( RADIANS (0.5 * ds_deg ));

SELECT ds_deg AS ds_deg
      ,3600 * ds_deg AS ds_arcsec
      ,ds_rad AS ds_rad
      ,ds_rad_squared AS "ds_rad_squared (input arg)"
      ,isint2 AS "4 sin^2 \theta"
;

create table catalog(id int, ra double, decl double);
insert into catalog values (1, 222.3, 79.5 ), (2, 122.3, 88.5), (3, 22.3, 79.5 ), (4, 88.0, 38.0);

create table sourcelist(id int, ra double, decl double);
insert into sourcelist values (11, 22.305, 79.499 ), (12,122.305, 88.499), (13, 222.305, 79.499 ), (14, 98.05, 47.99 );

-- we need the seq pipe for freeing and building
set optimizer='sequential_pipe';
select * from k3m_free();

select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog as s));

-- After tree has been built we reset to def pipe
set optimizer='default_pipe';

select * from catalog;
select * from sourcelist;

-- The counterparts, by k3m_query
select *
      ,sqrt(dist) as dist_rad
      ,180*sqrt(dist)/pi() as dist_deg
      ,3600*180*sqrt(dist)/pi() as dist_arcsec 
  from k3m_query((select id, ra*PI()/180, decl*PI()/180, ds_rad_squared from sourcelist)) 
order by idc;

-- The counterparts, by plain sql. Apart from rounding errors, the results should be identical
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
          from catalog c
              ,sourcelist s 
         where   power( (COS(RADIANS(c.decl)) * COS(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * COS(RADIANS(s.ra))) , 2)
               + power( (COS(RADIANS(c.decl)) * SIN(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * SIN(RADIANS(s.ra))), 2)
               + power( (SIN(RADIANS(c.decl)) - SIN(RADIANS(s.decl))), 2) < isint2
       ) t 
order by cid
;

-- Now we add new sources to the existing tree  
create table catalog2(id int, ra double, decl double);
insert into catalog2 values (5, 122.3, 89.5 ), (6, 32.3, 98.5), (7, 32.3, 89.5 ), (8, 98.0, 48.0);

set optimizer='sequential_pipe';
select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog2 as s));

-- After tree has been expanded we reset to def pipe
set optimizer='default_pipe';

-- The counterparts, by k3m_query
select *
      ,sqrt(dist) as dist_rad
      ,180*sqrt(dist)/pi() as dist_deg
      ,3600*180*sqrt(dist)/pi() as dist_arcsec 
  from k3m_query((select id, ra*PI()/180, decl*PI()/180, ds_rad_squared from sourcelist)) 
order by idc;

-- The counterparts, by plain sql. Apart from rounding errors, the results should be identical
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
          from catalog c
              ,sourcelist s 
         where   power( (COS(RADIANS(c.decl)) * COS(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * COS(RADIANS(s.ra))) , 2)
               + power( (COS(RADIANS(c.decl)) * SIN(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * SIN(RADIANS(s.ra))), 2)
               + power( (SIN(RADIANS(c.decl)) - SIN(RADIANS(s.decl))), 2) < isint2
        union all
        select c.id as cid
              ,s.id as sid
              ,2 * ASIN(SQRT( power( (COS(RADIANS(c.decl)) * COS(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * COS(RADIANS(s.ra))), 2)
                            + power( (COS(RADIANS(c.decl)) * SIN(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * SIN(RADIANS(s.ra))), 2)
                            + power( (SIN(RADIANS(c.decl)) - SIN(RADIANS(s.decl))), 2)
                            ) / 2) as dist_rad
          from catalog2 c
              ,sourcelist s 
         where   power( (COS(RADIANS(c.decl)) * COS(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * COS(RADIANS(s.ra))) , 2)
               + power( (COS(RADIANS(c.decl)) * SIN(RADIANS(c.ra)) - COS(RADIANS(s.decl)) * SIN(RADIANS(s.ra))), 2)
               + power( (SIN(RADIANS(c.decl)) - SIN(RADIANS(s.decl))), 2) < isint2
       ) t 
order by cid
;

create table catalog_union as select id, ra, decl from catalog 
                              union all 
                              select id, ra, decl from catalog2 with data; 

-- we need the seq pipe for freeing and building
set optimizer='sequential_pipe';
select * from k3m_free();

select * from k3m_build((select id, ra*PI()/180, decl*PI()/180 from catalog_union as s));

-- After tree has been built we reset to def pipe
set optimizer='default_pipe';

select * from catalog_union;
select * from sourcelist;

-- The counterparts, by k3m_query
select *
      ,sqrt(dist) as dist_rad
      ,180*sqrt(dist)/pi() as dist_deg
      ,3600*180*sqrt(dist)/pi() as dist_arcsec 
  from k3m_query((select id, ra*PI()/180, decl*PI()/180, ds_rad_squared from sourcelist)) 
order by idc;

-- The counterparts, by plain sql. Apart from rounding errors, the results should be identical
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
       ) t 
order by cid
;

ROLLBACK;
