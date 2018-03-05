create function getrss() 
returns BIGINT external name status.rss_cursize;

create function getvm() 
returns BIGINT external name status.mem_cursize;

create table test(a int, b int, c double);

insert into test VALUES (1, 0, 1);

create procedure loop_insert(maximum_size bigint)
BEGIN
    declare size BIGINT;
    set size = (select count(*) from test);

    WHILE size < maximum_size DO
        insert into test (select a+1, b+2, rand()*c from test);

        set size = (select count(*) from test);
    END WHILE;
END;

call loop_insert(10000000);

-- It seems that it requires an analytical query to keep memory in RAM.
select count(*) as record_count from test;

select getrss() as resident_set_size, getvm() as virtual_memory_size;
