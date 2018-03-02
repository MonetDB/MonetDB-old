create function getrss() 
returns BIGINT external name prologue.getrss;

create function getvm() 
returns BIGINT external name prologue.getvm;

create table test(a int, b int, c double);

insert into test VALUES (1, 0, 1);

select getrss() as resident_set_size, getvm() as virtual_memory_size;

create procedure loop_insert(n bigint)
BEGIN
    declare run BIGINT;
    set run = (select count(*) from test);

    WHILE run < n DO
        insert into test (select a+1, b+2, rand()*c from test);

        set run = (select count(*) from test);
    END WHILE;
END;

call loop_insert(10000000);

select count(*) from test;

select getrss() as resident_set_size, getvm() as virtual_memory_size;
