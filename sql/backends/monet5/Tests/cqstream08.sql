-- set scheduling properties of a stream relation
create stream table xyz (t timestamp, sensor integer, val decimal(8,2)) ;

call cquery.keep('iot','xyz');
select winsize from  cquery.baskets();
call cquery.deregister('iot','xyz');
select winsize from  cquery.baskets();

call cquery.window('iot','xyz',3);
call cquery.keep('iot','xyz');
select winsize from  cquery.baskets();
call cquery.deregister('iot','xyz');
select winsize from  cquery.baskets();


select * from cquery.errors();
drop table xyz;

