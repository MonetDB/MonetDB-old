COPY 5 RECORDS INTO region from 'PWD/region.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 25 RECORDS INTO nation from 'PWD/nation.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 100 RECORDS INTO supplier from 'PWD/supplier.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 1500 RECORDS INTO customer from 'PWD/customer.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 2000 RECORDS INTO part from 'PWD/part.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 8000 RECORDS INTO partsupp from 'PWD/partsupp.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 15000 RECORDS INTO orders from 'PWD/orders.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
COPY 70000 RECORDS INTO lineitem from 'PWD/lineitem.tbl' ON CLIENT USING DELIMITERS '|', E'|\n' LOCKED;
