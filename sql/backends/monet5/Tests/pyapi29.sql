# Parallel aggregates
# Aggregations created with PYTHON_MAP are computed in parallel over GROUP BY statements
# Meaning that manual looping over the 'aggr_group' parameter is not necessary
# instead, you can create a single PYTHON_MAP aggregate and it will be computed once per group in parallel

# (NOTE: Currently only works with INTEGER input columns, this will be fixed soon)


START TRANSACTION;

CREATE FUNCTION rvalfunc() RETURNS TABLE(groupcol INTEGER, datacol INTEGER) LANGUAGE PYTHON {
    return {'groupcol': [0,1,2,0,1,2,3], 
            'datacol' : [42,84,42,21,42,21,21] }  
};

CREATE AGGREGATE aggrmedian(val integer) RETURNS integer LANGUAGE PYTHON_MAP {
    return numpy.median(val)
};

SELECT groupcol,aggrmedian(datacol) FROM rvalfunc() GROUP BY groupcol;

DROP AGGREGATE aggrmedian;
DROP FUNCTION rvalfunc;

ROLLBACK;
