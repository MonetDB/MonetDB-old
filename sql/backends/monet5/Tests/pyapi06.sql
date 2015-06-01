START TRANSACTION;

CREATE TABLE rval(groupcol integer,datacol integer);
INSERT INTO rval VALUES (1,42), (1,84), (2,42), (2,21);

CREATE AGGREGATE aggrmedian(val integer) RETURNS integer LANGUAGE P {
	if 'aggr_group' in locals():
		unique = numpy.unique(aggr_group)
		x = numpy.zeros(shape=(unique.size))
		for i in range(0,unique.size):
			x[i] = numpy.median(val[aggr_group==unique[i]])
		return(x)
	else:
		return(numpy.median(val))
};

SELECT aggrmedian(datacol) FROM rval;
SELECT groupcol,aggrmedian(datacol) FROM rval GROUP BY groupcol;


DROP AGGREGATE aggrmedian;
DROP TABLE rval;

ROLLBACK;
