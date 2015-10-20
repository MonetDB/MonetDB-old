# Test storing the _values() dictionary in the database using the pickle module
# We encode all the (key,value) pairs int he store_values() function and store it in a table
# Then unpickle and load the objects back into the _values() dictionary in the restore_values() function
# Useful for making the _values() dictionary persistent, after the server shuts down, as it is normally only stored in memory.
START TRANSACTION;

CREATE FUNCTION initialize_values() RETURNS TABLE(a BOOLEAN) LANGUAGE PYTHON
{
    _values['a'] = 3
    _values['b'] = "hello"
    _values[33] = [3, 37, "hello"]
    _values['c'] = numpy.array([44, 55])
    _values[22.77] = numpy.ma.masked_array([1, 2, 3], [0, 1, 0])
    return True
};

CREATE FUNCTION clear_values() RETURNS TABLE(a BOOLEAN) LANGUAGE PYTHON
{
    for key in _values.keys():
        del _values[key]
    return True
};

CREATE FUNCTION store_values() RETURNS TABLE(keys STRING, vals STRING) LANGUAGE PYTHON
{
    import cPickle
    result = dict()
    result['keys'] = [cPickle.dumps(x) for x in _values.keys()]
    result['vals'] = [cPickle.dumps(x) for x in _values.values()]
    return result
};

CREATE FUNCTION restore_values(keys STRING, vals STRING) RETURNS BOOLEAN LANGUAGE PYTHON
{
    import cPickle
    for key, value in zip([cPickle.loads(x) for x in keys], [cPickle.loads(x) for x in vals]):
        _values[key] = value
    return True
};

CREATE FUNCTION print_values() RETURNS TABLE(a BOOLEAN) LANGUAGE PYTHON
{
    print("Printing _values dictionary")
    print("Items: %d" % len(_values))
    for key,val in _values.iteritems():
        print str(key) + ',' + str(val)
    print("End\n")
    return True
};

SELECT * FROM initialize_values();
SELECT * FROM print_values();
CREATE TABLE values_storage AS SELECT * FROM store_values() WITH DATA;
SELECT * FROM clear_values();
SELECT * FROM print_values();
SELECT restore_values(keys, vals) FROM values_storage;
SELECT * FROM print_values();

DROP TABLE values_storage;

ROLLBACK;
