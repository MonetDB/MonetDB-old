

def consts_to_string(consts):
    result = ""
    for const in consts:
        if not str(type(const).__name__) == "code":
            result = result + '(' + str(type(const).__name__) + ':' + str(const) + ')'
        else:
            result = result + '(code:{@' + code_object_to_string(const) + '})'
    return result

def names_to_string(names):
    result = ""
    for name in names:
        result = result + name + ','
    return result

def format_code(code):
    result = "";
    for i in code:
        if ord(i) < 16: result = result + '\\\\x0' + hex(ord(i))[2:]
        else: result = result + '\\\\x' + hex(ord(i))[2:]
    return result

def code_object_to_string(codeobject):
    args = codeobject.co_argcount
    nlocals = codeobject.co_nlocals
    stacksize = codeobject.co_stacksize
    flags = codeobject.co_flags
    code = format_code(codeobject.co_code)
    consts = codeobject.co_consts
    names = codeobject.co_names
    varnames = codeobject.co_varnames
    freevars = codeobject.co_freevars
    cellvars = codeobject.co_cellvars
    filename = codeobject.co_filename
    name = codeobject.co_name
    firstlineno = codeobject.co_firstlineno
    lnotab = format_code(codeobject.co_lnotab)
    return str(args) + '@' + str(nlocals) + '@' + str(stacksize) + '@' + str(flags) + '@' + code + '@' + consts_to_string(consts) + '@' + names_to_string(names) + '@' + names_to_string(varnames) + '@' + names_to_string(freevars) + '@' + names_to_string(cellvars) + '@' + filename + '@' + name + '@' + str(firstlineno) + '@' + lnotab + '@'

def function_to_string(fun):
    return code_object_to_string(fun.__code__)

def export_function(function, argtypes, returns, new_name=None, multithreading=False, table=False, test=True):
    name = function.__code__.co_name if new_name == None else new_name;
    argc = function.__code__.co_argcount
    argnames = function.__code__.co_varnames
    language = " language PYTHON_MAP" if multithreading else " language P"
    if len(argtypes) != argc:
        raise Exception('The length of the argument types does not match the amount of function arguments.')
    argstr = ""
    for i in range(0, argc): argstr = argstr + argnames[i] + ' ' + argtypes[i] + ','
    retstr = returns[0]
    if len(returns) > 1 or table:
        retstr = "table (";
        for i in returns: retstr = retstr + i + ','
        retstr= retstr[:len(retstr)-1] + ")";
    export = 'CREATE FUNCTION ' + name + '(' + argstr[:len(argstr)-1] + ') returns ' + retstr + language + ('{@' if test else '{NOTEST@') + function_to_string(function) + '};'
    for ch in export:
        if ord(ch) == 0:
            raise Exception("Zero byte!");
    return(export)

import math
import multiprocessing
import numpy
import platform
import os
import sys
python_version = str(sys.version_info.major) + '.' + str(sys.version_info.minor) + '.' + str(sys.version_info.micro)
numpy_version = numpy.version.version
amount_of_cores = str(multiprocessing.cpu_count())
main_memory = str(-1)
try: main_memory = str(int(os.popen("cat /proc/meminfo | grep MemTotal | awk '{ print $2 }'").read().translate(None, ' \n\t')) / 1000 ** 2)
except: pass
os_name = ' '.join(platform.dist())

def format_headers(*measurement_axes):
    result = 'Python Ver\tNumpy Ver\tCPU Cores\tMain Memory (GB)\tOS'
    for measurement in measurement_axes:
        result = result + '\t' + str(measurement)
    return result + '\n'

def format_output(*measurements):
    result = python_version + '\t' + numpy_version + '\t' + amount_of_cores + '\t' + main_memory + '\t' + os_name
    for measurement in measurements:
        result = result + '\t' + str(measurement)
    return result + '\n'

import os
import sys
import time

c_compiler = "gcc"

# The arguments are
# [1] => Database to test on ("MonetDB", "Postgres")
# [2] => Type of test ['INPUT', 'OUTPUT']
# [3] => Output file name
# [4] => Number of tests for each value
# [5] => Mapi Port
# [6+] => List of input values
arguments = sys.argv
if (len(arguments) <= 6):
    print("Too few arguments provided.")
    quit()

args_input_database = arguments[1]
args_test_type = arguments[2]
args_output_file = arguments[3]
args_ntests = arguments[4]
args_port = arguments[5]
parameters_start = 6

output_file = os.path.join(os.getcwd(), args_output_file)
temp_file = os.path.join(os.getcwd(), 'temp_output.tsv')
test_count = int(args_ntests)
port = int(args_port)
max_retries = 15
max_size = 1000
random_seed = 33

hot_test = True
split = args_test_type.split(':')
if len(split) > 1 and split[1].lower() == 'cold':
    hot_test = False
args_test_type = split[0]

drop_cache = os.environ["DROP_CACHE_COMMAND"]
def drop_all_caches():
    array = numpy.zeros(100000)
    numpy.savetxt("temp_file.csv", array, delimiter=",")
    barray = numpy.loadtxt("temp_file.csv", delimiter=",")
    os.remove("temp_file.csv")
    del array
    del barray
    os.system(drop_cache)



if str(args_input_database).lower() == "monetdb":
    import monetdb.sql
    # Try to connect to the database
    # We try a couple of times because starting up the database takes some time, so it might fail the first few times
    for i in range(0, max_retries):
        try:
            connection = monetdb.sql.connect(username="monetdb", password="monetdb", hostname="localhost",port=port,database="demo")
            break
        except:
            time.sleep(1)
        connection = None

    if connection is None:
        print("Failed to connect to MonetDB Server (mserver5) in " + str(max_retries) + " attempts.")
        sys.exit(1)
    cursor = connection.cursor()

    def run_test(testcommand, testcommand_nomem, *measurements):
        if hot_test: cursor.execute(testcommand_nomem) #run the test once to warm up
        for i in range(0,test_count):
            total_time, memory, pyapi_time = 0, 0, 0
            if testcommand != None:
                # clear the result file
                result_file = open(temp_file, 'w+')
                result_file.close()
                # run the command
                cursor.execute(testcommand)
                # now read the memory value from the file
                result_file = open(temp_file, 'r')
                pyapi_results = result_file.readline().translate(None, '\n').split('\t')
                result_file.close()
                memory = float(pyapi_results[0]) / 1000**2
            # now run the normal test (with malloc tracking disabled)
            result_file = open(temp_file, 'w+')
            result_file.close()
            if not hot_test: drop_all_caches() #drop caches everytime for cold tests
            start = time.time()
            cursor.execute(testcommand_nomem);
            cursor.fetchall();
            end = time.time()
            result_file = open(temp_file, 'r')
            pyapi_results = result_file.readline().translate(None, '\n').split('\t')
            result_file.close()
            total_time = end - start
            pyapi_time = pyapi_results[1]
            f.write(format_output(*(measurements + (total_time, memory, pyapi_time))))
            f.flush()

    if str(args_test_type).lower() == "input" or str(args_test_type).lower() == "input-map" or str(args_test_type).lower() == "input-null":
        # Input testing
        multithreading_test = str(args_test_type).lower() == "input-map"
        # First create a function that generates the desired input size (in MB) and pass it to the database
        if str(args_test_type).lower() == "input-null":
            #if the type is input-null, we simply set all negative numbers to NULL
            def generate_integers(mb, random_seed):
                import random
                import math
                numpy.random.seed(random_seed)
                byte_size = mb * 1000 * 1000
                integer_size_byte = 4
                max_int = math.pow(2,31) - 1
                min_int = -max_int
                integer_count = int(byte_size / integer_size_byte)
                integers = numpy.random.random_integers(min_int, max_int, integer_count).astype(numpy.int32)
                return numpy.ma.masked_array(integers, numpy.less(integers, 0))
        else:
            def generate_integers(mb, random_seed):
                import random
                import math
                numpy.random.seed(random_seed)
                byte_size = mb * 1000 * 1000
                integer_size_byte = 4
                max_int = math.pow(2,31) - 1
                min_int = -max_int
                integer_count = int(byte_size / integer_size_byte)
                return numpy.random.random_integers(min_int, max_int, integer_count).astype(numpy.int32)

        cursor.execute(export_function(generate_integers, ['float', 'integer'], ['i integer'], table=True, test=False))

        # Our import test function returns a single boolean value and doesn't do anything with the actual input
        # This way the input loading is the only relevant factor in running time, because the time taken for function execution/output handling is constant
        def import_test(inp):
            return(True)

        cursor.execute(export_function(import_test, ['integer'], ['boolean'], multithreading=str(args_test_type).lower() == "input-map"))
        cursor.execute(export_function(import_test, ['integer'], ['boolean'], new_name="import_test_nomem", test=False, multithreading=multithreading_test))

        f = open(output_file + '.tsv', "w+")
        if not multithreading_test:
            f.write(format_headers('[AXIS]:Data Size (MB)', '[MEASUREMENT]:Total Time (s)', '[MEASUREMENT]:PyAPI Memory (MB)', '[MEASUREMENT]:PyAPI Time (s)'))
        else:
            f.write(format_headers('[AXIS]:Data Size (MB)', '[MEASUREMENT]:Total Time (s)', '[MEASUREMENT]:PyAPI Time (s)'))
        mb = []
        for i in range(parameters_start, len(arguments)):
            mb.append(float(arguments[i]))

        for size in mb:
            cursor.execute('CREATE TABLE integers (i integer);')
            temp_size = size
            for increment in range(0, int(math.ceil(float(size) / float(max_size)))):
                current_size = temp_size if temp_size < max_size else max_size
                cursor.execute('INSERT INTO integers SELECT * FROM generate_integers(' + str(current_size) + ',' + str(random_seed + increment) + ');')
                temp_size -= max_size

            if (str(args_test_type).lower() == "input"):
                run_test('select import_test(i) from integers;', 'select import_test_nomem(i) from integers;', size)
            else:
                # for input-map we need to do some special analysis of the PyAPI output
                # this is because every thread writes memory usage and execution time to the temp_file
                # rather than just having one entry for per query
                # so we have to analyse the result file for every query we perform
                results = [[], [], []]

                if hot_test:
                    for i in range(0, 2):
                        cursor.execute('select import_test(i) from integers;');
                        cursor.fetchall();
                for i in range(0,test_count):
                    # clear the result file
                    result_file = open(temp_file, 'w+')
                    result_file.write("")
                    result_file.close();
                    if not hot_test: drop_all_caches() #drop caches everytime for cold tests
                    # execute the query, measure the total time
                    start = time.time()
                    cursor.execute('select import_test(i) from integers;');
                    cursor.fetchall();
                    end = time.time()
                    list.append(results[0], end - start)
                    # now we need to analyze the result file
                    # we use the highest of all the execution times of the threads (max)
                    # we ignore memory usage, because we're not measuring it correctly for mapped stuff
                    memory_usage = 0
                    peak_execution_time = 0
                    with open(temp_file, 'r') as result_file:
                        for line in result_file:
                            pyapi_results = line.translate(None, '\n').split('\t')
                            memory_usage = memory_usage + float(pyapi_results[0]) / 1000 ** 2
                            if float(pyapi_results[1]) > peak_execution_time: peak_execution_time = float(pyapi_results[1])
                    list.append(results[1], memory_usage)
                    list.append(results[2], peak_execution_time)
                for i in range(0, len(results[0])):
                    f.write(format_output(size, results[0][i], results[2][i]))
                    f.flush()
            cursor.execute('drop table integers;')
        f.close()
        cursor.execute('rollback')
    elif str(args_test_type).lower() == "output":
        # output testing

        # we use a single scalar as input (the amount of MB to generate) so the input handling is fast
        # we do some computation (namely creating the output array) but that should only be a single malloc call, and should be negligible compared to the copying
        # that malloc call is also the same for both zero copy and copy, so it shouldn't make any difference in the comparison
        def generate_output(mb):
            byte_size = mb * 1000 * 1000
            integer_size_byte = 4
            integer_count = int(byte_size / integer_size_byte)
            integers = numpy.zeros(integer_count, dtype=numpy.int32)
            return integers

        cursor.execute(export_function(generate_output, ['float'], ['i integer'], table=True))
        cursor.execute(export_function(generate_output, ['float'], ['i integer'], new_name="generate_output2", table=True, test=False))

        f = open(output_file + '.tsv', "w+")
        f.write(format_headers('[AXIS]:Data Size (MB)', '[MEASUREMENT]:Total Time (s)', '[MEASUREMENT]:PyAPI Memory (MB)', '[MEASUREMENT]:PyAPI Time (s)'))
        mb = []
        for i in range(parameters_start, len(arguments)):
            mb.append(float(arguments[i]))

        for size in mb:
            run_test("select count(*) from generate_output(" + str(size) + ");", "select count(*) from generate_output2(" + str(size) + ");", size)
        f.close()

        #cursor.execute('drop function generate_output');
        cursor.execute('rollback')

    elif str(args_test_type).lower() == "string_samelength" or str(args_test_type).lower() == "string_extremeunicode":
        benchmark_dir = os.environ["PYAPI_BENCHMARKS_DIR"]
        os.system("%s " % c_compiler + benchmark_dir + "/randomstrings.c -o randomstrings")
        result_path = os.path.join(os.getcwd(), 'result.txt')

        if str(args_test_type).lower() == "string_samelength":
            def generate_strings_samelength(length):
                return 'A' * length
            cursor.execute(export_function(generate_strings_samelength, ['integer'], ['i string'], table=True, test=False))
        else:
            def generate_strings_samelength(length):
                return unichr(0x100) * length
            cursor.execute(export_function(generate_strings_samelength, ['integer'], ['i string'], table=True, test=False))

        mb = []
        lens = []
        for i in range(parameters_start, len(arguments)):
            tple = arguments[i].translate(None, '()').split(',')
            mb.append(float(tple[0]))
            lens.append(int(tple[1]))

        def import_test(inp):
            return(True)

        cursor.execute(export_function(import_test, ['string'], ['boolean']))
        cursor.execute(export_function(import_test, ['string'], ['boolean'], new_name='import_test2', test=False))

        f = open(output_file + '.tsv', "w+")
        f.write(format_headers('[AXIS]:Data Size (MB)', '[AXIS]:String Length (Characters)', '[MEASUREMENT]:Total Time (s)', '[MEASUREMENT]:PyAPI Memory (MB)', '[MEASUREMENT]:PyAPI Time (s)'))
        for j in range(0,len(mb)):
            size = mb[j]
            length = lens[j]
            os.system("%s %s %s %s" % ("./randomstrings", str(size), str(length), result_path))
            cursor.execute('CREATE TABLE strings(i string);')
            cursor.execute("COPY INTO strings FROM '%s';" % result_path)
            cursor.execute('INSERT INTO strings SELECT * FROM generate_strings_samelength(' + str(length) + ');')
            run_test('select import_test(i) from strings;', 'select import_test2(i) from strings;', size, length)
            cursor.execute('drop table strings;')
        f.close()

        #cursor.execute('drop function generate_strings_samelength');
        #cursor.execute('drop function import_test');
        cursor.execute('rollback')
    elif str(args_test_type).lower() == "string_extremelength":
        benchmark_dir = os.environ["PYAPI_BENCHMARKS_DIR"]
        os.system("%s " % c_compiler + benchmark_dir + "/randomstrings.c -o randomstrings")
        result_path = os.path.join(os.getcwd(), 'result.txt')

        def generate_strings_extreme(extreme_length):
            def random_string(length):
                import random
                import string
                result = ""
                for i in range(0, length):
                    result += random.choice(string.printable)
                return result
            return random_string(extreme_length)
        cursor.execute(export_function(generate_strings_extreme, ['integer'], ['i string'], table=True, test=False))

        extreme_lengths = []
        string_counts = []
        for i in range(parameters_start, len(arguments)):
            tple = arguments[i].translate(None, '()').split(',')
            extreme_lengths.append(float(tple[0]))
            string_counts.append(int(tple[1]))

        def import_test(inp):
            return(True)

        cursor.execute(export_function(import_test, ['string'], ['boolean']))
        cursor.execute(export_function(import_test, ['string'], ['boolean'], new_name='import_test2', test=False))

        f = open(output_file + '.tsv', "w+")
        f.write(format_headers('[AXIS]:(Strings)', '[AXIS]:Extreme Length (Characters)', '[MEASUREMENT]:Total Time (s)', '[MEASUREMENT]:PyAPI Memory (MB)', '[MEASUREMENT]:PyAPI Time (s)'))
        for j in range(0,len(extreme_lengths)):
            str_len = extreme_lengths[j]
            str_count = string_counts[j]
            string_mb = float(str_count) / (1000 ** 2)
            os.system("%s %s %s %s" % ("./randomstrings", str(string_mb), str(1), result_path))
            cursor.execute('CREATE TABLE strings(i string);')
            cursor.execute("COPY INTO strings FROM '%s';" % result_path)
            cursor.execute('INSERT INTO strings SELECT * FROM generate_strings_extreme(' + str(str_len) + ');')
            run_test('select import_test(i) from strings;', 'select import_test2(i) from strings;', str_count, str_len)
            cursor.execute('DROP TABLE strings;')
        f.close()

        #cursor.execute('drop function generate_strings_extreme');
        #cursor.execute('drop function import_test');
        cursor.execute('rollback')
    elif "factorial" in str(args_test_type).lower():
        def generate_integers(mb, random_seed):
            import random
            import math
            numpy.random.seed(random_seed)
            byte_size = mb * 1000 * 1000
            integer_size_byte = 4
            max_int = 2000
            min_int = 1000
            integer_count = int(byte_size / integer_size_byte)
            return numpy.random.random_integers(min_int, max_int, integer_count).astype(numpy.int32)

        cursor.execute(export_function(generate_integers, ['float', 'integer'], ['i integer'], table=True, test=False))
        def factorial(i):
            import math
            result = numpy.zeros(i.shape[0])
            for a in range(0, i.shape[0]):
                result[a] = math.log(math.factorial(i[a]))
            return result

        cursor.execute(export_function(factorial, ['float'], ['double'], multithreading=True))
        if os.path.isfile(output_file + '.tsv'):
            f = open(output_file + '.tsv', "a")
        else:
            f = open(output_file + '.tsv', "w+")
            f.write(format_headers('[AXIS]:Cores (#)', '[AXIS]:Data Size (MB)', '[MEASUREMENT]:Total Time (s)', '[MEASUREMENT]:PyAPI Memory (MB)', '[MEASUREMENT]:PyAPI Time (s)'))
        cores = int(str(args_test_type).split('-')[1])
        mb = []
        for i in range(parameters_start, len(arguments)):
            mb.append(float(arguments[i]))

        for size in mb:
            cursor.execute('CREATE TABLE integers (i integer);')
            temp_size = size
            for increment in range(0, int(math.ceil(float(size) / float(max_size)))):
                current_size = temp_size if temp_size < max_size else max_size
                cursor.execute('INSERT INTO integers SELECT * FROM generate_integers(' + str(current_size) + ',' + str(random_seed + increment) + ');')
                temp_size -= max_size

            results = [[], [], []]
            if hot_test:
                for i in range(0, 2):
                    cursor.execute('select min(factorial(i)) from integers;');
                    cursor.fetchall();
            for i in range(0,test_count):
                result_file = open(temp_file, 'w+')
                result_file.write("")
                result_file.close();
                if not hot_test: drop_all_caches() #drop caches everytime for cold tests
                start = time.time()
                cursor.execute('select min(factorial(i)) from integers;');
                cursor.fetchall();
                end = time.time()
                list.append(results[0], end - start)
                memory_usage = 0
                peak_execution_time = 0
                with open(temp_file, 'r') as result_file:
                    for line in result_file:
                        pyapi_results = line.translate(None, '\n').split('\t')
                        memory_usage = memory_usage + float(pyapi_results[0]) / 1000 ** 2
                        if float(pyapi_results[1]) > peak_execution_time: peak_execution_time = float(pyapi_results[1])
                list.append(results[1], memory_usage)
                list.append(results[2], peak_execution_time)
            for i in range(0, len(results[0])):
                f.write(format_output(cores, size, results[0][i], results[1][i], results[2][i]))
                f.flush()

            cursor.execute('drop table integers;')
        f.close()


    elif str(args_test_type).lower() == "pquantile" or str(args_test_type).lower() == "rquantile" or str(args_test_type).lower() == "quantile":
        quantile_function = "quantile"
        if str(args_test_type).lower() == "pquantile":
            def pyquantile(i, j):
                return numpy.percentile(i, j * 100)

            cursor.execute(export_function(pyquantile, ['double', 'double'], ['double'], test=False))
            quantile_function = "pyquantile"
        elif str(args_test_type).lower() == "rquantile":
            cursor.execute("CREATE FUNCTION rquantile(v double, q double) RETURNS double LANGUAGE R { quantile(v,q) };");
            quantile_function = "rquantile"

        def generate_integers(mb, random_seed):
            import random
            import math
            numpy.random.seed(random_seed)
            byte_size = mb * 1000 * 1000
            integer_size_byte = 4
            max_int = math.pow(2,31) - 1
            min_int = -max_int
            integer_count = int(byte_size / integer_size_byte)
            return numpy.random.random_integers(min_int, max_int, integer_count).astype(numpy.int32)

        cursor.execute(export_function(generate_integers, ['float', 'integer'], ['i integer'], table=True, test=False))

        f = open(output_file + '.tsv', "w+")
        f.write(format_headers('[AXIS]:Data Size (MB)', '[MEASUREMENT]:Total Time (s)'))
        mb = []
        for i in range(parameters_start, len(arguments)):
            mb.append(float(arguments[i]))

        for size in mb:
            cursor.execute('CREATE TABLE integers (i integer);')
            temp_size = size
            for increment in range(0, int(math.ceil(float(size) / float(max_size)))):
                current_size = temp_size if temp_size < max_size else max_size
                cursor.execute('INSERT INTO integers SELECT * FROM generate_integers(' + str(current_size) + ',' + str(random_seed + increment) + ');')
                temp_size -= max_size

            results = []
            if hot_test:
                for i in range(0, 2):
                    cursor.execute('select ' + quantile_function + '(i, 0.5) from integers;') #run the test once to warm up
                    cursor.fetchall()
            for i in range(0,test_count):
                if not hot_test: drop_all_caches() #drop caches everytime for cold tests
                start = time.time()
                cursor.execute('select ' + quantile_function + '(i, 0.5) from integers;');
                cursor.fetchall();
                end = time.time()
                list.append(results, end - start)
            for result in results:
                f.write(format_output(size, result))
                f.flush()
            cursor.execute('drop table integers;')
    else:
        print("Unrecognized test type \"" + args_test_type + "\", exiting...")
        sys.exit(1)

else:
    input_file = os.environ["POSTGRES_INPUT_FILE"]
    input_dir = os.environ["POSTGRES_CWD"]
    def execute_test(input_type, database_init, database_load, database_execute, database_clear, database_final):
        database_init()

        f = open(output_file + '.tsv', "w+")
        f.write(format_headers('[AXIS]:Data Size (MB)', '[MEASUREMENT]:Total Time (s)'))

        mb = []
        for i in range(parameters_start, len(arguments)):
            mb.append(float(arguments[i]))

        for size in mb:
            os.system("%s " % c_compiler + input_dir + "/randomstrings.c -o " + input_dir + "/randomstrings")
            os.system("%s %s %s %s" % (input_dir + "/randomstrings", size, input_type, input_file))

            database_load()
            if hot_test:
                for i in range(0, 2):
                    database_execute()
            for i in range(0,test_count):
                if not hot_test: drop_all_caches() #drop caches everytime for cold tests
                start = time.time()
                database_execute()
                end = time.time()
                f.write(format_output(size, end - start))
            database_clear()
        f.close()
        database_final

    input_type = "integer"
    function_name = str(args_test_type).lower()
    function = None
    if function_name == "identity":
        def identity(a):
            return numpy.min(a)
        function = identity
        return_value = "integer"
    if function_name == "sqroot":
        def sqroot(a):
            return numpy.min(numpy.sqrt(numpy.abs(a)))
        function = sqroot
        return_value = "double precision"

    import inspect
    inspect_result = inspect.getsourcelines(function)
    source_code = "".join(["  " + x.lstrip() for x in inspect_result[0][1:]])

    if str(args_input_database).lower() == "postgres":
        client = os.environ["POSTGRES_CLIENT_COMMAND"]
        dropdb = os.environ["POSTGRES_DROPDB_COMMAND"]
        initdb = os.environ["POSTGRES_CREATEDB_COMMAND"]

        function_name = str(args_test_type).lower()
        def postgres_init():
            input_dir = os.environ["POSTGRES_CWD"]

            createdb_file = "%s/%s.createdb.sql" % (input_dir, function_name)
            run_file = "%s/%s.sql" % (input_dir, function_name)

            if function_name == "identity":
                source = "  return a"
            if function_name == "sqroot":
                source = "  import math\n  return math.sqrt(abs(a))"

            createdb_sql = """
CREATE TABLE integers(i integer);

COPY integers FROM '%s' DELIMITER ',' CSV;

CREATE FUNCTION %s(a integer)
  RETURNS %s
AS $$
%s
$$ LANGUAGE plpythonu;""" % (input_file, function_name, return_value, source)

            run_sql = """
            SELECT MIN(%s(i)) FROM integers;
            """ % function_name

            createdb = open(createdb_file, 'w+')
            createdb.write(createdb_sql)
            createdb.close()

            run = open(run_file, 'w+')
            run.write(run_sql)
            run.close()

        def postgres_load():
            os.system(initdb)
            os.system("%s -f %s/%s.createdb.sql > /dev/null" % (client, input_dir, function_name))

        def postgres_execute():
            os.system("%s -f %s/%s.sql" % (client, input_dir, function_name))

        def postgres_clear():
            os.system(dropdb)

        def postgres_final():
            os.remove(createdb_file)
            os.remove(run_file)
            os.remove(input_file)

        execute_test(input_type, postgres_init, postgres_load, postgres_execute, postgres_clear, postgres_final)
    elif str(args_input_database).lower() == "sqlitemem" or str(args_input_database).lower() == "sqlitedb":
        import csv, sqlite3
        database_file = os.environ["SQLITE_DB_FILE"]
        database_name = ":memory:" if str(args_input_database).lower() == "sqlitemem" else database_file

        conn = sqlite3.connect(database_file)
        c = conn.cursor()

        def sqlite_init():
            return None

        def sqlite_load():
            c.execute("CREATE TABLE integers(i int);")
            inp = open(input_file, 'r')
            result = [(int(x.strip('\n')),) for x in inp]
            inp.close()
            c.executemany('INSERT INTO integers VALUES (?);', result)

        def sqlite_execute():
            cursor = c.execute('SELECT * FROM integers')
            result = cursor.fetchall()
            function(numpy.array(result, dtype=numpy.int32))

        def sqlite_clear():
            c.execute("DROP TABLE integers")
            conn.commit();

        def sqlite_final():
            if str(args_input_database).lower() != "sqlitemem":
                os.remove(database_file)
            os.remove(input_file)
            conn.close()

        execute_test(input_type, sqlite_init, sqlite_load, sqlite_execute, sqlite_clear, sqlite_final)
    elif str(args_input_database).lower() == "monetdbmapi" or str(args_input_database).lower() == "pyapi" or str(args_input_database).lower() == "pyapimap" or str(args_input_database).lower() == "rapi":
        import monetdb.sql
        for i in range(0, max_retries):
            try:
                conn = monetdb.sql.connect(username="monetdb", password="monetdb", hostname="localhost",port=port,database="demo")
                break
            except:
                time.sleep(1)
            conn = None

        if conn is None:
            print("Failed to connect to MonetDB Server (mserver5) in " + str(max_retries) + " attempts.")
            sys.exit(1)
        c = conn.cursor()

        def monetdb_init():
            return None

        def monetdb_load():
            c.execute("CREATE TABLE integers(i int);")
            c.execute("COPY INTO integers FROM '%s';" % input_file)
            if str(args_input_database).lower() == "pyapi" or str(args_input_database).lower() == "pyapimap":
                func_language = "PYTHON" if str(args_input_database).lower() == "pyapi" else "PYTHON_MAP"
                c.execute("CREATE FUNCTION FUNC_%s(a integer) RETURNS %s LANGUAGE %s {%s};" % (function_name,return_value,func_language, source_code))
            elif str(args_input_database).lower() == "rapi":
                if function_name == "identity":
                    c.execute("CREATE FUNCTION FUNC_%s(a integer) RETURNS %s LANGUAGE R { min(a) };" % (function_name,return_value))
                if function_name == "sqroot":
                    c.execute("CREATE FUNCTION FUNC_%s(a integer) RETURNS %s LANGUAGE R { min(sqrt(abs(a))) };" % (function_name,return_value))

        if str(args_input_database).lower() == "pyapi" or str(args_input_database).lower() == "pyapimap" or str(args_input_database).lower() == "rapi":
            def monetdb_execute():
                c.execute("SELECT FUNC_%s(i) FROM integers;" % function_name)
                result = c.fetchall()
        else:
            def monetdb_execute():
                c.execute('SELECT * FROM integers')
                result = c.fetchall()
                function(numpy.array(result, dtype=numpy.int32))

        def monetdb_clear():
            c.execute("DROP TABLE integers;")
            if str(args_input_database).lower() == "pyapi" or str(args_input_database).lower() == "pyapimap" or str(args_input_database).lower() == "rapi":
                c.execute("DROP FUNCTION FUNC_%s;" % function_name)

        def monetdb_final():
            os.remove(input_file)
            conn.close()

        execute_test(input_type, monetdb_init, monetdb_load, monetdb_execute, monetdb_clear, monetdb_final)
    elif str(args_input_database).lower() == "psycopg2":
        dbname = os.environ["POSTGRES_DB_NAME"]
        initdb = os.environ["POSTGRES_CREATEDB_COMMAND"]
        dropdb = os.environ["POSTGRES_DROPDB_COMMAND"]
        os.system(initdb)
        import psycopg2
        conn = psycopg2.connect("dbname=%s host=/tmp/" % dbname)
        c = conn.cursor()
        def psycopg2_init():
            return None

        def psycopg2_load():
            c.execute("CREATE TABLE integers(i integer);")
            c.execute("COPY integers FROM '%s' DELIMITER ',' CSV;" % input_file)


        def psycopg2_execute():
            c.execute("SELECT * FROM integers;")
            result = c.fetchall()
            print function(numpy.array(result, dtype=numpy.int32))

        def psycopg2_clear():
            c.execute("DROP TABLE integers;")

        def psycopg2_final():
            os.system(dropdb)
            conn.close()
            os.remove(input_file)

        execute_test(input_type, psycopg2_init, psycopg2_load, psycopg2_execute, psycopg2_clear, psycopg2_final)
    elif str(args_input_database).lower() == "pytables":
        import tables
        import csv

        table_file = 'testfile.h5'

        description = dict()
        description['i'] = tables.Int32Col()

        def pytables_init():
            return None

        def pytables_load():
            file = tables.open_file(table_file, mode='w', title='test file')
            group = file.create_group('/', 'integers', 'integer_data')
            table = file.create_table(group, 'values', description, "example")
            values = table.row
            with open(input_file, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                result = [x for x in reader]
                for x in result:
                    values['i'] = int(x[0])
                    values.append()
            table.flush()
            file.close()

        def pytables_execute():
            file = tables.open_file(table_file, mode='r')
            table = file.root.integers.values
            result = [x['i'] for x in table.iterrows()]
            function(numpy.array(result, dtype=numpy.int32))

        def pytables_clear():
            os.remove('testfile.h5')

        def pytables_final():
            os.remove(input_file)

        execute_test(input_type, pytables_init, pytables_load, pytables_execute, pytables_clear, pytables_final)
    elif str(args_input_database).lower() == "csv":
        import csv
        def csv_init():
            return None

        def csv_load():
            return None

        def csv_execute():
            with open(input_file, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                result = [int(x[0]) for x in reader]
                function(numpy.array(result, dtype=numpy.int32))

        def csv_clear():
            return None

        def csv_final():
            os.remove(input_file)

        execute_test(input_type, csv_init, csv_load, csv_execute, csv_clear, csv_final)
    elif str(args_input_database).lower() == "numpybinary":
        import csv, numpy
        numpy_binary = 'tempfile.npy'
        def numpy_init():
            return None

        def numpy_load():
            with open(input_file, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                result = [int(x[0]) for x in reader]
                numpy_array = numpy.array(result, dtype=numpy.int32)
                numpy.save(numpy_binary, numpy_array)

        def numpy_execute():
            numpy_array = numpy.load(numpy_binary)
            function(numpy_array)

        def numpy_clear():
            return None

        def numpy_final():
            os.remove(input_file)
            os.remove(numpy_binary)

        execute_test(input_type, numpy_init, numpy_load, numpy_execute, numpy_clear, numpy_final)
    elif str(args_input_database).lower() == "castra":
        import castra, csv, shutil, pandas as pd
        castra_binary = 'data.castra'
        def castra_init():
            return None

        def castra_load():
            with open(input_file, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                result = [int(x[0]) for x in reader]
                numpy_array = numpy.array(result, dtype=numpy.int32)
                A = pd.DataFrame({'i': numpy_array})
                c = castra.Castra(castra_binary, template=A)
                c.extend(A)

        def castra_execute():
            c = castra.Castra(castra_binary, readonly=True)
            print function(c[:, 'i'].values)

        def castra_clear():
            shutil.rmtree(castra_binary)

        def castra_final():
            os.remove(input_file)

        execute_test(input_type, castra_init, castra_load, castra_execute, castra_clear, castra_final)
    elif str(args_input_database).lower() == "numpymemorymap":
        import csv, numpy
        numpy_binary = 'tempfile.npy'
        def numpy_init():
            return None

        def numpy_load():
            with open(input_file, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                result = [int(x[0]) for x in reader]
                numpy_array = numpy.array(result, dtype=numpy.int32)
                numpy.save(numpy_binary, numpy_array)

        def numpy_execute():
            numpy_array = numpy.memmap(numpy_binary, dtype=numpy.int32)
            function(numpy_array)

        def numpy_clear():
            return None

        def numpy_final():
            os.remove(input_file)
            os.remove(numpy_binary)

        execute_test(input_type, numpy_init, numpy_load, numpy_execute, numpy_clear, numpy_final)
    elif str(args_input_database).lower() == "monetdbembedded":
        import monetdb_embedded, csv

        def monetdbembedded_init():
            monetdb_embedded.init('/tmp/dbfarm')
            try: monetdb_embedded.sql('DROP TABLE integers')
            except: pass

        def monetdbembedded_load():
            with open(input_file, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                result = [int(x[0]) for x in reader]
                monetdb_embedded.create('integers', ['i'], result)

        def monetdbembedded_execute():
            result = monetdb_embedded.sql('SELECT * FROM integers')
            function(result['i'])

        def monetdbembedded_clear():
            monetdb_embedded.sql('DROP TABLE integers')

        def monetdbembedded_final():
            os.remove(input_file)

        execute_test(input_type, monetdbembedded_init, monetdbembedded_load, monetdbembedded_execute, monetdbembedded_clear, monetdbembedded_final)

    else:
        print("Unrecognized database type %s, exiting..." % args_input_database)
