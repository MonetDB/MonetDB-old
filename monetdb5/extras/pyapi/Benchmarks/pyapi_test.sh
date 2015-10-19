        

# The base directory of testing, a new folder is created in this base directory [$PYAPI_TEST_DIR], and everything is done in that new folder
export PYAPI_BASE_DIR=/tmp
# The terminal to start mserver with, examples are gnome-terminal, xterm, konsole
export TERMINAL=x-terminal-emulator
# Port used by the MSERVER
export MSERVER_PORT=49979
# A command that tests if the mserver is still running (used to find out when the shutting down of mserver is completed)
export MSERVERTEST='netstat -ant | grep "127.0.0.1:$MSERVER_PORT.*LISTEN">/dev/null'
# Testing parameters
# Input test (zero copy vs copy)
# The input sizes to test (in MB)
export INPUT_TESTING_SIZES="1000 10000 100000"
# Amount of tests to run for each size
export INPUT_TESTING_NTESTS=3

# Output test (zero copy vs copy)
# The output sizes to test (in MB)
export OUTPUT_TESTING_SIZES="1000 10000 100000"
# Amount of tests to run for each size
export OUTPUT_TESTING_NTESTS=3

# String tests
# Strings of the same length (mb, length)
export STRINGSAMELENGTH_TESTING_SIZES="(1000,1) (1000,10) (1000,100) (1000,1000) (1000,10000) (1000,100000)"
export STRINGSAMELENGTH_TESTING_NTESTS=3
# Extreme length string testing (all strings have length 1 except for one string, which has EXTREME length)
# Arguments are (Extreme Length, String Count)
export STRINGEXTREMELENGTH_TESTING_SIZES="(10,1000000) (100,1000000) (1000,1000000) (10000,1000000)"
export STRINGEXTREMELENGTH_TESTING_NTESTS=3
# Check Unicode vs Always Unicode (ASCII) (mb, length)
export STRINGUNICODE_TESTING_SIZES="(1000,10) (1000,100) (1000,1000) (1000,10000)"
export STRINGUNICODE_TESTING_NTESTS=3

# Multithreading tests
export MULTITHREADING_NR_THREADS="1 2 4 8 16 32 64 96"
export MULTITHREADING_TESTING_SIZES="1"
#amount of tests for each thread
export MULTITHREADING_TESTING_NTESTS=3

# Quantile speedtest
# The input sizes to test (in MB)
export QUANTILE_TESTING_SIZES="1000 8000 40000"
# Amount of tests to run for each size
export QUANTILE_TESTING_NTESTS=3

# PyAPI TAR url
export PYAPI_BRANCH_NAME=pyapi
export PYAPI_TAR_FILE=$PYAPI_BRANCH_NAME.tar.gz
export PYAPI_TAR_URL=http://dev.monetdb.org/hg/MonetDB/archive/$PYAPI_TAR_FILE
# You probably dont need to change these
export PYAPI_TEST_DIR=$PYAPI_BASE_DIR/monetdb_pyapi_test
export PYAPI_MONETDB_DIR=$PYAPI_TEST_DIR/MonetDB-$PYAPI_BRANCH_NAME
export PYAPI_SRC_DIR=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi
export PYAPI_BUILD_DIR=$PYAPI_TEST_DIR/build
export PYAPI_OUTPUT_DIR=$PYAPI_TEST_DIR/output

# Used for downloading the python-monetdb connector (import monetdb.sql)
export PYTHON_MONETDB_CONNECTOR_VERSION=11.19.3.2
export PYTHON_MONETDB_DIR=python-monetdb-$PYTHON_MONETDB_CONNECTOR_VERSION
export PYTHON_MONETDB_FILE=python-monetdb-$PYTHON_MONETDB_CONNECTOR_VERSION.tar.gz
export PYTHON_MONETDB_URL=https://pypi.python.org/packages/source/p/python-monetdb/$PYTHON_MONETDB_FILE

# Datafarm Dir
export PYAPI_DATAFARM_DIR=$PYAPI_BUILD_DIR/var/monetdb5/dbfarm
# Benchmarks DIR
export PYAPI_BENCHMARKS_DIR=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi/Benchmarks
# Python testfile location
export PYAPI_TESTFILE=$PYAPI_BENCHMARKS_DIR/monetdb_testing.py
# Graph file location
export PYAPI_GRAPHFILE=$PYAPI_BENCHMARKS_DIR/graph.py

# Try a bunch of popular different terminals
export SETSID=0
type $TERMINAL >/dev/null 2>&1
if [ $? -ne 0  ]; then
    export TERMINAL=gnome-terminal
fi
type $TERMINAL >/dev/null 2>&1
if [ $? -ne 0  ]; then
    export TERMINAL=xterm
fi
type $TERMINAL >/dev/null 2>&1
if [ $? -ne 0  ]; then
    export TERMINAL=konsole
fi
type $TERMINAL >/dev/null 2>&1
if [ $? -ne 0  ]; then
    export TERMINAL=setsid
    export SETSID=1
fi

function pyapi_build() {
    cd $PYAPI_TEST_DIR
    python -c "import numpy"
    if [ $? -ne 0 ]; then
        read -p "Failed to load library Numpy. Would you like me to try and install it for you? (y/n):  " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        wget https://github.com/numpy/numpy/archive/master.zip && unzip master.zip && cd numpy-master && python setup.py install --user && cd .. && python -c "import numpy"
        if [$? -eq 0]; then
            echo "Successfully installed Numpy."
        else
            echo "Failed to install Numpy. Exiting..."
            return 1
        fi
    fi    
    python -c "import monetdb.sql"
    if [ $? -ne 0 ]; then
        read -p "Failed to load library MonetDB Python connector. Would you like me to try and install it for you? (y/n):  " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        wget $PYTHON_MONETDB_URL && tar xvzf $PYTHON_MONETDB_FILE && cd $PYTHON_MONETDB_DIR && python setup.py install --user && cd .. && python -c "import monetdb.sql"
        if [$? -eq 0]; then
            echo "Successfully installed monetdb.sql."
        else
            echo "Failed to install monetdb.sql. Exiting..."
            return 1
        fi
    fi
    echo "Finished testing for libraries. Downloading and installing MonetDB."
    wget $PYAPI_TAR_URL && tar xvzf $PYAPI_TAR_FILE && cd $PYAPI_MONETDB_DIR && printf '#ifndef _PYAPI_TESTING_\n#define _PYAPI_TESTING_\n#define _PYAPI_VERBOSE_\n#endif\n' | cat - $PYAPI_SRC_DIR/pyapi.h > $PYAPI_SRC_DIR/temp && mv $PYAPI_SRC_DIR/temp $PYAPI_SRC_DIR/pyapi.h && ./bootstrap && ./configure --prefix=$PYAPI_BUILD_DIR --enable-debug=no --enable-assert=no --enable-optimize=yes && make -j install
    if [ $? -ne 0 ]; then
        echo "Failed to download and install MonetDB. Exiting..."
        return 1
    fi
}

function pyapi_run_single_test_echo() {
    echo \$PYAPI_BUILD_DIR/bin/mserver5 --set mapi_port=\$MSERVER_PORT --set embedded_py=true --set enable_pyverbose=true --set pyapi_benchmark_output=\$PYAPI_OUTPUT_DIR/temp_output.tsv $2
    echo python \$PYAPI_TESTFILE MONETDB $3 $4 $5 \$MSERVER_PORT $6
}

function pyapi_run_single_test() {
    echo "Beginning Test $1"
    rm -rf $PYAPI_DATAFARM_DIR
    if [ $SETSID -eq 1 ]; then
        $TERMINAL $PYAPI_BUILD_DIR/bin/mserver5 --set mapi_port=$MSERVER_PORT --set embedded_py=true --set enable_pyverbose=true --set enable_oldnullmask=true --set pyapi_benchmark_output=$PYAPI_OUTPUT_DIR/temp_output.tsv $2 && python $PYAPI_TESTFILE MONETDB $3 $4 $5 $MSERVER_PORT $6 && killall mserver5
    else
        $TERMINAL -e "$PYAPI_BUILD_DIR/bin/mserver5  --set mapi_port=$MSERVER_PORT --set embedded_py=true --set enable_pyverbose=true --set pyapi_benchmark_output=$PYAPI_OUTPUT_DIR/temp_output.tsv $2" && python $PYAPI_TESTFILE MONETDB $3 $4 $5 $MSERVER_PORT $6 && killall mserver5
    fi
    if [ $? -ne 0 ]; then
        echo "Failed Test $1"
        killall mserver5
        return 1
    fi
    for i in `seq 1 20`; do
        eval $MSERVERTEST
        if [ $? -eq 0 ]; then
            sleep 1
        else 
            echo "Finished Test $1"
            return 0
        fi
    done
    echo "Failed to close mserver, exiting..."
    return 1
}

export POSTGRES_BASEDIR=$PYAPI_TEST_DIR/postgres
export POSTGRES_BUILD_DIR=$POSTGRES_BASEDIR/build
export PGDATA=$POSTGRES_BASEDIR/postgres_data

export POSTGRES_VERSION=9.4.4
export POSTGRES_BASE=postgresql-$POSTGRES_VERSION
export POSTGRES_TAR_FILE=$POSTGRES_BASE.tar.gz
export POSTGRES_TAR_URL=https://ftp.postgresql.org/pub/source/v$POSTGRES_VERSION/$POSTGRES_TAR_FILE

export POSTGRES_DB_NAME=python_test
export POSTGRES_SERVER_COMMAND=$POSTGRES_BUILD_DIR/bin/postgres
export POSTGRES_CREATEDB_COMMAND="$POSTGRES_BUILD_DIR/bin/createdb $POSTGRES_DB_NAME && $POSTGRES_BUILD_DIR/bin/createlang --dbname=$POSTGRES_DB_NAME plpythonu" 
export POSTGRES_CLIENT_COMMAND="$POSTGRES_BUILD_DIR/bin/psql --dbname=$POSTGRES_DB_NAME"
export POSTGRES_DROPDB_COMMAND="$POSTGRES_BUILD_DIR/bin/dropdb $POSTGRES_DB_NAME"
export POSTGRES_INPUT_FILE=$POSTGRES_BASEDIR/input.csv

export POSTGRES_CWD=$(pwd)

function postgres_build() {
    wget $POSTGRES_TAR_URL && tar xvzf $POSTGRES_TAR_FILE && cd $POSTGRES_BASE && ./configure --prefix=$POSTGRES_BUILD_DIR --with-python && make && make install && $POSTGRES_BUILD_DIR/bin/initdb
}

function monetdbmapi_run_single_test() {
    rm -rf $PYAPI_DATAFARM_DIR
    setsid $PYAPI_BUILD_DIR/bin/mserver5 --set mapi_port=$MSERVER_PORT --set embedded_py=true $2 > /dev/null
    python $PYAPI_TESTFILE $1 $3 $4 $5 $MSERVER_PORT $6
    killall mserver5
    if [ $? -ne 0 ]; then
        echo "Failed Test $1"
        killall mserver5
        return 1
    fi
    for i in `seq 1 20`; do
        eval $MSERVERTEST
        if [ $? -eq 0 ]; then
            sleep 1
        else 
            echo "Finished Test $1"
            return 0
        fi
    done
    echo "Failed to close mserver, exiting..."
    return 1
}

function postgres_run_single_test() {
    # start server
    setsid $POSTGRES_SERVER_COMMAND -c autovacuum=off -c random_page_cost=3.5 -c geqo_threshold=15 -c from_collapse_limit=14 -c join_collapse_limit=14 -c default_statistics_target=10000 -c constraint_exclusion=on -c checkpoint_completion_target=0.9 -c wal_buffers=16MB -c checkpoint_segments=128 -c shared_buffers=256GB -c effective_cache_size=768GB -c work_mem=128GB > /dev/null && sleep 5
    # call python test script
    python "$PYAPI_TESTFILE" $5 $1 $2 $3 $MSERVER_PORT $4
    # finish testing, kill postgres
    killall postgres
}

export SQLITE_DB_FILE=$PYAPI_TEST_DIR/sqlite_dbfile.dbname
function python_run_single_test() {
    python "$PYAPI_TESTFILE" $1 $2 $3 $4 $MSERVER_PORT $5
}

function pyapi_test_input() {
    echo "Beginning Input Testing (Copy vs Zero Copy)"
    pyapi_run_single_test "Input Testing (Zero Copy)" "" "INPUT" input_zerocopy "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Input Testing (Copy)" "--set disable_pyzerocopyinput=true" "INPUT" input_copy "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Input Testing (Map)" "--forcemito" "INPUT-MAP" input_zerocopy_map "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}


function pyapi_test_input_null() {
    echo "Beginning Input Testing [NULL] (Copy vs Zero Copy)"
    pyapi_run_single_test "Input Testing (Zero Copy)" "" "INPUT-NULL" input_zerocopy_null "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Input Testing (Copy)" "--set disable_pyzerocopyinput=true" "INPUT-NULL" input_copy_null "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_output() {
    pyapi_run_single_test "Output Testing (Zero Copy)" "--set gdk_mmap_minsize=99999999999999999999999" "OUTPUT" output_zerocopy "$OUTPUT_TESTING_NTESTS" "$OUTPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "Output Testing (Copy)" "--set disable_pyzerocopyoutput=true --set gdk_mmap_minsize=99999999999999999999999" "OUTPUT" output_copy "$OUTPUT_TESTING_NTESTS" "$OUTPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_samelength() {
    pyapi_run_single_test "String Testing (LazyArray, Same Length)" "--set enable_lazyarray=true" "STRING_SAMELENGTH" string_samelength_lazyarray "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_OBJECT, Same Length)" "--set disable_numpystringarray=true" "STRING_SAMELENGTH" string_samelength_npyobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Same Length)" "" "STRING_SAMELENGTH" string_samelength_npystring "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_extreme() {
    pyapi_run_single_test "String Testing (NPY_OBJECT, Extreme Length)" "--set disable_numpystringarray=true" "STRING_EXTREMELENGTH" string_extremelength_npyobject "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Extreme Length)" "" "STRING_EXTREMELENGTH" string_extremelength_npystring "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_unicode_ascii() {
    pyapi_run_single_test "String Testing (Check Unicode, ASCII)" "--set disable_numpystringarray=true" "STRING_SAMELENGTH" string_unicode_ascii_check "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Always Unicode, ASCII)" "--set disable_numpystringarray=true --set enable_alwaysunicode=true" "STRING_SAMELENGTH" string_unicode_ascii_always "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Check Unicode, Extreme)" "--set disable_numpystringarray=true" "STRING_EXTREMEUNICODE" string_unicode_extreme_check "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Always Unicode, Extreme)" "--set disable_numpystringarray=true" "STRING_EXTREMEUNICODE" string_unicode_extreme_always "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_bytearray_vs_string() {
    pyapi_run_single_test "String Testing (ByteArray Object)" "--set disable_numpystringarray=true --set enable_bytearray=true" "STRING_SAMELENGTH" string_bytearrayobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (String Object)" "--set disable_numpystringarray=true" "STRING_SAMELENGTH" string_stringobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_threads() {
    rm multithreading.tsv
    for thread in $MULTITHREADING_NR_THREADS
    do
        pyapi_run_single_test "Multithreading ($thread Threads)" "--forcemito --set gdk_nr_threads=$thread" "FACTORIAL-$thread" multithreading "$MULTITHREADING_TESTING_NTESTS" "$MULTITHREADING_TESTING_SIZES"
        if [ $? -ne 0 ]; then
            return 1
        fi
    done
}

function pyapi_test_quantile() {
    echo "Beginning Quantile Testing (Python vs R vs MonetDB)"
    pyapi_run_single_test "Quantile Testing (Python)" "" "PQUANTILE" quantile_python "$QUANTILE_TESTING_NTESTS" "$QUANTILE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Quantile Testing (MonetDB)" "" "QUANTILE" quantile_monetdb "$QUANTILE_TESTING_NTESTS" "$QUANTILE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Quantile Testing (R)" "--set embedded_r=true" "RQUANTILE" quantile_r "$QUANTILE_TESTING_NTESTS" "$QUANTILE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_run_tests() {
    if [ -d $PYAPI_OUTPUT_DIR ]; then
        read -p "Directory $PYAPI_OUTPUT_DIR already exists, should we delete it? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf $PYAPI_OUTPUT_DIR
        else
            return 1
        fi
    fi
    mkdir $PYAPI_OUTPUT_DIR && cd $PYAPI_OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "Failed to create output directory."
        return 1
    fi

    pyapi_test_input
    pyapi_test_input_null
    pyapi_test_output
    pyapi_test_string_samelength
    pyapi_test_string_extreme
    pyapi_test_string_unicode_ascii
    pyapi_test_bytearray_vs_string
    pyapi_test_quantile
    pyapi_test_threads
}

function pyapi_graph() {
    python $PYAPI_GRAPHFILE "SAVE" "Input (Both)" "-xlog" "Zero Copy:input_zerocopy.tsv" "Copy:input_copy.tsv" "Zero Copy (Null):input_zerocopy_null.tsv" "Copy (Null):input_copy_null.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input" "-xlog" "Zero Copy:input_zerocopy.tsv" "Copy:input_copy.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input-Null" "-xlog" "Zero Copy:input_zerocopy_null.tsv" "Copy:input_copy_null.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input-Map" "-xlog" "Zero Copy:input_zerocopy.tsv" "Zero Copy (Map):input_zerocopy_map.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Output" "-xlog" "Zero Copy:output_zerocopy.tsv" "Copy:output_copy.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "String Samelength" "-xlog" "Numpy Object:string_samelength_npyobject.tsv" "Numpy String:string_samelength_npystring.tsv" "Lazy Array:string_samelength_lazyarray.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "String Extremelength" "-xlog" "Numpy Object:string_extremelength_npyobject.tsv" "Numpy String:string_extremelength_npystring.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (ASCII)" "-xlog" "Check Unicode:string_unicode_ascii_check.tsv" "Always Unicode:string_unicode_ascii_always.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (Extreme)" "-xlog" "Check Unicode:string_unicode_extreme_check.tsv" "Always Unicode:string_unicode_extreme_always.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "ByteArrayObject vs StringObject" "-xlog" "Byte Array Object:string_bytearrayobject.tsv" "String Object:string_stringobject.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Quantile Speedtest" "-xlog" "Python:quantile_python.tsv" "R:quantile_r.tsv" "MonetDB:quantile_monetdb.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Multithreading Test" "-lineplot" "Threads:multithreading.tsv"

    python $PYAPI_GRAPHFILE "SAVE" "Input (Both) y-log" "-xlog" "-ylog" "Zero Copy:input_zerocopy.tsv" "Copy:input_copy.tsv" "Zero Copy (Null):input_zerocopy_null.tsv" "Copy (Null):input_copy_null.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input y-log" "-xlog" "-ylog" "Zero Copy:input_zerocopy.tsv" "Copy:input_copy.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input-Null y-log" "-xlog" "-ylog" "Zero Copy:input_zerocopy_null.tsv" "Copy:input_copy_null.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input-Map y-log" "-xlog" "-ylog" "Zero Copy:input_zerocopy.tsv" "Zero Copy (Map):input_zerocopy_map.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Output y-log" "-xlog" "-ylog" "Zero Copy:output_zerocopy.tsv" "Copy:output_copy.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "String Samelength y-log" "-xlog" "-ylog" "Numpy Object:string_samelength_npyobject.tsv" "Numpy String:string_samelength_npystring.tsv" "Lazy Array:string_samelength_lazyarray.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "String Extremelength y-log" "-xlog" "-ylog" "Numpy Object:string_extremelength_npyobject.tsv" "Numpy String:string_extremelength_npystring.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (ASCII) y-log" "-xlog" "-ylog" "Check Unicode:string_unicode_ascii_check.tsv" "Always Unicode:string_unicode_ascii_always.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (Extreme) y-log" "-xlog" "-ylog" "Check Unicode:string_unicode_extreme_check.tsv" "Always Unicode:string_unicode_extreme_always.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "ByteArrayObject vs StringObject y-log" "-xlog" "-ylog" "Byte Array Object:string_bytearrayobject.tsv" "String Object:string_stringobject.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Quantile Speedtest y-log" "-xlog" "-ylog" "Python:quantile_python.tsv" "R:quantile_r.tsv" "MonetDB:quantile_monetdb.tsv"
}


function pyapi_cleanup() {
    read -p "Finished testing, would you like me to remove the test directory $PYAPI_TEST_DIR and everything in it? (y/n):  " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $PYAPI_TEST_DIR
    fi
    return 0
}

function pyapi_test() {
    if [ ! -d $PYAPI_TEST_DIR ]; then
        mkdir $PYAPI_TEST_DIR
    fi

    if [ -d $PYAPI_MONETDB_DIR ]; then
        read -p "Directory $PYAPI_MONETDB_DIR already exists, skip the building and continue to testing? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            read -p "Should we delete the directory $PYAPI_MONETDB_DIR and rebuild everything? WARNING: This will delete everything in the directory $PYAPI_TEST_DIR. (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                read -p "Are you absolutely sure you want to delete everything in $PYAPI_MONETDB_DIR? (y/n): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf $PYAPI_MONETDB_DIR
                    pyapi_build
                    if [ $? -ne 0 ]; then
                        return 1
                    fi
                else
                    return 1
                fi
            else
                return 1
            fi
        fi
    else
        pyapi_build
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi

    if ! [[ -a $PYAPI_BUILD_DIR/bin/mserver5 ]]; then 
        echo "mserver5 not found, building monetdb failed."
        return 1
    fi
    type $TERMINAL >/dev/null 2>&1
    if [ $? -ne 0  ]; then
        echo "\"$TERMINAL\" could not be found, please set the \$TERMINAL variable to a proper value."
        return 1
    fi

    pyapi_run_tests
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_graph
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_cleanup
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function postgres_test() {
    if [ ! -d $PYAPI_TEST_DIR ]; then
        mkdir $PYAPI_TEST_DIR
    fi
    cd $PYAPI_TEST_DIR

    if [ -d $POSTGRES_BASEDIR ]; then
        read -p "Directory $POSTGRES_BASEDIR already exists, skip the building and continue to testing? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            read -p "Should we delete the directory $POSTGRES_BASEDIR and rebuild everything? WARNING: This will delete everything in the directory $PYAPI_TEST_DIR. (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                read -p "Are you absolutely sure you want to delete everything in $POSTGRES_BASEDIR? (y/n): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf $POSTGRES_BASEDIR
                    postgres_build
                    if [ $? -ne 0 ]; then
                        return 1
                    fi
                else
                    return 1
                fi
            else
                return 1
            fi
        fi
    else
        postgres_build
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi

    postgres_run_tests
}

export DROP_CACHE_COMMAND='sync && echo 3 | sudo /usr/bin/tee /proc/sys/vm/drop_caches'

export ntests_identity=3
export sizes_identity="10"

export ntests_sqroot=3
export sizes_sqroot="10"

export ntests_quantile=3
export sizes_quantile="1 1000"

export PYTHON_TESTS=("identity" "sqroot" "quantile")
export PYTHON_MAP_TESTS=("identity" "sqroot")
export PLPYTHON_TESTS=("quantile")
export POSTGRES_TESTS=("quantile")
export MONETDB_TESTS=("quantile")

function plpython_run_tests() {
    for i in "${PLPYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        postgres_run_single_test $i plpython_$i ${!n} "${!s}" PLPYTHON
        sleep 5
    done
}

function postgres_run_tests() {
    for i in "${POSTGRES_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        postgres_run_single_test $i postgres_$i ${!n} "${!s}" POSTGRES
        sleep 5
    done
}

function sqlite_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test SQLITEMEM $i sqlitemem_$i ${!n} "${!s}"
        python_run_single_test SQLITEDB $i sqlitedb_$i ${!n} "${!s}"
    done
}

function csv_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test CSV $i csv_$i ${!n} "${!s}"
    done
}


function numpy_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test NUMPYBINARY $i:COLD numpy_cold_$i ${!n} "${!s}"
        python_run_single_test NUMPYBINARY $i:HOT numpy_hot_$i ${!n} "${!s}"
    done
}


function monetdbembedded_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test MONETDBEMBEDDED $i monetdbembedded_$i ${!n} "${!s}"
    done
}

function numpymmap_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test NUMPYMEMORYMAP $i:COLD numpymmap_cold_$i ${!n} "${!s}"
        python_run_single_test NUMPYMEMORYMAP $i:HOT numpymmap_hot_$i ${!n} "${!s}"
    done
}

function monetdbmapi_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        monetdbmapi_run_single_test MONETDBMAPI "" $i monetdbmapi_$i ${!n} "${!s}"
    done
}

function monetdbpyapi_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        monetdbmapi_run_single_test PYAPI "--set gdk_nr_threads=1" $i monetdbpyapi_$i ${!n} "${!s}"
    done
}

function monetdbpyapimap_test() {
    for i in "${PYTHON_MAP_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        monetdbmapi_run_single_test PYAPIMAP "" $i monetdbpyapimap_$i ${!n} "${!s}"
    done
}

function monetdb_test() {
    for i in "${MONETDB_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        monetdbmapi_run_single_test MONETDB "" $i monetdb_$i ${!n} "${!s}"
    done
}

function monetdbrapi_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        monetdbmapi_run_single_test RAPI "--set gdk_nr_threads=1 --set embedded_r=true" $i monetdbrapi_$i ${!n} "${!s}"
    done
}

function pytables_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test PYTABLES $i pytables_$i ${!n} "${!s}"
    done
}

function pyfits_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test PYFITS $i pyfits_$i ${!n} "${!s}"
    done
}

function pandascsv_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test PANDASCSV $i pandascsv_$i ${!n} "${!s}"
    done
}

function castra_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        python_run_single_test CASTRA $i castra_$i ${!n} "${!s}"
    done
}

function psycopg2_test() {
    for i in "${PYTHON_TESTS[@]}"
    do
        n=ntests_$i
        s=sizes_$i
        postgres_run_single_test $i psycopg2_$i ${!n} "${!s}" PSYCOPG2
        sleep 5
    done
}

function psycopg2_install() {
    wget http://initd.org/psycopg/tarballs/PSYCOPG-2-6/psycopg2-2.6.1.tar.gz && tar xvzf psycopg2-2.6.1.tar.gz && rm psycopg2-2.6.1.tar.gz && cd psycopg2-2.6.1 && python setup.py install --user build_ext --pg-config $POSTGRES_BUILD_DIR/bin/pg_config
}

function cython_install() {
    wget http://cython.org/release/Cython-0.23.1.tar.gz && tar xvzf Cython-0.23.1.tar.gz && cd Cython-0.23.1 && python setup.py install --user
}

function pytables_install() {
    wget https://github.com/PyTables/PyTables/archive/develop.zip && unzip develop.zip && cd PyTables-develop && python setup.py install --user
}

function comparison_test() {
    if [ ! -d $PYAPI_OUTPUT_DIR ]; then
        mkdir $PYAPI_OUTPUT_DIR
    fi
    cd $PYAPI_OUTPUT_DIR
    
    plpython_run_tests
    postgres_run_tests
    #sqlite_test
    #csv_test
    numpy_test
    numpymmap_test
    #monetdbembedded_test
    #monetdbmapi_test
    #monetdbpyapi_test
    #monetdbpyapimap_test
    #monetdbrapi_test
    #psycopg2_test
    #pytables_test
    castra_test
    monetdb_test
    #pyfits_test
    #pandascsv_test
}

function castra_install() {
    wget https://github.com/blaze/castra/archive/master.zip && unzip master.zip && cd castra-master && python setup.py install --user
}

function pandas_install() {
    wget https://pypi.python.org/packages/source/p/pandas/pandas-0.16.2.tar.gz && tar xvzf pandas-0.16.2.tar.gz && cd pandas-0.16.2 && python setup.py install --user
}

function pyfits_install() {
    wget https://pypi.python.org/packages/source/p/pyfits/pyfits-3.3.tar.gz && tar xvzf pyfits-3.3.tar.gz && cd pyfits-3.3 && python setup.py install --user && cd .. && rm -rf pyfits-3.3
}

function comparison_graph() {
    python $PYAPI_GRAPHFILE "SAVE" "Identity" "postgres:postgres_identity.tsv" "sqlitemem:sqlitemem_identity.tsv" "sqlitedb:sqlitedb_identity.tsv" "csv:csv_identity.tsv" "numpy:numpy_identity.tsv" "numpymmap:numpymmap_identity.tsv" "monetdbembedded:monetdbembedded_identity.tsv" "monetdbmapi:monetdbmapi_identity.tsv" "monetdbpyapi:monetdbpyapi_identity.tsv" "monetdbpyapimap:monetdbpyapimap_identity.tsv" "monetdbrapi:monetdbrapi_identity.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Identity [Fast Only]" "numpy:numpy_identity.tsv" "numpymmap:numpymmap_identity.tsv" "monetdbembedded:monetdbembedded_identity.tsv" "monetdbpyapi:monetdbpyapi_identity.tsv" "monetdbpyapimap:monetdbpyapimap_identity.tsv" "monetdbrapi:monetdbrapi_identity.tsv"

    python $PYAPI_GRAPHFILE "SAVE" "Square Root" "postgres:postgres_sqroot.tsv" "sqlitemem:sqlitemem_sqroot.tsv" "sqlitedb:sqlitedb_sqroot.tsv" "csv:csv_sqroot.tsv" "numpy:numpy_sqroot.tsv" "numpymmap:numpymmap_sqroot.tsv" "monetdbembedded:monetdbembedded_sqroot.tsv" "monetdbmapi:monetdbmapi_sqroot.tsv" "monetdbpyapi:monetdbpyapi_sqroot.tsv" "monetdbpyapimap:monetdbpyapimap_sqroot.tsv" "monetdbrapi:monetdbrapi_sqroot.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Square Root [Fast Only]" "numpy:numpy_sqroot.tsv" "numpymmap:numpymmap_sqroot.tsv" "monetdbembedded:monetdbembedded_sqroot.tsv" "monetdbpyapi:monetdbpyapi_sqroot.tsv" "monetdbpyapimap:monetdbpyapimap_sqroot.tsv" "monetdbrapi:monetdbrapi_sqroot.tsv"

}

export BUILD_DIR=/export/scratch2/raasveld/build
export CPATH=/export/scratch2/raasveld/build/include
export LIBRARY_PATH=/export/scratch2/raasveld/build/lib
function install_cfitsio() {
    wget ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio_latest.tar.gz && tar xvzf cfitsio_latest.tar.gz && cd cfitsio && ./configure --enable-sse2 --prefix=$BUILD_DIR --enable-reentrant && make install
}

function install_wcslib() {
    wget ftp://ftp.atnf.csiro.au/pub/software/wcslib/wcslib.tar.bz2 && tar jxf wcslib.tar.bz2 && cd wcslib-5.9 && ./configure --prefix=$BUILD_DIR --without-pgplot && gmake && gmake check && gmake install
}

function install_casacore() {
    export PATH=$PATH:$BUILD_DIR
    wget https://github.com/casacore/casacore/archive/master.zip && unzip master.zip && rm master.zip && cd casacore-master && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX:PATH=$BUILD_DIR -DBUILD_PYTHON=ON .. && make all install
}

function install_pythoncasacore() {
    wget https://github.com/casacore/python-casacore/archive/master.zip && unzip master.zip && rm master.zip && cd python-casacore-master && python setup.py install --user
}

function install_lofar() {
    wget https://github.com/transientskp/tkp/archive/master.zip && unzip master.zip && rm master.zip && cd tkp-master && python setup.py install --user
}


function install_mongodb() {
    wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-3.0.7.tgz && tar xvzf mongodb-linux-x86_64-3.0.7.tgz && cd mongodb-linux-x86_64-3.0.7 && cp -R -n bin/ $BUILD_DIR
    mongod --dbpath=/export/scratch2/raasveld/mongodata
}

wget https://bitbucket.org/djcbeach/monary/get/5b0fb0c2de0a.zip

function install_scons() {
    wget http://prdownloads.sourceforge.net/scons/scons-2.3.6.tar.gz && tar xvzf scons-2.3.6.tar.gz && cd scons-2.3.6 && python setup.py install --user    
}

function install_mongodb() {
    rm master.zip
    wget https://github.com/mongodb/mongo/archive/master.zip && unzip master.zip && cd mongo-master && ~/.local/bin/scons all
}



export PYAPI_TESTFILE=/local/raasveld/monetdb_testing.py
export LD_LIBRARY_PATH=/local/raasveld/build/lib
