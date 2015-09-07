        

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
export INPUT_TESTING_SIZES="10"
# Amount of tests to run for each size
export INPUT_TESTING_NTESTS=10

# Output test (zero copy vs copy)
# The output sizes to test (in MB)
export OUTPUT_TESTING_SIZES="0.1 1 10 100 1000"
# Amount of tests to run for each size
export OUTPUT_TESTING_NTESTS=10

# String tests
# Strings of the same length (mb, length)
export STRINGSAMELENGTH_TESTING_SIZES="(100,1) (100,10) (100,100) (100,125) (100,150) (100,175) (100,200) (100,201) (100,210) (100,211) (100,212) (100,213) (100,214) (100,215) (100,216) (100,217) (100,218) (100,219) (100,220) (100,225) (100,250) (100,500) (100,750) (100,1000) (100,1250) (100,1500) (100,2000) (100,2500) (100,10000) (100,100000)"
export STRINGSAMELENGTH_TESTING_NTESTS=5
# Extreme length string testing (all strings have length 1 except for one string, which has EXTREME length)
# Arguments are (Extreme Length, String Count)
export STRINGEXTREMELENGTH_TESTING_SIZES="(10,1000000) (100,1000000) (1000,1000000) (10000,1000000)"
export STRINGEXTREMELENGTH_TESTING_NTESTS=1
# Check Unicode vs Always Unicode (ASCII) (mb, length)
export STRINGUNICODE_TESTING_SIZES="(1000,10) (10000,10) (1000,100) (10000,100)"
export STRINGUNICODE_TESTING_NTESTS=1

# Multithreading tests
export MULTITHREADING_NR_THREADS="1 2 3 4 5 6 7 8"
export MULTITHREADING_TESTING_SIZES="1"
#amount of tests for each thread
export MULTITHREADING_TESTING_NTESTS=10

# Quantile speedtest
# The input sizes to test (in MB)
export QUANTILE_TESTING_SIZES="0.1 1 10 100 1000"
# Amount of tests to run for each size
export QUANTILE_TESTING_NTESTS=10

# PyAPI TAR url
export PYAPI_TAR_NAME=pyapi
export PYAPI_TAR_FILE=$PYAPI_TAR_NAME.tar.gz
export PYAPI_TAR_URL=http://dev.monetdb.org/hg/MonetDB/archive/$PYAPI_TAR_FILE
# You probably dont need to change these
export PYAPI_TEST_DIR=$PYAPI_BASE_DIR/monetdb_pyapi_test
export PYAPI_MONETDB_DIR=$PYAPI_TEST_DIR/MonetDB-$PYAPI_TAR_NAME
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
    echo "Making directory $PYAPI_MONETDB_DIR."
    mkdir $PYAPI_MONETDB_DIR && cd $PYAPI_TEST_DIR
    if [ $? -ne 0 ]; then
        echo "Failed to create testing directory, exiting..."
        return 1
    fi
    python -c "import numpy"
    if [ $? -ne 0 ]; then
        read -p "Failed to load library Numpy. Would you like me to try and install it for you? (y/n):  " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        pip install --user numpy && python -c "import numpy"
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
        wget $PYTHON_MONETDB_URL && tar xvzf $PYTHON_MONETDB_FILE && cd $PYTHON_MONETDB_DIR && python setup.py install --user && python -c "import monetdb.sql"
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
        $TERMINAL $PYAPI_BUILD_DIR/bin/mserver5 --set mapi_port=$MSERVER_PORT --set embedded_py=true --set enable_pyverbose=true --set pyapi_benchmark_output=$PYAPI_OUTPUT_DIR/temp_output.tsv $2 && python $PYAPI_TESTFILE MONETDB $3 $4 $5 $MSERVER_PORT $6 && killall mserver5
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
    setsid $POSTGRES_SERVER_COMMAND > /dev/null && sleep 5
    # call python test script
    python "$PYAPI_TESTFILE" POSTGRES $1 $2 $3 $MSERVER_PORT $4
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

    pyapi_run_single_test "String Testing (NPY_OBJECT, Same Length)" "" "STRING_SAMELENGTH" string_samelength_npyobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Same Length)" "--set enable_numpystringarray=true" "STRING_SAMELENGTH" string_samelength_npystring "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_extreme() {
    pyapi_run_single_test "String Testing (NPY_OBJECT, Extreme Length)" "" "STRING_EXTREMELENGTH" string_extremelength_npyobject "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Extreme Length)" "--set enable_numpystringarray=true" "STRING_EXTREMELENGTH" string_extremelength_npystring "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_unicode_ascii() {
    pyapi_run_single_test "String Testing (Check Unicode, ASCII)" "" "STRING_SAMELENGTH" string_unicode_ascii_check "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Always Unicode, ASCII)" "--set enable_alwaysunicode=true" "STRING_SAMELENGTH" string_unicode_ascii_always "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Check Unicode, Extreme)" "" "STRING_EXTREMEUNICODE" string_unicode_extreme_check "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Always Unicode, Extreme)" "" "STRING_EXTREMEUNICODE" string_unicode_extreme_always "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_bytearray_vs_string() {
    pyapi_run_single_test "String Testing (ByteArray Object)" "--set enable_bytearray=true" "STRING_SAMELENGTH" string_bytearrayobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (String Object)" "" "STRING_SAMELENGTH" string_stringobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
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
    
    #pyapi_test_input
    #pyapi_test_input_null
    #pyapi_test_output
    pyapi_test_string_samelength
    #pyapi_test_string_extreme
    #pyapi_test_string_unicode_ascii
    #pyapi_test_bytearray_vs_string
    #pyapi_test_quantile
    #pyapi_test_threads
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

export IDENTITY_NTESTS=3
export IDENTITY_SIZES="100"

function postgres_run_tests() {
    if [ ! -d $PYAPI_OUTPUT_DIR ]; then
        mkdir $PYAPI_OUTPUT_DIR
    fi
    cd $PYAPI_OUTPUT_DIR
    postgres_run_single_test IDENTITY postgres_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    postgres_run_single_test SQROOT postgres_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function sqlite_test() {
    python_run_single_test SQLITEMEM IDENTITY sqlitemem_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    python_run_single_test SQLITEDB IDENTITY sqlitedb_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"

    python_run_single_test SQLITEMEM SQROOT sqlitemem_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
    python_run_single_test SQLITEDB SQROOT sqlitedb_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function csv_test() {
    python_run_single_test CSV IDENTITY csv_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    python_run_single_test CSV SQROOT csv_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}


function numpy_test() {
    python_run_single_test NUMPYBINARY IDENTITY numpy_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    python_run_single_test NUMPYBINARY SQROOT numpy_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}


function monetdbembedded_test() {
    python_run_single_test MONETDBEMBEDDED IDENTITY monetdbembedded_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    python_run_single_test MONETDBEMBEDDED SQROOT monetdbembedded_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function numpymmap_test() {
    python_run_single_test NUMPYMEMORYMAP IDENTITY numpymmap_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    python_run_single_test NUMPYMEMORYMAP SQROOT numpymmap_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function monetdbmapi_test() {
    #monetdbmapi_run_single_test MONETDBMAPI "" IDENTITY monetdbmapi_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    monetdbmapi_run_single_test MONETDBMAPI "" SQROOT monetdbmapi_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function monetdbpyapi_test() {
    monetdbmapi_run_single_test PYAPI "--set gdk_nr_threads=1" IDENTITY monetdbpyapi_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    monetdbmapi_run_single_test PYAPI "--set gdk_nr_threads=1" SQROOT monetdbpyapi_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function monetdbpyapimap_test() {
    monetdbmapi_run_single_test PYAPIMAP "" IDENTITY monetdbpyapimap_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    monetdbmapi_run_single_test PYAPIMAP "" SQROOT monetdbpyapimap_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function monetdbrapi_test() {
    monetdbmapi_run_single_test RAPI "--set gdk_nr_threads=1 --set embedded_r=true" IDENTITY monetdbrapi_identity $IDENTITY_NTESTS "$IDENTITY_SIZES"
    monetdbmapi_run_single_test RAPI "--set gdk_nr_threads=1 --set embedded_r=true" SQROOT monetdbrapi_sqroot $IDENTITY_NTESTS "$IDENTITY_SIZES"
}

function psycopg_install() {
    wget http://initd.org/psycopg/tarballs/PSYCOPG-2-6/psycopg2-2.6.1.tar.gz && tar xvzf psycopg2-2.6.1.tar.gz && rm tar xvzf psycopg2-2.6.1.tar.gz && cd psycopg2-2.6.1 && python setup.py install --user build_ext --pg-config $POSTGRES_BUILD_DIR/bin/pg_config
}

function comparison_test() {
    postgres_run_tests
    sqlite_test
    csv_test
    numpy_test
    numpymmap_test
    monetdbembedded_test
    monetdbmapi_test
    monetdbpyapi_test
    monetdbpyapimap_test
    monetdbrapi_test
}

function comparison_graph() {
    python $PYAPI_GRAPHFILE "SAVE" "Identity" "postgres:postgres_identity.tsv" "sqlitemem:sqlitemem_identity.tsv" "sqlitedb:sqlitedb_identity.tsv" "csv:csv_identity.tsv" "numpy:numpy_identity.tsv" "numpymmap:numpymmap_identity.tsv" "monetdbembedded:monetdbembedded_identity.tsv" "monetdbmapi:monetdbmapi_identity.tsv" "monetdbpyapi:monetdbpyapi_identity.tsv" "monetdbpyapimap:monetdbpyapimap_identity.tsv" "monetdbrapi:monetdbrapi_identity.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Identity [Fast Only]" "numpy:numpy_identity.tsv" "numpymmap:numpymmap_identity.tsv" "monetdbembedded:monetdbembedded_identity.tsv" "monetdbpyapi:monetdbpyapi_identity.tsv" "monetdbpyapimap:monetdbpyapimap_identity.tsv" "monetdbrapi:monetdbrapi_identity.tsv"

    python $PYAPI_GRAPHFILE "SAVE" "Square Root" "postgres:postgres_sqroot.tsv" "sqlitemem:sqlitemem_sqroot.tsv" "sqlitedb:sqlitedb_sqroot.tsv" "csv:csv_sqroot.tsv" "numpy:numpy_sqroot.tsv" "numpymmap:numpymmap_sqroot.tsv" "monetdbembedded:monetdbembedded_sqroot.tsv" "monetdbmapi:monetdbmapi_sqroot.tsv" "monetdbpyapi:monetdbpyapi_sqroot.tsv" "monetdbpyapimap:monetdbpyapimap_sqroot.tsv" "monetdbrapi:monetdbrapi_sqroot.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Square Root [Fast Only]" "numpy:numpy_sqroot.tsv" "numpymmap:numpymmap_sqroot.tsv" "monetdbembedded:monetdbembedded_sqroot.tsv" "monetdbpyapi:monetdbpyapi_sqroot.tsv" "monetdbpyapimap:monetdbpyapimap_sqroot.tsv" "monetdbrapi:monetdbrapi_sqroot.tsv"

}
