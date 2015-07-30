

# The base directory of testing, a new folder is created in this base directory [$PYAPI_TEST_DIR], and everything is done in that new folder
export PYAPI_BASE_DIR=$HOME
# The terminal to start mserver with, examples are 'gnome-terminal', 'xterm', 'konsole'
export TERMINAL=x-terminal-emulator
export MSERVER_PORT=49979
# A command that tests if the mserver is still running (used to find out when the shutting down of mserver is completed)
export MSERVERTEST='netstat -ant | grep "127.0.0.1:$MSERVER_PORT.*LISTEN">/dev/null'
# Testing parameters
# Input test (zero copy vs copy)
# The input sizes to test (in MB)
export INPUT_TESTING_SIZES="0.1 1 10 100 1000"
# Amount of tests to run for each size
export INPUT_TESTING_NTESTS=10

# Output test (zero copy vs copy)
# The output sizes to test (in MB)
export OUTPUT_TESTING_SIZES="0.1 1 10 100 1000"
# Amount of tests to run for each size
export OUTPUT_TESTING_NTESTS=10

# String tests
# Strings of the same length (mb, length)
export STRINGSAMELENGTH_TESTING_SIZES="(100,1) (100,10) (100,100) (100,1000) (100,1000) (100,10000) (100,100000)"
export STRINGSAMELENGTH_TESTING_NTESTS=10
# Extreme length string testing (all strings have length 1 except for one string, which has EXTREME length)
# Arguments are (Extreme Length, String Count)
export STRINGEXTREMELENGTH_TESTING_SIZES="(10,100000) (100,100000) (1000,100000) (10000,100000)"
export STRINGEXTREMELENGTH_TESTING_NTESTS=10
# Check Unicode vs Always Unicode (ASCII) (mb, length)
export STRINGUNICODE_TESTING_SIZES="(0.1,10) (1,10) (10,10) (0.1,100) (1,100) (10,100) (100,100) (1000,100)"
export STRINGUNICODE_TESTING_NTESTS=10

# Multithreading tests
export MULTITHREADING_NR_THREADS="1 2 3 4 5 6 7 8"
export MULTITHREADING_TESTING_SIZES="1"
#amount of tests for each thread
export MULTITHREADING_TESTING_NTESTS=1

# Quantile speedtest
# The input sizes to test (in MB)
export QUANTILE_TESTING_SIZES="0.1 1 10 100 1000"
# Amount of tests to run for each size
export QUANTILE_TESTING_NTESTS=10


# You probably don't need to change these
export PYAPI_TEST_DIR=$PYAPI_BASE_DIR/monetdb_pyapi_test
export PYAPI_MONETDB_DIR=$PYAPI_TEST_DIR/MonetDB-pyapi
export PYAPI_BUILD_DIR=$PYAPI_TEST_DIR/build
export PYAPI_OUTPUT_DIR=$PYAPI_TEST_DIR/output
# PyAPI TAR url
export PYAPI_TAR_URL=http://dev.monetdb.org/hg/MonetDB/archive/pyapi.tar.gz

# Used for downloading the python-monetdb connector (import monetdb.sql)
export PYTHON_MONETDB_CONNECTOR_VERSION=11.19.3.2
export PYTHON_MONETDB_DIR=python-monetdb-$PYTHON_MONETDB_CONNECTOR_VERSION
export PYTHON_MONETDB_FILE=python-monetdb-$PYTHON_MONETDB_CONNECTOR_VERSION.tar.gz
export PYTHON_MONETDB_URL=https://pypi.python.org/packages/source/p/python-monetdb/$PYTHON_MONETDB_FILE

# Python testfile location
export PYAPI_TESTFILE=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi/Benchmarks/monetdb_testing.py
# Graph file location
export PYAPI_GRAPHFILE=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi/Benchmarks/graph.py

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

function pyapi_build {
    echo "Making directory $PYAPI_TEST_DIR."
    mkdir $PYAPI_TEST_DIR && cd $PYAPI_TEST_DIR
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
    wget $PYAPI_TAR_URL && tar xvzf pyapi.tar.gz && cd $PYAPI_MONETDB_DIR && ./bootstrap && ./configure prefix=$PYAPI_BUILD_DIR && make -j install
    if [ $? -ne 0 ]; then
        echo "Failed to download and install MonetDB. Exiting..."
        return 1
    fi
}

function pyapi_run_single_test() {
    echo "Beginning Test $1"
    if [ $SETSID -eq 1 ]; then
        $TERMINAL $PYAPI_BUILD_DIR/bin/mserver5 --set mapi_port=$MSERVER_PORT --set embedded_py=true --set enable_pyverbose=true --set pyapi_benchmark_output=$PYAPI_OUTPUT_DIR/temp_output.tsv $2 && python $PYAPI_TESTFILE $3 $4 $5 $MSERVER_PORT $6 && killall mserver5
    else
        $TERMINAL -e "$PYAPI_BUILD_DIR/bin/mserver5  --set mapi_port=$MSERVER_PORT --set embedded_py=true --set enable_pyverbose=true --set pyapi_benchmark_output=$PYAPI_OUTPUT_DIR/temp_output.tsv $2" && python $PYAPI_TESTFILE $3 $4 $5 $MSERVER_PORT $6 && killall mserver5
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

function pyapi_test_input {
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


function pyapi_test_input_null {
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

function pyapi_test_output {
    pyapi_run_single_test "Output Testing (Zero Copy)" "" "OUTPUT" output_zerocopy "$OUTPUT_TESTING_NTESTS" "$OUTPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "Output Testing (Copy)" "--set disable_pyzerocopyoutput=true" "OUTPUT" output_copy "$OUTPUT_TESTING_NTESTS" "$OUTPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_samelength {
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

function pyapi_test_string_extreme {
    pyapi_run_single_test "String Testing (NPY_OBJECT, Extreme Length)" "" "STRING_EXTREMELENGTH" string_extremelength_npyobject "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Extreme Length)" "--set enable_numpystringarray=true" "STRING_EXTREMELENGTH" string_extremelength_npystring "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_unicode_ascii {
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

function pyapi_test_bytearray_vs_string {
    pyapi_run_single_test "String Testing (ByteArray Object)" "" "STRING_SAMELENGTH" string_bytearrayobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (String Object)" "--set disable_bytearray=true" "STRING_SAMELENGTH" string_stringobject "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_threads {
    rm multithreading.tsv
    for thread in $MULTITHREADING_NR_THREADS
    do
        pyapi_run_single_test "Multithreading ($thread Threads)" "--forcemito --set gdk_nr_threads=$thread" "FACTORIAL-$thread" multithreading "$MULTITHREADING_TESTING_NTESTS" "$MULTITHREADING_TESTING_SIZES"
        if [ $? -ne 0 ]; then
            return 1
        fi
    done
}

function pyapi_test_quantile {
    echo "Beginning Quantile Testing (Python vs R vs MonetDB)"
    pyapi_run_single_test "Quantile Testing (Python)" "" "PQUANTILE" quantile_python "$QUANTILE_TESTING_NTESTS" "$QUANTILE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Quantile Testing (R)" "--set embedded_r=true" "RQUANTILE" quantile_r "$QUANTILE_TESTING_NTESTS" "$QUANTILE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Quantile Testing (MonetDB)" "" "QUANTILE" quantile_monetdb "$QUANTILE_TESTING_NTESTS" "$QUANTILE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_run_tests {
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

function pyapi_graph {
    python $PYAPI_GRAPHFILE "SAVE" "Input (Both)" "-xlog" "Zero Copy:input_zerocopy.tsv" "Copy:input_copy.tsv" "Zero Copy (Null):input_zerocopy_null.tsv" "Copy (Null):input_copy_null.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input" "-xlog" "Zero Copy:input_zerocopy.tsv" "Copy:input_copy.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input-Null" "-xlog" "Zero Copy:input_zerocopy_null.tsv" "Copy:input_copy_null.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Input-Map" "-xlog" "Zero Copy:input_zerocopy.tsv" "Zero Copy (Map):input_zerocopy_map.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Output" "-xlog" "Zero Copy:output_zerocopy.tsv" "Copy:output_copy.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "String Samelength" "-xlog" "Numpy Object:string_samelength_npyobject.tsv" "Numpy String:string_samelength_npystring.tsv"
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
    python $PYAPI_GRAPHFILE "SAVE" "String Samelength y-log" "-xlog" "-ylog" "Numpy Object:string_samelength_npyobject.tsv" "Numpy String:string_samelength_npystring.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "String Extremelength y-log" "-xlog" "-ylog" "Numpy Object:string_extremelength_npyobject.tsv" "Numpy String:string_extremelength_npystring.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (ASCII) y-log" "-xlog" "-ylog" "Check Unicode:string_unicode_ascii_check.tsv" "Always Unicode:string_unicode_ascii_always.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (Extreme) y-log" "-xlog" "-ylog" "Check Unicode:string_unicode_extreme_check.tsv" "Always Unicode:string_unicode_extreme_always.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "ByteArrayObject vs StringObject y-log" "-xlog" "-ylog" "Byte Array Object:string_bytearrayobject.tsv" "String Object:string_stringobject.tsv"
    python $PYAPI_GRAPHFILE "SAVE" "Quantile Speedtest y-log" "-xlog" "-ylog" "Python:quantile_python.tsv" "R:quantile_r.tsv" "MonetDB:quantile_monetdb.tsv"
}


function pyapi_cleanup {
    read -p "Finished testing, would you like me to remove the test directory $PYAPI_TEST_DIR and everything in it? (y/n):  " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $PYAPI_TEST_DIR
    fi
    return 0
}

function pyapi_test {
    if [ -d $PYAPI_TEST_DIR ]; then
        read -p "Directory $PYAPI_TEST_DIR already exists, skip the building and continue to testing? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            read -p "Should we delete the directory $PYAPI_TEST_DIR and rebuild everything? WARNING: This will delete everything in the directory $PYAPI_TEST_DIR. (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                read -p "Are you absolutely sure you want to delete everything in $PYAPI_TEST_DIR? (y/n): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf $PYAPI_TEST_DIR
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

