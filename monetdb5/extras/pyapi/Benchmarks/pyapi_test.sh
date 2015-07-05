

# The base directory of testing, a new folder is created in this base directory [$PYAPI_TEST_DIR], and everything is done in that new folder
export PYAPI_BASE_DIR=$HOME

# Testing parameters
# Input test (zero copy vs copy)
# The input sizes to test (in MB)
export INPUT_TESTING_SIZES="0.1 1 10 100"
# Amount of tests to run for each size
export INPUT_TESTING_NTESTS=10

# Output test (zero copy vs copy)
# The output sizes to test (in MB)
export OUTPUT_TESTING_SIZES="0.1 1 10 100 1000"
# Amount of tests to run for each size
export OUTPUT_TESTING_NTESTS=10

# String tests
# Strings of the same length (mb, length)
export STRINGSAMELENGTH_TESTING_SIZES="(10,1) (10,10) (10,100) (10,1000) (10,1000) (10, 10000) (10, 100000)"
export STRINGSAMELENGTH_TESTING_NTESTS=10
# Extreme length string testing (all strings have length 1 except for one string, which has EXTREME length)
# Arguments are (Extreme Length, String Count)
export STRINGEXTREMELENGTH_TESTING_SIZES="(10,100000) (100,100000) (1000,100000) (10000,100000)"
export STRINGEXTREMELENGTH_TESTING_NTESTS=10
# Check Unicode vs Always Unicode (ASCII) (mb, length)
export STRINGUNICODE_TESTING_SIZES="(0.1,10) (1,10) (10,10) (100,10)"
export STRINGUNICODE_TESTING_NTESTS=10

# Multithreading tests

# You probably don't need to change these
export PYAPI_TEST_DIR=$PYAPI_BASE_DIR/monetdb_pyapi_test
export PYAPI_MONETDB_DIR=$PYAPI_TEST_DIR/MonetDB-pyapi
export PYAPI_BUILD_DIR=$PYAPI_TEST_DIR/build
export PYAPI_OUTPUT_DIR=$PYAPI_TEST_DIR/output
# PyAPI TAR url
export PYAPI_TAR_URL=http://dev.monetdb.org/hg/MonetDB/archive/pyapi.tar.gz
# The terminal to start mserver with, examples are 'gnome-terminal', 'xterm', 'konsole'
export TERMINAL=x-terminal-emulator

# Used for downloading the python-monetdb connector (import monetdb.sql)
export PYTHON_MONETDB_CONNECTOR_VERSION=11.19.3.2
export PYTHON_MONETDB_DIR=python-monetdb-$PYTHON_MONETDB_CONNECTOR_VERSION
export PYTHON_MONETDB_FILE=python-monetdb-$PYTHON_MONETDB_CONNECTOR_VERSION.tar.gz
export PYTHON_MONETDB_URL=https://pypi.python.org/packages/source/p/python-monetdb/$PYTHON_MONETDB_FILE

# Import test location (a simple python script that returns an exit code of 1 if the import of a library fails)
export IMPORT_TESTFILE=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi/Benchmarks/importtest.py
# Python testfile location
export PYAPI_TESTFILE=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi/Benchmarks/monetdb_testing.py
# Graph file location
export PYAPI_GRAPHFILE=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi/Benchmarks/graph.py

function pyapi_build {
    echo "Making directory $PYAPI_TEST_DIR."
    mkdir $PYAPI_TEST_DIR && cd $PYAPI_TEST_DIR
    if [ $? -ne 0 ]; then
        echo "Failed to create testing directory, exiting..."
        return 1
    fi
    python $IMPORT_TESTFILE numpy
    if [ $? -ne 0 ]; then
        read -p "Failed to load library Numpy. Would you like me to try and install it for you? (y/n):  " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        pip install numpy && python $IMPORT_TESTFILE numpy
        if [$? -eq 0]; then
            echo "Successfully installed Numpy."
        else
            echo "Failed to install Numpy. Exiting..."
            return 1
        fi
    fi
    python $IMPORT_TESTFILE monetdb.sql
    if [ $? -ne 0 ]; then
        read -p "Failed to load library MonetDB Python connector. Would you like me to try and install it for you? (y/n):  " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        wget $PYTHON_MONETDB_URL && tar xvzf $PYTHON_MONETDB_FILE && cd $PYTHON_MONETDB_DIR && python setup.py install && python $IMPORT_TESTFILE monetdb.sql
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
    $TERMINAL -e "$PYAPI_BUILD_DIR/bin/mserver5 --set embedded_py=true $2" && python $PYAPI_TESTFILE $3 $4 $5 $6 && killall mserver5
    if [ $? -ne 0 ]; then
        echo "Failed Test $1"
        killall mserver5
        return 1
    fi
    for i in `seq 1 20`; do
        nc -z 127.0.0.1 50000
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
    pyapi_run_single_test "Input Testing (Zero Copy)" "" "INPUT" input_zerocopy.out "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Input Testing (Copy)" "--set disable_pyzerocopyinput=true" "INPUT" input_copy.out "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
    pyapi_run_single_test "Input Testing (Map)" "" "INPUT-MAP" input_zerocopy_map.out "$INPUT_TESTING_NTESTS" "$INPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_output {
    pyapi_run_single_test "Output Testing (Zero Copy)" "" "OUTPUT" output_zerocopy.out "$OUTPUT_TESTING_NTESTS" "$OUTPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "Output Testing (Copy)" "--set disable_pyzerocopyoutput=true" "OUTPUT" output_copy.out "$OUTPUT_TESTING_NTESTS" "$OUTPUT_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_samelength {
    pyapi_run_single_test "String Testing (NPY_OBJECT, Same Length)" "" "STRING_SAMELENGTH" string_samelength_npyobject.out "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Same Length)" "--set enable_numpystringarray=true" "STRING_SAMELENGTH" string_samelength_npystring.out "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_extreme {
    pyapi_run_single_test "String Testing (NPY_OBJECT, Extreme Length)" "" "STRING_EXTREMELENGTH" string_extremelength_npyobject.out "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (NPY_STRING, Extreme Length)" "--set enable_numpystringarray=true" "STRING_EXTREMELENGTH" string_extremelength_npystring.out "$STRINGEXTREMELENGTH_TESTING_NTESTS" "$STRINGEXTREMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_string_unicode_ascii {
    pyapi_run_single_test "String Testing (Check Unicode, ASCII)" "" "STRING_SAMELENGTH" string_unicode_ascii_check.out "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Always Unicode, ASCII)" "--set enable_alwaysunicode=true" "STRING_SAMELENGTH" string_unicode_ascii_always.out "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Check Unicode, Extreme)" "" "STRING_EXTREMEUNICODE" string_unicode_extreme_check.out "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (Always Unicode, Extreme)" "" "STRING_EXTREMEUNICODE" string_unicode_extreme_always.out "$STRINGUNICODE_TESTING_NTESTS" "$STRINGUNICODE_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi
}

function pyapi_test_bytearray_vs_string {
    pyapi_run_single_test "String Testing (ByteArray Object)" "" "STRING_SAMELENGTH" string_bytearrayobject.out "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
    if [ $? -ne 0 ]; then
        return 1
    fi

    pyapi_run_single_test "String Testing (String Object)" "--set disable_bytearray=true" "STRING_SAMELENGTH" string_stringobject.out "$STRINGSAMELENGTH_TESTING_NTESTS" "$STRINGSAMELENGTH_TESTING_SIZES"
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
    pyapi_test_output
    pyapi_test_string_samelength
    pyapi_test_string_extreme
    pyapi_test_string_unicode_ascii
    pyapi_test_bytearray_vs_string
}

function pyapi_graph {
    python $PYAPI_GRAPHFILE "SAVE" "Input" "Table Size" "Zero Copy:input_zerocopy.out" "Copy:input_copy.out" "Zero Copy (Map):input_zerocopy_map.out"
    python $PYAPI_GRAPHFILE "SAVE" "Output" "Output Size" "Zero Copy:output_zerocopy.out" "Copy:output_copy.out"
    python $PYAPI_GRAPHFILE "SAVE" "String Samelength" "String Length" "Numpy Object:string_samelength_npyobject.out" "Numpy String:string_samelength_npystring.out"
    python $PYAPI_GRAPHFILE "SAVE" "String Extremelength" "Extreme Length" "Numpy Object:string_extremelength_npyobject.out" "Numpy String:string_extremelength_npystring.out"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (ASCII)" "Table Size" "Check Unicode:string_unicode_ascii_check.out" "Always Unicode:string_unicode_ascii_always.out"
    python $PYAPI_GRAPHFILE "SAVE" "Unicode Check vs Always Unicode (Extreme)" "Table Size" "Check Unicode:string_unicode_extreme_check.out" "Always Unicode:string_unicode_extreme_always.out"
    python $PYAPI_GRAPHFILE "SAVE" "ByteArrayObject vs StringObject" "String Length" "Byte Array Object:string_bytearrayobject.out" "String Object:string_stringobject.out"
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
                else
                    return 1
                fi
            else
                return 1
            fi
        fi
    else
        pyapi_build
    fi

    pyapi_run_tests
    pyapi_graph
    pyapi_cleanup
}
