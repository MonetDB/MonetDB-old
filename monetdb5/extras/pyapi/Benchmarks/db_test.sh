

export PYAPI_BASE_DIR=/export/scratch1/raasveld

export PYAPI_TEST_DIR=$PYAPI_BASE_DIR/monetdb_pyapi_test
export PYAPI_MONETDB_DIR=$PYAPI_TEST_DIR/MonetDB-pyapi
export PYAPI_SRC_DIR=$PYAPI_MONETDB_DIR/monetdb5/extras/pyapi
export PYAPI_BUILD_DIR=$PYAPI_TEST_DIR/build
export PYAPI_OUTPUT_DIR=$PYAPI_TEST_DIR/output

export POSTGRES_BASEDIR=$PYAPI_TEST_DIR/postgres
export POSTGRES_TEST_DIR=$POSTGRES_BASEDIR/postgres_test
export POSTGRES_BUILD_DIR=$POSTGRES_BASEDIR/build
export PGDATA=$POSTGRES_BASEDIR/postgres_data

export POSTGRES_VERSION=9.4.4
export POSTGRES_BASE=postgresql-$POSTGRES_VERSION
export POSTGRES_TAR_FILE=$POSTGRES_BASE.tar.gz
export POSTGRES_TAR_URL=https://ftp.postgresql.org/pub/source/v$POSTGRES_VERSION/$POSTGRES_TAR_FILE


function postgres_build() {
	cd $PYAPI_TEST_DIR
	wget $POSTGRES_TAR_URL && tar xvzf $POSTGRES_TAR_FILE && cd $POSTGRES_BASE && ./configure --prefix=$POSTGRES_BUILD_DIR --with-python && make && make install && $POSTGRES_BUILD_DIR/bin/initdb && $POSTGRES_BUILD_DIR/bin/createdb python_test
}

