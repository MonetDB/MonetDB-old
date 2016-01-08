#!/bin/bash

if [ -z "$1" ]; then
	echo Please provide a location for the MonetDB installation
	exit 1;
fi

export MONETDB_HOME=$1

cd ../../
## Bootstrap
sh bootstrap
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

## Configure
mkdir BUILD
cd BUILD
../configure --prefix=$MONETDB_HOME \
--enable-embedded --enable-embedded-java \
--disable-fits --disable-geom --disable-rintegration --disable-gsl --disable-netcdf \
--disable-merocontrol --disable-odbc --disable-console --disable-microhttpd \
--without-perl --without-python2 --without-python3 --without-rubygem --without-unixodbc \
--without-samtools --without-sphinxclient --without-geos --without-samtools --without-readline \
--enable-optimize --enable-silent-rules --disable-assert --disable-strict --disable-int128
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

## Build
make -j
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

## Combine libraries
OFILES=`find common clients/mapilib/ gdk monetdb5/mal monetdb5/modules monetdb5/optimizer sql tools/embedded java/embedded -name "*.o" | tr "\n" " "`
gcc -shared -o libmonetdb5.dylib $OFILES -lpthread -lpcre -lbz2 -llzma -lcurl -lz -liconv

## Copy the single lib
cp libmonetdb5.dylib $MONETDB_HOME/lib

## Build embedded Java and test
cd ../java/embedded
mvn clean install
