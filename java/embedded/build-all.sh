#!/bin/bash

if [ -z "$1" ]; then
	echo Please provide a location for the MonetDB installation
	exit 1;
fi

start_dir=`pwd`
source_main_dir=`pwd`/../../

cd $source_main_dir
## Bootstrap
sh bootstrap
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

## Configure
mkdir BUILD
cd BUILD
../configure --prefix=source_main_dir/BUILD/installation \
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
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

## Copy the single lib
mkdir -p $start_dir/src/main/resources/lib
cp libmonetdb5.dylib $start_dir/src/main/resources/lib/

## Build embedded Java and test
cd $start_dir
mvn clean install
