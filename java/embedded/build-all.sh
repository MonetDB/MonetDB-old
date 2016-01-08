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
make -j clean install
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
## Build embedded Java and test
cd ../java/embedded
mvn clean install
