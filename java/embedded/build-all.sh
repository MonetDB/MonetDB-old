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
sh ../configure --prefix=$MONETDB_HOME --disable-fits --disable-netcdf --disable-gsl --disable-geom --disable-merocontrol --disable-odbc --disable-microhttpd --without-perl --without-python2 --without-python3 --without-rubygem --without-unixodbc --without-readline --enable-embedded --enable-embedded-java
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
## Build
make -j && make install
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
## Build embedded Java and test
cd ../java/embedded
mvn clean install
