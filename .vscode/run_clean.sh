#!/bin/bash

FULL_PATH_TO_SCRIPT=`dirname ${0}`

cd `cygpath ${FULL_PATH_TO_SCRIPT}` && cd ..


# Empty the build directory. 
rm -rf build/*

# Remove all generated files outside of the build directory
find . -path "NT/rules.msc" -prune -regex ".*\.am$\|.*\.msc$\|.*\.lst$\|.*\.orig$\|.*cheader\.text\.h" -exec rm \{\} +
