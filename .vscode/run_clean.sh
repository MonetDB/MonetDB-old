#!/bin/bash

FULL_PATH_TO_SCRIPT=`dirname ${0}`

cd `cygpath ${FULL_PATH_TO_SCRIPT}` && cd ..


# Empty the build directory. 
rm -rf build/*

# Remove all generated files outside of the build directory

find . -regex ".*\.am$\|.*\.msc$\|.*\.lst$\|.*\.orig$\|.*cheader\.text\.h" | grep -v rules.msc | xargs rm
