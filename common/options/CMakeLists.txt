#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(OPTIONS_OBJECTS monet_options.c monet_options.h monet_getopt.h)

add_library(moptions_headers INTERFACE)
target_include_directories(moptions_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

add_library(moptions STATIC ${OPTIONS_OBJECTS})
set_target_properties(moptions PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(moptions PUBLIC moptions_headers)

install(FILES monet_options.h DESTINATION ${INCLUDEDIR}/monetdb)
