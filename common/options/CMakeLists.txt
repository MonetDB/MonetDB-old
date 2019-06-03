#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../utils)

set(OPTIONS_OBJECTS monet_options.c monet_options.h monet_getopt.h)

if(WIN32)
	add_library(moptions SHARED ${OPTIONS_OBJECTS})
	target_compile_definitions(moptions PRIVATE LIBMOPTIONS)
	install(TARGETS moptions DESTINATION ${LIBDIR})
	install(FILES $<TARGET_PDB_FILE:moptions> DESTINATION ${LIBDIR} OPTIONAL)
else()
	add_library(moptions STATIC ${OPTIONS_OBJECTS})
	set_target_properties(moptions PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(moptions PRIVATE mutils)

install(FILES monet_options.h DESTINATION ${INCLUDEDIR}/monetdb)
