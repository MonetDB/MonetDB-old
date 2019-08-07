#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_library(store_headers INTERFACE)
target_include_directories(store_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

add_subdirectory(bat)

if(HAVE_SQL)
	add_library(store STATIC
				store_dependency.c store_sequence.c
				store.c sql_catalog.c
				sql_storage.h store_dependency.h
				store_sequence.h)
	set_target_properties(store PROPERTIES POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(store PRIVATE LIBSQL LIBSTORE)
	target_link_libraries(store PUBLIC store_headers PRIVATE utils_headers stream_headers gdk_headers sql_headers
						  sqlcommon_headers)
endif()
