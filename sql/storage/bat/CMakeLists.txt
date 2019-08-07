#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_library(batstore_headers INTERFACE)
target_include_directories(batstore_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

if(HAVE_SQL)
	add_library(batstore STATIC
				bat_table.c bat_storage.c bat_logger.c bat_utils.c
				res_table.c
				bat_logger.h bat_storage.h bat_table.h bat_utils.h res_table.h)
	set_target_properties(batstore PROPERTIES POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(batstore PRIVATE LIBSQL LIBSTORE)
	target_link_libraries(batstore PRIVATE store_headers utils_headers stream_headers gdk_headers mal_headers
						  kernel_headers sql_headers sqlcommon_headers)
endif()
