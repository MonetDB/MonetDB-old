#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CMAKE_CURRENT_BINARY_DIR} ../include ../common ../storage ../../monetdb5/modules/atoms
					../../monetdb5/modules/kernel ../../monetdb5/mal ../../monetdb5/modules/atoms
					../../monetdb5/modules/mal ../../common/options ../../common/stream ../../common/utils
					../backends/monet5 ../server ../../gdk)

set(SQL_SCRIPTS_LIST1
	"${CMAKE_CURRENT_SOURCE_DIR}/09_like.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/10_math.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/11_times.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/12_url.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/13_date.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/14_inet.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/15_querylog.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/16_tracelog.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/17_temporal.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/18_index.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/20_vacuum.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/21_dependency_views.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/22_clients.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/23_skyserver.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/25_debug.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/26_sysmon.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/27_rejects.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/39_analytics.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/40_json.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/41_md5sum.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/45_uuid.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/46_profiler.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/51_sys_schema_extension.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/60_wlcr.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/70_storagemodel.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/71_statistics.sql"
	"${CMAKE_CURRENT_SOURCE_DIR}/90_generator.sql")
if(HAVE_HGE)
	list(APPEND SQL_SCRIPTS_LIST1
		"${CMAKE_CURRENT_SOURCE_DIR}/39_analytics_hge.sql"
		"${CMAKE_CURRENT_SOURCE_DIR}/40_json_hge.sql"
		"${CMAKE_CURRENT_SOURCE_DIR}/90_generator_hge.sql")
endif()

# These scripts loads after all custom installed SQL scripts
set(SQL_SCRIPTS_LIST2 "${CMAKE_CURRENT_SOURCE_DIR}/99_system.sql")

build_embedded_sql_scripts(createdb_inline1 "${SQL_SCRIPTS_LIST1}")
build_embedded_sql_scripts(createdb_inline2 "${SQL_SCRIPTS_LIST2}")

# The sql script header files should be in CMAKE_CURRENT_BINARY_DIR
add_library(sqlscripts STATIC sql_scripts.c sql_scripts.h)
set_target_properties(sqlscripts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlscripts PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
