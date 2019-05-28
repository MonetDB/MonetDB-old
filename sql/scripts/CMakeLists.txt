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

set(SQL_SCRIPTS_LIST 09_like.sql 10_math.sql 11_times.sql 12_url.sql 13_date.sql 14_inet.sql 15_querylog.sql
	16_tracelog.sql 17_temporal.sql 18_index.sql 20_vacuum.sql 21_dependency_views.sql 22_clients.sql 23_skyserver.sql
	25_debug.sql 26_sysmon.sql 27_rejects.sql 39_analytics_hge.sql 39_analytics.sql 40_json_hge.sql 40_json.sql
	41_md5sum.sql 45_uuid.sql 46_profiler.sql 51_sys_schema_extension.sql 60_wlcr.sql 70_storagemodel.sql
	71_statistics.sql 90_generator.sql 90_generator_hge.sql 99_system.sql)

foreach(script IN LISTS SQL_SCRIPTS_LIST)
	execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/buildtools/scripts/sql2h.py" "-f"
					"${CMAKE_CURRENT_SOURCE_DIR}/${script}" "-t" "-o" "${CMAKE_CURRENT_BINARY_DIR}/${script}.h"
					RESULT_VARIABLE PY_SCRIPT_RC OUTPUT_QUIET ERROR_QUIET)
	if(NOT PY_SCRIPT_RC EQUAL 0)
		message(FATAL_ERROR "Could not generate ${script}.h file")
	endif()
endforeach()

# The sql script header files should be in CMAKE_CURRENT_BINARY_DIR
add_library(sqlscripts STATIC sql_scripts.c sql_scripts.h)
set_target_properties(sqlscripts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlscripts PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
