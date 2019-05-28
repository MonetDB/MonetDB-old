#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../include ../common ../storage ../../monetdb5/modules/atoms ../../monetdb5/modules/kernel
					../../monetdb5/mal ../../monetdb5/modules/atoms ../../monetdb5/modules/mal ../../common/options
					../../common/stream ../../common/utils ../backends/monet5 ../server ../../gdk)

add_library(sqlscripts STATIC
			sql_scripts.c sql_scripts.h
			09_like.sql.h
			10_math.sql.h
			11_times.sql.h
			12_url.sql.h
			13_date.sql.h
			14_inet.sql.h
			15_querylog.sql.h
			16_tracelog.sql.h
			17_temporal.sql.h
			18_index.sql.h
			20_vacuum.sql.h
			21_dependency_views.sql.h
			22_clients.sql.h
			23_skyserver.sql.h
			25_debug.sql.h
			26_sysmon.sql.h
			27_rejects.sql.h
			39_analytics_hge.sql.h
			39_analytics.sql.h
			40_json_hge.sql.h
			40_json.sql.h
			41_md5sum.sql.h
			45_uuid.sql.h
			46_profiler.sql.h
			51_sys_schema_extension.sql.h
			60_wlcr.sql.h
			70_storagemodel.sql.h
			71_statistics.sql.h
			90_generator.sql.h
			90_generator_hge.sql.h
			99_system.sql.h)
set_target_properties(sqlscripts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlscripts PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
