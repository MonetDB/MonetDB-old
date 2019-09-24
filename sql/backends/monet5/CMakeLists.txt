#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_library(sqlmonet5_headers INTERFACE)
target_include_directories(sqlmonet5_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

if(HAVE_SQL)
	if(NOT WIN32)
		add_subdirectory(vaults)
	endif()
	add_subdirectory(UDF)

	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	set(SQL_MAL_SCRIPTS_LIST
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_sysmon.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sqlcatalog.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_transaction.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_decimal.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_rank.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_bte.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_sht.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_int.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_lng.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_flt.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_dbl.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_inspect.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/sql_generator.mal"
		"${CMAKE_CURRENT_SOURCE_DIR}/wlr.mal")
	if(HAVE_HGE)
		list(APPEND SQL_MAL_SCRIPTS_LIST
			"${CMAKE_CURRENT_SOURCE_DIR}/sql_hge.mal"
			"${CMAKE_CURRENT_SOURCE_DIR}/sql_decimal_hge.mal"
			"${CMAKE_CURRENT_SOURCE_DIR}/sql_rank_hge.mal"
			"${CMAKE_CURRENT_SOURCE_DIR}/sql_aggr_hge.mal"
			"${CMAKE_CURRENT_SOURCE_DIR}/sql_generator_hge.mal")
	endif()

	set(SQL_SCRIPTS_LIST1
		"${CMAKE_SOURCE_DIR}/sql/scripts/09_like.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/10_math.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/12_url.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/13_date.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/14_inet.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/15_querylog.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/16_tracelog.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/17_temporal.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/18_index.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/20_vacuum.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/21_dependency_views.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/22_clients.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/23_skyserver.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/25_debug.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/26_sysmon.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/27_rejects.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/39_analytics.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/40_json.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/41_md5sum.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/45_uuid.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/46_profiler.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/51_sys_schema_extension.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/60_wlcr.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/70_storagemodel.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/71_statistics.sql"
		"${CMAKE_SOURCE_DIR}/sql/scripts/90_generator.sql")
	if(HAVE_HGE)
		list(APPEND SQL_SCRIPTS_LIST1
			"${CMAKE_SOURCE_DIR}/sql/scripts/39_analytics_hge.sql"
			"${CMAKE_SOURCE_DIR}/sql/scripts/40_json_hge.sql"
			"${CMAKE_SOURCE_DIR}/sql/scripts/90_generator_hge.sql")
	endif()

	# These scripts loads after all custom installed SQL scripts
	set(SQL_SCRIPTS_LIST2 "${CMAKE_SOURCE_DIR}/sql/scripts/99_system.sql")

	build_embedded_mal_scripts(sql_mal_inline sqlMalModules --no-remove-command-comments "${SQL_MAL_SCRIPTS_LIST}")
	build_embedded_sql_scripts(createdb_inline1 "${SQL_SCRIPTS_LIST1}")
	build_embedded_sql_scripts(createdb_inline2 "${SQL_SCRIPTS_LIST2}")

	set(SQL_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
		sql_headers kernel_headers optimizer_headers sqlserver store batstore sqlcommon)
	if(WIN32)
		list(APPEND SQL_LINK_LIBRARIES mcrypt msabaoth stream mapi gdk monetdb5 ${CRYPTO_LIBRARIES})
	endif()

	add_library(sql MODULE
				sql.c sql.h
				mal_backend.c mal_backend.h
				sql_user.c sql_user.h
				sql_scenario.c sql_scenario.h
				sql_execute.c sql_execute.h
				sql_assert.c sql_assert.h
				sql_upgrades.c sql_upgrades.h
				rel_bin.c rel_bin.h
				sql_cat.c sql_cat.h
				sql_transaction.c sql_transaction.h
				sql_statement.c sql_statement.h
				sql_statistics.c sql_statistics.h
				sql_gencode.c sql_gencode.h
				sql_optimizer.c sql_optimizer.h
				sql_result.c sql_result.h
				sql_cast.c sql_cast.h
				sql_cast_impl_down_from_flt.h
				sql_cast_impl_int.h
				sql_cast_impl_up_to_flt.h
				sql_round.c sql_round_impl.h sql_bat2time.c
				sql_fround.c sql_fround_impl.h
				sql_orderidx.c sql_orderidx.h
				wlr.c wlr.h
				sql_datetrunc.c
				sql_rank.c sql_rank.h
				sql_generator.c sql_generator.h) # The mal script header files should be in CMAKE_CURRENT_BINARY_DIR
	target_link_libraries(sql PUBLIC sqlmonet5_headers PRIVATE ${SQL_LINK_LIBRARIES})
	set_target_properties(sql PROPERTIES OUTPUT_NAME _sql)
	target_compile_definitions(sql PRIVATE LIBSQL LIBSTORE)

	install(TARGETS sql DESTINATION ${LIBDIR}/monetdb5)
	install(FILES sql.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 40_sql.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
