#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(NOT WIN32)
	add_subdirectory(vaults)
endif()
add_subdirectory(UDF)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ../../../gdk ../../include ../../common ../../storage ../../server
					../../../monetdb5/modules/atoms ../../../monetdb5/modules/kernel ../../../monetdb5/mal
					../../../monetdb5/modules/mal ../../../monetdb5/optimizer ../../../monetdb5/scheduler
					../../../monetdb5/extras/rapi ../../../common/stream ../../../common/utils ../../scripts)

set(SQL_MAL_SCRIPTS_LIST
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

execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/buildtools/scripts/mal2h.py"
				"${CMAKE_CURRENT_BINARY_DIR}/sql_mal_inline.h" ${SQL_MAL_SCRIPTS_LIST}
				RESULT_VARIABLE PY_SCRIPT_RC OUTPUT_QUIET)
if(NOT PY_SCRIPT_RC EQUAL 0)
	message(FATAL_ERROR "Could not generate sql_mal_inline.h file")
endif()

# Generate the list of the sql mal module names
set(SQL_MAL_NAMES "str sqlMalModules[MAXMODULES] = {")
foreach(script IN LISTS SQL_MAL_SCRIPTS_LIST)
	get_filename_component(SCRIPT_BASENAME "${script}" NAME)
	string(APPEND SQL_MAL_NAMES "\"${SCRIPT_BASENAME}\",")
endforeach()
string(APPEND SQL_MAL_NAMES "0};\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sql_mal_inline_names.h" "${SQL_MAL_NAMES}")

set(SQL_LINK_LIBRARIES sqlserver store batstore sqlcommon sqlscripts)
if(WIN32)
	list(APPEND SQL_LINK_LIBRARIES mapi monetdb5 gdk stream ${CRYPTO_LIBRARIES})
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
target_link_libraries(sql PRIVATE ${SQL_LINK_LIBRARIES})
set_target_properties(sql PROPERTIES OUTPUT_NAME _sql)
target_compile_definitions(sql PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)

install(TARGETS sql DESTINATION ${LIBDIR}/monetdb5)
install(FILES sql.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 40_sql.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
