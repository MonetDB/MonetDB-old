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
add_subdirectory(generator)

add_library(sqlmonet5_headers INTERFACE)
target_include_directories(sqlmonet5_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

set(SQL_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers sql_headers
					   kernel_headers optimizer_headers sqlserver store batstore sqlcommon)
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
			sql_session.c sql_session.h
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
			sql_rank.c sql_rank.h sql_subquery.c sql_subquery.h)
target_link_libraries(sql PUBLIC sqlmonet5_headers PRIVATE ${SQL_LINK_LIBRARIES})
set_target_properties(sql PROPERTIES OUTPUT_NAME _sql)
target_compile_definitions(sql PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)

install(TARGETS sql DESTINATION ${LIBDIR}/monetdb5)
install(FILES
		sql_aggr_bte.mal sql_aggr_flt.mal sql_aggr_dbl.mal sql_aggr_int.mal sql_aggr_lng.mal
		sql_aggr_sht.mal sql_decimal.mal sql_inspect.mal wlr.mal
		sql_subquery.mal sql_rank.mal sqlcatalog.mal sql_transaction.mal sql.mal
		DESTINATION ${LIBDIR}/monetdb5)
install(FILES 40_sql.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:sql> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()

if(HAVE_HGE)
	install(FILES sql_hge.mal sql_aggr_hge.mal sql_decimal_hge.mal sql_rank_hge.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 41_sql_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
