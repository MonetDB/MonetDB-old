#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CMAKE_CURRENT_BINARY_DIR} ../common/options ../common/utils ../common/stream ../gdk
					../monetdb5/mal ../monetdb5/modules/kernel ../monetdb5/modules/atoms ../monetdb5/modules/mal
					../monetdb5/optimizer ../sql/backends/monet5 ../sql/include ../sql/common ../sql/server
					../sql/storage ../sql/storage/bat)

add_definitions(${COMPILER_OPTION}DLIBGDK ${COMPILER_OPTION}DLIBMONETDB5 ${COMPILER_OPTION}DLIBSQL
				${COMPILER_OPTION}DLIBEMBEDDED ${COMPILER_OPTION}DHAVE_EMBEDDED)

bison_target(sqlparserlite ../sql/server/sql_parser.y ${CMAKE_CURRENT_BINARY_DIR}/sql_sqlparserlite.tab.c
			 COMPILE_FLAGS "-d -p sql -r all" DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/sql_sqlparserlite.tab.h)
add_library(bisonlite_obj OBJECT ${BISON_sqlparserlite_OUTPUTS})
set_target_properties(bisonlite_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(COMPILER_Wnounreachablecode)
	target_compile_options(bisonlite_obj PRIVATE -Wno-unreachable-code)
endif()

set(MAL_SCRIPTS_LIST
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/mmath.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/bat5.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/batExtensions.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/algebra.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/orderidx.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/group.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/aggr.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mkey.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/blob.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/str.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/mtime.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/streams.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/batmmath.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/batmtime.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/batstr.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/pcre.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/bbp.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/manifold.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mat.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/inspect.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/language.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/tablet.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/sample.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/optimizer/optimizer.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/iterator.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/language.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/01_batcalc.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/01_calc.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql.mal" # sql.mal MUST be here
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sqlcatalog.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_transaction.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_decimal.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_rank.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_aggr_bte.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_aggr_sht.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_aggr_int.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_aggr_lng.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_aggr_flt.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_aggr_dbl.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_inspect.mal"
	"${CMAKE_SOURCE_DIR}/sql/backends/monet5/sql_generator.mal")

set(SQL_SCRIPTS_LIST
	"${CMAKE_SOURCE_DIR}/sql/scripts/09_like.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/10_math.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/11_times.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/13_date.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/16_tracelog.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/17_temporal.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/18_index.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/20_vacuum.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/21_dependency_views.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/25_debug.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/39_analytics.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/51_sys_schema_extension.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/70_storagemodel.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/71_statistics.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/90_generator.sql"
	"${CMAKE_SOURCE_DIR}/sql/scripts/99_system.sql")

build_embedded_mal_scripts(mal_inline malModules --remove-command-comments "${MAL_SCRIPTS_LIST}")
build_embedded_sql_scripts(createdb_inline1 "${SQL_SCRIPTS_LIST}")

set(MONETDBLITE_TRANSLATION_UNITS
	monetdb_embedded.c
	../common/options/monet_options.c
	../common/stream/stream.c
	../common/utils/mutils.c
	../common/utils/revision.c
	../gdk/gdk_aggr.c
	../gdk/gdk_align.c
	../gdk/gdk_analytic_bounds.c
	../gdk/gdk_analytic_func.c
	../gdk/gdk_atoms.c
	../gdk/gdk_bat.c
	../gdk/gdk_batop.c
	../gdk/gdk_bbp.c
	../gdk/gdk_calc.c
	../gdk/gdk_cand.c
	../gdk/gdk_cross.c
	../gdk/gdk_delta.c
	../gdk/gdk_firstn.c
	../gdk/gdk_group.c
	../gdk/gdk_hash.c
	../gdk/gdk_heap.c
	../gdk/gdk_imprints.c
	../gdk/gdk_join.c
	../gdk/gdk_logger.c
	../gdk/gdk_orderidx.c
	../gdk/gdk_posix.c
	../gdk/gdk_project.c
	../gdk/gdk_qsort.c
	../gdk/gdk_sample.c
	../gdk/gdk_search.c
	../gdk/gdk_select.c
	../gdk/gdk_ssort.c
	../gdk/gdk_storage.c
	../gdk/gdk_string.c
	../gdk/gdk_system.c
	../gdk/gdk_tm.c
	../gdk/gdk_unique.c
	../gdk/gdk_utils.c
	../gdk/gdk_value.c
	../monetdb5/mal/mal.c
	../monetdb5/mal/mal_atom.c
	../monetdb5/mal/mal_builder.c
	../monetdb5/mal/mal_client.c
	../monetdb5/mal/mal_dataflow.c
	../monetdb5/mal/mal_embedded.c
	../monetdb5/mal/mal_exception.c
	../monetdb5/mal/mal_function.c
	../monetdb5/mal/mal_import.c
	../monetdb5/mal/mal_instruction.c
	../monetdb5/mal/mal_interpreter.c
	../monetdb5/mal/mal_linker.c
	../monetdb5/mal/mal_listing.c
	../monetdb5/mal/mal_module.c
	../monetdb5/mal/mal_namespace.c
	../monetdb5/mal/mal_parser.c
	../monetdb5/mal/mal_resolve.c
	../monetdb5/mal/mal_resource.c
	../monetdb5/mal/mal_runtime.c
	../monetdb5/mal/mal_scenario.c
	../monetdb5/mal/mal_session.c
	../monetdb5/mal/mal_stack.c
	../monetdb5/mal/mal_type.c
	../monetdb5/mal/mal_utils.c
	../monetdb5/modules/kernel/aggr.c
	../monetdb5/modules/atoms/streams.c
	../monetdb5/modules/kernel/algebra.c
	../monetdb5/modules/kernel/bat5.c
	../monetdb5/modules/mal/batcalc.c
	../monetdb5/modules/mal/batExtensions.c
	../monetdb5/modules/kernel/batmmath.c
	../monetdb5/modules/kernel/batstr.c
	../monetdb5/modules/atoms/blob.c
	../monetdb5/modules/mal/bbp.c
	../monetdb5/modules/mal/calc.c
	../monetdb5/modules/kernel/group.c
	../monetdb5/modules/mal/inspect.c
	../monetdb5/modules/mal/iterator.c
	../monetdb5/modules/mal/language.c
	../monetdb5/modules/mal/manifold.c
	../monetdb5/modules/mal/mat.c
	../monetdb5/modules/mal/mkey.c
	../monetdb5/modules/kernel/mmath.c
	../monetdb5/modules/atoms/mtime.c
	../monetdb5/modules/atoms/mtime_analytic.c
	../monetdb5/modules/atoms/strptime.c
	../monetdb5/modules/mal/orderidx.c
	../monetdb5/modules/mal/pcre.c
	../monetdb5/modules/mal/projectionpath.c
	../monetdb5/modules/mal/sample.c
	../monetdb5/modules/atoms/str.c
	../monetdb5/modules/mal/tablet.c
	../monetdb5/optimizer/opt_aliases.c
	../monetdb5/optimizer/opt_candidates.c
	../monetdb5/optimizer/opt_coercion.c
	../monetdb5/optimizer/opt_commonTerms.c
	../monetdb5/optimizer/opt_constants.c
	../monetdb5/optimizer/opt_costModel.c
	../monetdb5/optimizer/opt_dataflow.c
	../monetdb5/optimizer/opt_deadcode.c
	../monetdb5/optimizer/opt_emptybind.c
	../monetdb5/optimizer/opt_evaluate.c
	../monetdb5/optimizer/opt_garbageCollector.c
	../monetdb5/optimizer/opt_generator.c
	../monetdb5/optimizer/opt_inline.c
	../monetdb5/optimizer/opt_macro.c
	../monetdb5/optimizer/opt_matpack.c
	../monetdb5/optimizer/opt_mergetable.c
	../monetdb5/optimizer/opt_mitosis.c
	../monetdb5/optimizer/opt_multiplex.c
	../monetdb5/optimizer/opt_pipes.c
	../monetdb5/optimizer/opt_prelude.c
	../monetdb5/optimizer/opt_profiler.c
	../monetdb5/optimizer/opt_projectionpath.c
	../monetdb5/optimizer/opt_pushselect.c
	../monetdb5/optimizer/opt_remap.c
	../monetdb5/optimizer/opt_reorder.c
	../monetdb5/optimizer/opt_support.c
	../monetdb5/optimizer/opt_wrapper.c
	../monetdb5/optimizer/optimizer.c
	../sql/backends/monet5/mal_backend.c
	../sql/backends/monet5/rel_bin.c
	../sql/backends/monet5/sql.c
	../sql/backends/monet5/sql_assert.c
	../sql/backends/monet5/sql_bat2time.c
	../sql/backends/monet5/sql_cast.c
	../sql/backends/monet5/sql_cat.c
	../sql/backends/monet5/sql_datetrunc.c
	../sql/backends/monet5/sql_execute.c
	../sql/backends/monet5/sql_fround.c
	../sql/backends/monet5/sql_gencode.c
	../sql/backends/monet5/sql_generator.c
	../sql/backends/monet5/sql_optimizer.c
	../sql/backends/monet5/sql_orderidx.c
	../sql/backends/monet5/sql_rank.c
	../sql/backends/monet5/sql_result.c
	../sql/backends/monet5/sql_round.c
	../sql/backends/monet5/sql_scenario.c
	../sql/backends/monet5/sql_statement.c
	../sql/backends/monet5/sql_statistics.c
	../sql/backends/monet5/sql_transaction.c
	../sql/backends/monet5/sql_upgrades.c
	../sql/backends/monet5/sql_user.c
	../sql/common/sql_backend.c
	../sql/common/sql_changeset.c
	../sql/common/sql_hash.c
	../sql/common/sql_keyword.c
	../sql/common/sql_list.c
	../sql/common/sql_mem.c
	../sql/common/sql_stack.c
	../sql/common/sql_string.c
	../sql/common/sql_types.c
	../sql/server/rel_distribute.c
	../sql/server/rel_dump.c
	../sql/server/rel_exp.c
	../sql/server/rel_optimizer.c
	../sql/server/rel_partition.c
	../sql/server/rel_planner.c
	../sql/server/rel_prop.c
	../sql/server/rel_psm.c
	../sql/server/rel_rel.c
	../sql/server/rel_remote.c
	../sql/server/rel_schema.c
	../sql/server/rel_select.c
	../sql/server/rel_semantic.c
	../sql/server/rel_sequence.c
	../sql/server/rel_trans.c
	../sql/server/rel_unnest.c
	../sql/server/rel_updates.c
	../sql/server/rel_xml.c
	../sql/server/sql_atom.c
	../sql/server/sql_datetime.c
	../sql/server/sql_decimal.c
	../sql/server/sql_env.c
	../sql/server/sql_mvc.c
	../sql/server/sql_privileges.c
	../sql/server/sql_qc.c
	../sql/server/sql_query.c
	../sql/server/sql_scan.c
	../sql/server/sql_semantic.c
	../sql/server/sql_symbol.c
	../sql/storage/bat/bat_logger.c
	../sql/storage/bat/bat_storage.c
	../sql/storage/bat/bat_table.c
	../sql/storage/bat/bat_utils.c
	../sql/storage/bat/res_table.c
	../sql/storage/sql_catalog.c
	../sql/storage/store.c
	../sql/storage/store_dependency.c
	../sql/storage/store_sequence.c
	$<TARGET_OBJECTS:bisonlite_obj>)
if(${ENABLE_EMBEDDED} STREQUAL "OBJECT")
	add_library(monetdblite OBJECT ${MONETDBLITE_TRANSLATION_UNITS})
	set_target_properties(monetdblite PROPERTIES POSITION_INDEPENDENT_CODE ON)
	install(TARGETS monetdblite bisonlite_obj DESTINATION ${LOCALSTATEDIR}/monetdblite)
else()
	add_library(monetdblite SHARED ${MONETDBLITE_TRANSLATION_UNITS})
	target_link_libraries(monetdblite PRIVATE ${MATH_LIBRARIES} ${THREAD_LIBRARIES} ${DL_LIBRARIES})
	set_target_properties(monetdblite PROPERTIES VERSION ${MONETDB_VERSION} SOVERSION ${MONETDB_VERSION_MAJOR})
	install(TARGETS monetdblite DESTINATION ${LIBDIR})
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:monetdblite> DESTINATION ${LIBDIR} OPTIONAL)
	endif()
endif()

install(FILES
		monetdb_embedded.h
		../common/stream/stream.h
		../common/utils/mutils.h
		../gdk/gdk.h
		../gdk/gdk_atoms.h
		../gdk/gdk_bbp.h
		../gdk/gdk_calc.h
		../gdk/gdk_delta.h
		../gdk/gdk_posix.h
		../gdk/gdk_hash.h
		../gdk/gdk_system.h
		../gdk/gdk_utils.h
		../monetdb5/mal/mal.h
		../monetdb5/mal/mal_client.h
		../monetdb5/mal/mal_errors.h
		../monetdb5/mal/mal_exception.h
		../monetdb5/mal/mal_factory.h
		../monetdb5/mal/mal_function.h
		../monetdb5/mal/mal_instruction.h
		../monetdb5/mal/mal_interpreter.h
		../monetdb5/mal/mal_linker.h
		../monetdb5/mal/mal_listing.h
		../monetdb5/mal/mal_module.h
		../monetdb5/mal/mal_namespace.h
		../monetdb5/mal/mal_profiler.h
		../monetdb5/mal/mal_resolve.h
		../monetdb5/mal/mal_stack.h
		../monetdb5/mal/mal_type.h
		../monetdb5/modules/atoms/blob.h
		../monetdb5/modules/atoms/mtime.h
		../sql/include/sql_catalog.h
		../sql/include/sql_hash.h
		../sql/include/sql_list.h
		../sql/include/sql_mem.h
		../sql/include/sql_querytype.h
		../sql/storage/bat/res_table.h
		DESTINATION ${INCLUDEDIR}/monetdb)
