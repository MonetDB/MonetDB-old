#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CMAKE_CURRENT_BINARY_DIR} ../../common/stream ../../common/utils ../../clients/mapilib ../../gdk
					../optimizer ../modules/mal ${CRYPTO_INCLUDE_DIR})

set(MAL_SCRIPTS_LIST )
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	list(APPEND MAL_SCRIPTS_LIST "${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mdb.mal")
endif()
list(APPEND MAL_SCRIPTS_LIST
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/alarm.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/mmath.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/streams.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/bat5.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/batExtensions.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/algebra.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/orderidx.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/status.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/groupby.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/group.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/aggr.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mkey.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/blob.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/str.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/mtime.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/color.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/url.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/uuid.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/json.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/json_util.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/inet.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/identifier.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/xml.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/batxml.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/batmmath.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/batmtime.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/batstr.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/batcolor.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/pcre.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/clients.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/bbp.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mal_io.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/manifold.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/factories.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/remote.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mat.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/inspect.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/manual.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/language.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/profiler.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/querylog.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/sysmon.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/tablet.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/sample.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/optimizer/optimizer.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/iterator.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/txtsim.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/tokenizer.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/mal_mapi.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/oltp.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/microbenchmark.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/wlc.mal")
if(HAVE_HGE)
	list(APPEND MAL_SCRIPTS_LIST
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/kernel/00_aggr_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_batcalc_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_calc_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_batExtensions_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_iterator_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_language_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_mkey_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/00_mal_mapi_hge.mal"
		"${CMAKE_SOURCE_DIR}/monetdb5/modules/atoms/00_json_hge.mal")
endif()
list(APPEND MAL_SCRIPTS_LIST
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/language.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/01_batcalc.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/modules/mal/01_calc.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/scheduler/run_adder.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/scheduler/run_isolate.mal"
	"${CMAKE_SOURCE_DIR}/monetdb5/scheduler/run_memo.mal")

build_embedded_mal_scripts(mal_inline malModules --no-remove-command-comments "${MAL_SCRIPTS_LIST}")

add_library(mal OBJECT
			mal.c mal.h
			mal_atom.c mal_atom.h
			mal_authorize.c mal_authorize.h
			mal_builder.c mal_builder.h
			mal_client.c mal_client.h
			mal_debugger.c mal_debugger.h
			mal_embedded.c mal_embedded.h
			mal_errors.h
			mal_exception.c mal_exception.h
			mal_factory.c mal_factory.h
			mal_function.c mal_function.h
			mal_import.c mal_import.h
			mal_runtime.c mal_runtime.h
			mal_instruction.c mal_instruction.h
			mal_resource.c mal_resource.h
			mal_interpreter.c mal_interpreter.h
			mal_dataflow.c mal_dataflow.h
			mal_linker.c mal_linker.h
			mal_listing.c mal_listing.h
			mal_module.c mal_module.h
			mal_namespace.c mal_namespace.h
			mal_parser.c mal_parser.h
			mal_profiler.c mal_profiler.h
			mal_resolve.c mal_resolve.h
			mal_scenario.c mal_scenario.h
			mal_session.c mal_session.h
			mal_stack.c mal_stack.h
			mal_type.c mal_type.h
			mal_utils.c mal_utils.h
			mal_private.h)
target_compile_definitions(mal PRIVATE LIBMONETDB5)
set_target_properties(mal PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:mal> PARENT_SCOPE)

install(FILES
		mal.h
		mal_client.h
		mal_errors.h
		mal_exception.h
		mal_factory.h
		mal_function.h
		mal_instruction.h
		mal_interpreter.h
		mal_listing.h
		mal_module.h
		mal_namespace.h
		mal_profiler.h
		mal_resolve.h
		mal_stack.h
		mal_type.h
		DESTINATION ${INCLUDEDIR}/monetdb)
