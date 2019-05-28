#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CMAKE_CURRENT_BINARY_DIR} ../../common/stream ../../common/utils ../../gdk ../mal ../modules/atoms
					../modules/kernel ../modules/mal ../optimizer ../scheduler)

set(MAL_SCRIPTS_LIST
	modules/mal/mdb.mal
	modules/kernel/alarm.mal
	modules/kernel/mmath.mal
	modules/atoms/streams.mal
	modules/kernel/bat5.mal
	modules/mal/batExtensions.mal
	modules/kernel/algebra.mal
	modules/mal/orderidx.mal
	modules/kernel/status.mal
	modules/mal/groupby.mal
	modules/kernel/group.mal
	modules/kernel/aggr.mal
	modules/mal/mkey.mal
	modules/atoms/blob.mal
	modules/atoms/str.mal
	modules/atoms/mtime.mal
	modules/atoms/color.mal
	modules/atoms/url.mal
	modules/atoms/uuid.mal
	modules/atoms/json.mal
	modules/mal/json_util.mal
	modules/atoms/inet.mal
	modules/atoms/identifier.mal
	modules/atoms/xml.mal
	modules/kernel/batmmath.mal
	modules/mal/batmtime.mal
	modules/kernel/batstr.mal
	modules/kernel/batcolor.mal
	modules/atoms/batxml.mal
	modules/mal/pcre.mal
	modules/mal/clients.mal
	modules/mal/bbp.mal
	modules/mal/mal_io.mal
	modules/mal/manifold.mal
	modules/mal/factories.mal
	modules/mal/remote.mal
	modules/mal/mat.mal
	modules/mal/inspect.mal
	modules/mal/manual.mal
	modules/mal/language.mal
	modules/mal/profiler.mal
	modules/mal/querylog.mal
	modules/mal/sysmon.mal
	modules/mal/tablet.mal
	modules/mal/sample.mal
	optimizer/optimizer.mal
	modules/mal/iterator.mal
	modules/mal/txtsim.mal
	modules/mal/tokenizer.mal
	modules/mal/mal_mapi.mal
	modules/mal/oltp.mal
	modules/kernel/microbenchmark.mal
	modules/mal/wlc.mal
	modules/kernel/00_aggr_hge.mal
	modules/mal/00_batcalc_hge.mal
	modules/mal/00_calc_hge.mal
	modules/mal/00_batExtensions_hge.mal
	modules/mal/00_iterator_hge.mal
	modules/mal/00_language_hge.mal
	modules/mal/00_mkey_hge.mal
	modules/mal/00_mal_mapi_hge.mal
	modules/atoms/00_json_hge.mal
	modules/mal/language.mal
	modules/mal/01_batcalc.mal
	modules/mal/01_calc.mal
	scheduler/run_adder.mal
	scheduler/run_isolate.mal
	scheduler/run_memo.mal)

foreach(script IN LISTS MAL_SCRIPTS_LIST)
	get_filename_component(SCRIPT_BASENAME "${script}" NAME)
	execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/buildtools/scripts/mal2h.py" "-f"
					"${CMAKE_SOURCE_DIR}/monetdb5/${script}" "-t" "-o" "${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_BASENAME}.h"
					RESULT_VARIABLE PY_SCRIPT_RC OUTPUT_QUIET)
	if(NOT PY_SCRIPT_RC EQUAL 0)
		message(FATAL_ERROR "Could not generate ${SCRIPT_BASENAME}.h file")
	endif()
endforeach()

add_library(malembedded OBJECT mal_embedded.c mal_embedded.h) # The mal script header files should be in CMAKE_CURRENT_BINARY_DIR
target_compile_definitions(malembedded PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)
set_target_properties(malembedded PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:malembedded> PARENT_SCOPE)
