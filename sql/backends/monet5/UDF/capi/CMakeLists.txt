#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if("${ENABLE_CINTEGRATION}" STREQUAL "YES")
	if(NOT HAVE_PYTHON3)
		message(FATAL_ERROR "A valid Python 3 interpreter is required to compile the C UDFs module")
	elseif(WIN32)
		message(FATAL_ERROR "The C UDFs module is not avaiable on Windows")
	endif()
endif()
if(NOT "${ENABLE_CINTEGRATION}" STREQUAL "NO" AND HAVE_PYTHON3 AND NOT WIN32)
	set(HAVE_CUDF ON CACHE INTERNAL "C udfs extension is available")

	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	set(CAPI_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
							sqlmonet5_headers sql_headers sqlserver_headers sqlcommon_headers store_headers)
	if(WIN32)
		list(APPEND CAPI_LINK_LIBRARIES monetdb5 gdk)
	endif()

	# Generate cheader.text.h file
	execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/boot.py"
					"${CMAKE_CURRENT_SOURCE_DIR}/cheader.h" "${CMAKE_CURRENT_BINARY_DIR}/cheader.text.h"
					RESULT_VARIABLE CHEADER_CODE OUTPUT_QUIET)
	if(NOT CHEADER_CODE EQUAL 0)
		message(FATAL_ERROR "Could not generate cheader.text.h file")
	endif()

	add_library(capi MODULE capi.c capi.h cheader.h)
	target_link_libraries(capi PRIVATE ${CAPI_LINK_LIBRARIES})
	set_target_properties(capi PROPERTIES OUTPUT_NAME _capi)
	target_compile_definitions(capi PRIVATE LIBCUDF)

	install(TARGETS capi DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:capi> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES capi.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 50_capi.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
