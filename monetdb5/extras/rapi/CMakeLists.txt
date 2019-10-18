#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(HAVE_LIBR OFF CACHE INTERNAL "rapi extension is available" FORCE)
if(NOT ${ENABLE_RINTEGRATION} STREQUAL "NO")
	if(WIN32)
		find_path(R_INCLUDE_DIR R.h)
		find_library(R_LIBRARIES R)
		if(NOT R_INCLUDE OR NOT R_LIBRARY)
			if(${ENABLE_RINTEGRATION} STREQUAL "YES")
				message(FATAL_ERROR "R include files and libraries are required for R integration support")
			else()
				message(STATUS "R include files and libraries are required for R integration support")
			endif()
		endif()
	else() # UNIX
		execute_process(COMMAND "R" "RHOME" RESULT_VARIABLE R_HOME_RC OUTPUT_VARIABLE R_HOME_OUTPUT
						OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND "R" "CMD" "config" "--cppflags" RESULT_VARIABLE R_CPP_RC OUTPUT_VARIABLE R_CPP_OUTPUT
						OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND "R" "CMD" "config" "--ldflags" RESULT_VARIABLE R_LD_RC OUTPUT_VARIABLE R_LD_OUTPUT
						OUTPUT_STRIP_TRAILING_WHITESPACE)
		if(R_HOME_RC EQUAL 0 AND R_CPP_RC EQUAL 0 AND R_LD_RC EQUAL 0)
			string(REPLACE "-I" "" R_INCLUDE_DIR ${R_CPP_OUTPUT}) # cmake will get confused
			string(REPLACE "-fopenmp" "" R_LIBRARIES ${R_LD_OUTPUT}) # We don't want to link to OpenMP
			string(REPLACE "-fopenmp=libgomp" "" R_LIBRARIES ${R_LIBRARIES})
			string(REGEX REPLACE "-specs=[^\ ]*" "" R_LIBRARIES ${R_LIBRARIES}) # -specs argument is not used and both GCC and Clang complain about it

			set(R_INCLUDE_DIR "${R_INCLUDE_DIR}" CACHE INTERNAL "R include directory path")
			set(R_LIBRARIES "${R_LIBRARIES}" CACHE INTERNAL "R libraries path")
			set(RHOME "${R_HOME_OUTPUT}" CACHE INTERNAL "R home directory") # No R_HOME on Windows
			message(STATUS "R executable, library and include files found")
		elseif(${ENABLE_RINTEGRATION} STREQUAL "YES")
			message(FATAL_ERROR "R executable required for R integration support")
		else()
			message(STATUS "R executable not found")
		endif()
	endif()
	find_path(R_EMBEDDED_INCLUDE_DIR Rembedded.h PATH_SUFFIXES R)
	if(NOT R_EMBEDDED_INCLUDE_DIR)
		if(${ENABLE_RINTEGRATION} STREQUAL "YES")
			message(FATAL_ERROR "Rembedded.h include file is required for R integration support")
		else()
			message(STATUS "Rembedded.h include file is required for R integration support")
		endif()
	endif()
	if(R_INCLUDE_DIR AND R_LIBRARIES AND R_EMBEDDED_INCLUDE_DIR)
		set(HAVE_LIBR ON CACHE INTERNAL "rapi extension is available" FORCE)
	endif()
endif()

if(HAVE_LIBR)
	include_directories(${R_INCLUDE_DIR} ${R_EMBEDDED_INCLUDE_DIR})

	set(RAPI_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
		sqlmonet5_headers sql_headers sqlserver_headers sqlcommon_headers store_headers ${R_LIBRARIES})
	if(WIN32)
		list(APPEND RAPI_LINK_LIBRARIES monetdb5 gdk)
	endif()

	add_library(rapi MODULE rapi.c rapi.h converters.c.h)
	target_link_libraries(rapi PRIVATE ${RAPI_LINK_LIBRARIES})
	set_target_properties(rapi PROPERTIES OUTPUT_NAME _rapi)
	target_compile_definitions(rapi PRIVATE LIBRAPI)
	if(${CMAKE_C_COMPILER_ID} MATCHES "^GNU|Clang|AppleClang$")
		cmake_push_check_state()
		set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS};-Wno-redundant-decls")
		check_c_source_compiles("int main(int argc,char** argv){(void)argc;(void)argv;return 0;}" COMPILER_Wnoredundantdecls) # Warning don't add '-' or '/' to the output variable!
		cmake_pop_check_state()
		if(COMPILER_Wnoredundantdecls)
			target_compile_options(rapi PRIVATE -Wno-redundant-decls)
		endif()
	endif()

	install(TARGETS rapi DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:rapi> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES rapi.mal rapi.R DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 50_rapi.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
