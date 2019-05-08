#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(NOT ${ENABLE_RINTEGRATION} STREQUAL "NO")
	message(STATUS "Checking for R interpreter instalation")
	if(WIN32)
		find_path(R_INCLUDE R.h)
		find_library(R_LIBRARY R)
		if(R_INCLUDE AND R_LIBRARY)
			set(R_INCLUDE_DIR "${R_INCLUDE}")
			set(R_LIBRARIES "${R_LIBRARY}")
		elseif(${ENABLE_RINTEGRATION} STREQUAL "YES")
			message(FATAL_ERROR "R include files and libraries are required for R integration support")
		else()
			message(STATUS "R include files and libraries are required for R integration support")
		endif()
	else()
		exec_program("R" ARGS "RHOME" OUTPUT_VARIABLE R_HOME_OUTPUT RETURN_VALUE R_HOME_RC)
		exec_program("R" ARGS "CMD config --cppflags" OUTPUT_VARIABLE R_CPP_OUTPUT RETURN_VALUE R_CPP_RC)
		exec_program("R" ARGS "CMD config --ldflags" OUTPUT_VARIABLE R_LD_OUTPUT RETURN_VALUE R_LD_RC)
		if(R_HOME_RC EQUAL 0 AND R_CPP_RC EQUAL 0 AND R_LD_RC EQUAL 0)
			string(REPLACE "-I" "" R_INCLUDE_DIR ${R_CPP_OUTPUT}) # cmake will get confused
			string(REPLACE "-fopenmp" "" R_LIBRARIES ${R_LD_OUTPUT}) # We don't want to link to OpenMP
			string(REPLACE "-fopenmp=libgomp" "" R_LIBRARIES ${R_LIBRARIES})

			message(STATUS "R executable, library and include files found")
		elseif(${ENABLE_RINTEGRATION} STREQUAL "YES")
			message(FATAL_ERROR "R executable required for R integration support")
		else()
			message(STATUS "R executable not found")
		endif()
	endif()
	if(R_INCLUDE_DIR AND R_LIBRARIES)
		set(HAVE_LIBR ON CACHE INTERNAL "rapi extension is available")
	endif()
endif()
set(RHOME "${R_HOME_OUTPUT}" CACHE INTERNAL "R home directory") # No R_HOME on Windows

if(HAVE_LIBR)
	include_directories(../../../common/stream ../../../common/utils ../../../gdk ../../mal ../../modules/kernel
						../../modules/atoms ../../modules/mal ../../../sql/common ../../../sql/include
						../../../sql/server ../../../sql/storage ../../../sql/backends/monet5 ${R_INCLUDE_DIR})

	set(RAPI_LINK_LIBRARIES ${R_LIBRARIES})
	if(WIN32)
		list(APPEND RAPI_LINK_LIBRARIES monetdb5 gdk)
	endif()

	add_library(rapi MODULE rapi.c rapi.h converters.c.h)
	target_link_libraries(rapi PRIVATE ${RAPI_LINK_LIBRARIES})
	set_target_properties(rapi PROPERTIES OUTPUT_NAME _rapi)
	target_compile_definitions(rapi PRIVATE LIBRAPI)
	cmake_push_check_state()
	set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS};-Wno-redundant-decls")
	check_c_source_compiles("int main(int argc,char** argv){(void)argc;(void)argv;return 0;}" COMPILER_Wnoredundantdecls) # Warning don't add '-' or '/' to the output variable!
	cmake_pop_check_state()
	if(COMPILER_Wnoredundantdecls)
		target_compile_options(rapi PRIVATE -Wno-redundant-decls)
	endif()

	install(TARGETS rapi DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:rapi> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES rapi.mal rapi.R DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 50_rapi.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
