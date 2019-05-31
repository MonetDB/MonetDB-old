#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

# This file holds macros for compilation objects common to MonetDB and MonetDBLite

# Compile sqlparser with Bison only once for both MonetDB and MonetDBLite
# Warning! So far the HAVE_EMBEDDED macro is not used in sql_parser.y
macro(COMPILE_SQLPARSER)
	bison_target(sqlparser sql/server/sql_parser.y ${CMAKE_BINARY_DIR}/sql_parser.tab.c
				COMPILE_FLAGS "-d -p sql -r all" DEFINES_FILE ${CMAKE_BINARY_DIR}/sql_parser.tab.h)
	add_library(bison_obj OBJECT ${BISON_sqlparser_OUTPUTS})
	set_target_properties(bison_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
	target_include_directories(bison_obj PRIVATE common/utils common/stream gdk monetdb5/mal sql/common sql/include
							   sql/server sql/storage sql/storage/bat)
	if(NOT MSVC)
		cmake_push_check_state()
		set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS};-Wno-unreachable-code")
		check_c_source_compiles("int main(int argc,char** argv){(void)argc;(void)argv;return 0;}" COMPILER_Wnounreachablecode) # Warning don't add '-' or '/' to the output variable!
		cmake_pop_check_state()
		if(COMPILER_Wnounreachablecode)
			target_compile_options(bison_obj PRIVATE -Wno-unreachable-code) # use this flag only to compile the bison output
		endif()
	endif()
endmacro()

# Create C array with MAL scripts content as well the module names bundled
macro(BUILD_EMBEDDED_MAL_SCRIPTS BUNDLE_NAME VARIABLE_NAME SCRIPTS_LIST)
	execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/buildtools/scripts/mal2h.py"
			"${CMAKE_CURRENT_BINARY_DIR}/${BUNDLE_NAME}.h" ${SCRIPTS_LIST}
			RESULT_VARIABLE PY_SCRIPT_RC OUTPUT_QUIET)
	if(NOT PY_SCRIPT_RC EQUAL 0)
		message(FATAL_ERROR "Could not generate sql_mal_inline.h file")
	endif()
	# Generate file with mal module names
	set(BUNDLED_MAL_NAMES_LIST "str ${VARIABLE_NAME}[MAXMODULES] = {")
	set(MYLIST ${SCRIPTS_LIST}) # We need this to work on cmake
	foreach(script IN LISTS MYLIST)
		get_filename_component(SCRIPT_BASENAME "${script}" NAME)
		string(REGEX REPLACE "\\.[^.]*" "" SCRIPT_BASENAME ${SCRIPT_BASENAME})
		string(APPEND BUNDLED_MAL_NAMES_LIST "\"${SCRIPT_BASENAME}\",")
	endforeach()
	string(APPEND BUNDLED_MAL_NAMES_LIST "0};\n")
	file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${BUNDLE_NAME}_names.h" "${BUNDLED_MAL_NAMES_LIST}")
endmacro()

# Create C array with SQL scripts content
macro(BUILD_EMBEDDED_SQL_SCRIPTS BUNDLE_NAME SCRIPTS_LIST)
	execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/buildtools/scripts/sql2h.py"
			"${CMAKE_CURRENT_BINARY_DIR}/${BUNDLE_NAME}.h" ${SCRIPTS_LIST}
			RESULT_VARIABLE PY_SCRIPT_RC OUTPUT_QUIET)
	if(NOT PY_SCRIPT_RC EQUAL 0)
		message(FATAL_ERROR "Could not generate ${BUNDLE_NAME}.h file")
	endif()
endmacro()
