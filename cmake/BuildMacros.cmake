#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

# This file holds macros for compilation objects common to MonetDB and MonetDBLite

# Create C array with MAL scripts content as well the module names bundled
macro(BUILD_EMBEDDED_MAL_SCRIPTS BUNDLE_NAME VARIABLE_NAME REMOVE_COMMENTS SCRIPTS_LIST)
	execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/buildtools/scripts/mal2h.py"
					"${REMOVE_COMMENTS}" "${CMAKE_CURRENT_BINARY_DIR}/${BUNDLE_NAME}.h" ${SCRIPTS_LIST}
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
