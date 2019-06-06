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

# This macros sets the required system libraries besides the C standard library. It should be used by the MonetDBLite
# programing language bindings
macro(SET_SYSTEM_LIBRARIES)
	if(NOT WIN32)
		set(THREADS_PREFER_PTHREAD_FLAG ON) # We do prefer pthreads on UNIX platforms
	endif()
	find_package(Threads)
	set(THREAD_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")

	set(DL_LIBRARIES "")
	set(KVM_LIBRARIES "")
	set(MATH_LIBRARIES "")
	set(PSAPI_LIBRARIES "")
	set(SOCKET_LIBRARIES "")
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
		set(DL_LIBRARIES "${CMAKE_DL_LIBS}")
	endif()
	if(${CMAKE_SYSTEM_NAME} MATCHES "^FreeBSD|DragonFly|NetBSD$") # Warning - I checked the man pages and only tested on FreeBSD yet
		set(KVM_LIBRARIES "kvm")
	endif()
	if(${CMAKE_SYSTEM_NAME} MATCHES "^Linux|FreeBSD|NetBSD$")
		set(MATH_LIBRARIES "m")
	endif()
	if(WIN32) # Both these libraries and respective include files (psapi.h and winsock2.h) come with the Windows SDK <version>, which should be installed with Visual Studio and set on the path by MSVC
		set(PSAPI_LIBRARIES "psapi") # We need the psapi library for GetProcessMemoryInfo function, which is no longer required from Windows 7 and Windows Server 2008 R2 up (the latter is suported until January 2020)
		set(SOCKET_LIBRARIES "ws2_32")
	endif()
endmacro()
