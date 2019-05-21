#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(NOT ${CMAKE_ARGC} EQUAL 6)
	message(FATAL_ERROR "Six arguments must be given to this script (called from common/utils/CMakeLists.txt)")
endif()

# Get the current version control revision
if(EXISTS "${CMAKE_ARGV4}/.hg")
	exec_program("hg" "${CMAKE_ARGV4}" ARGS "id -i" OUTPUT_VARIABLE HG_OUPUT_RES RETURN_VALUE HG_RETURN_CODE)
	if(HG_RETURN_CODE EQUAL 0 AND HG_OUPUT_RES)
		set(MERCURIAL_ID "${HG_OUPUT_RES}")
	else()
		message(FATAL_ERROR "Failed to find mercurial ID")
	endif()
elseif(EXISTS "${CMAKE_ARGV4}/.git")
	exec_program("git" "${CMAKE_ARGV4}" ARGS "rev-parse --short HEAD" OUTPUT_VARIABLE GIT_OUPUT_RES RETURN_VALUE GIT_RETURN_CODE)
	if(GIT_RETURN_CODE EQUAL 0 AND GIT_OUPUT_RES)
		set(MERCURIAL_ID "${GIT_OUPUT_RES}")
	else()
		message(FATAL_ERROR "Failed to find git ID")
	endif()
else()
	set(MERCURIAL_ID "Unknown")
endif()

# Write it to monetdb_hgversion.h file
file(WRITE "${CMAKE_ARGV5}" "#define MERCURIAL_ID \"${MERCURIAL_ID}\"")
