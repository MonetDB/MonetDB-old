#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

# In this file, we set cpack configurations for building tarballs and eventually binary releases

# Cpack general configurations
set(CPACK_SOURCE_GENERATOR "7Z;TBZ2;TGZ;TXZ;ZIP")
set(CPACK_GENERATOR "${CPACK_SOURCE_GENERATOR}") # All what we need I think
# DEB;RPM;productbuild;FREEBSD;WIX; -> for these we will continue with the previous method
set(CPACK_PACKAGE_VENDOR "MonetDBSolutions")
set(CPACK_PACKAGE_CHECKSUM "SHA512")
set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/buildtools/conf/monetdb.ico")
set(CPACK_PACKAGE_FILE_NAME "MonetDB-${MONETDB_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/license.txt")

# Source build configurations
# Using regular expressions for ignored files, later we could move to a generated variable
set(CPACK_SOURCE_IGNORE_FILES "/bootstrap" "/buildtools/autogen/" "/clients/odbc/doc" "/debian/" "/de-bootstrap"
	"/libversions" "/MacOSX/" "/tools/monetdbbincopy" "/testing/quicktest" "/testing/cmptests.py" "/vertoo.config"
	"/vertoo.data" "/\\\\.idea/" "/\\\\.git/" "/\\\\.hg/" "ChangeLog.*" "CMakeFiles*" "\\\\.hg.*" "#" "~" "\\\\.ac$"
	"\\\\.ag$" "\\\\.lst$" "\\\\.mal\\\\.sh$")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "MonetDB-${MONETDB_VERSION}")
include(CPack)

# Build rpms only on Linux, because that's what the find_linux_distro script is for
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	if(HAVE_LIDAR) # needed for rpm.mk.in
		set(LIBLAS_RPM "with")
	else()
		set(LIBLAS_RPM "without")
	endif()

	find_program(ECHO NAMES echo DOC "echo program fullpath")
	if(NOT ECHO OR NOT BASH)
		message(FATAL_ERROR "echo and bash programs are required to build rpms")
	endif()

	# Create a temporary file in CMakeFiles and copy it to the final location while setting proper permissions
	# We need these two steps because I couldn't find a better way to do it with cmake...
	configure_file(${CMAKE_SOURCE_DIR}/find_linux_distro.sh.in ${CMAKE_BINARY_DIR}/CMakeFiles/find_linux_distro.sh @ONLY)
	file(COPY ${CMAKE_BINARY_DIR}/CMakeFiles/find_linux_distro.sh DESTINATION ${CMAKE_BINARY_DIR}
		 FILE_PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT})
	file(REMOVE ${CMAKE_BINARY_DIR}/CMakeFiles/find_linux_distro.sh)

	execute_process(COMMAND "${CMAKE_BINARY_DIR}/find_linux_distro.sh" RESULT_VARIABLE LINUX_DIST_RC
					OUTPUT_VARIABLE LINUX_DIST OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(LINUX_DIST AND LINUX_DIST_RC EQUAL 0)
		configure_file(${CMAKE_SOURCE_DIR}/rpm.mk.in ${CMAKE_BINARY_DIR}/rpm.mk @ONLY)
		install(FILES ${CMAKE_BINARY_DIR}/rpm.mk DESTINATION ${INCLUDEDIR}/monetdb)
	else()
		message(WARNING "Could not determine Linux distribution, thus rpm.mk could not be generated")
	endif()
endif()
