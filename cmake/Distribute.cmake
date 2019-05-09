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
# DEB;RPM;productbuild;FREEBSD;WIX; -> for these we will continue with the previous
set(CPACK_PACKAGE_VENDOR "MonetDBSolutions")
set(CPACK_PACKAGE_CHECKSUM "SHA512")
set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/buildtools/conf/monetdb.ico")
set(CPACK_PACKAGE_FILE_NAME "MonetDB-${MONETDB_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/license.txt")

# Source build configurations
# Using regular expressions for ignored files, later we could move to a generated variable
set(CPACK_SOURCE_IGNORE_FILES "/bootstrap" "/buildtools/autogen/" "/clients/odbc/doc" "/debian/" "/de-bootstrap"
	"/libversions" "/MacOSX/" "/rpm.mk.in" "/tools/monetdbbincopy" "/testing/quicktest" "/testing/cmptests.py"
	"/vertoo.config" "/vertoo.data" "/\\\\.idea/" "/\\\\.git/" "/\\\\.hg/" "ChangeLog.*" "CMakeFiles*" "\\\\.hg.*" "#"
	"~" "\\\\.ac$" "\\\\.ag$" "\\\\.lst$" "\\\\.mal\\\\.sh$")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "MonetDB-${MONETDB_VERSION}")
include(CPack)

# RPM configurations
#set(CPACK_RPM_USER_BINARY_SPECFILE "${CMAKE_SOURCE_DIR}/MonetDB.spec") ...
