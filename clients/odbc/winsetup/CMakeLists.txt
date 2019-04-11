#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../driver ${ODBC_INCLUDE_DIR})

add_library(MonetODBCs SHARED setup.c setup.rc setup.syms resource.h)
target_link_libraries(MonetODBCs PRIVATE ${ODBCINST_LIBRARIES})
install(TARGETS MonetODBCs DESTINATION ${LIBDIR})

add_executable(odbcinstall install.c)
target_link_libraries(odbcinstall PRIVATE version shlwapi ${ODBC_LIBRARIES} ${ODBCINST_LIBRARIES})
install(TARGETS odbcinstall DESTINATION ${BINDIR})
