#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${ODBC_INCLUDE_DIR})

add_library(MonetODBCs MODULE drvcfg.c drvcfg.h)
target_link_libraries(MonetODBCs PRIVATE utils_headers)
target_compile_definitions(MonetODBCs PRIVATE LIBMONETODBCS)
install(TARGETS MonetODBCs DESTINATION ${LIBDIR})
