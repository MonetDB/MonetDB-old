#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../common/options ../../common/stream ../../common/utils ${OPENSSL_INCS})

add_library(mapi SHARED mapi.c)
add_dependencies(mapi moptions mcrypt stream)
target_link_libraries(mapi moptions mcrypt stream ${SOCKET_LIBRARIES} ${OPENSSL_LIBRARIES})
set_target_properties(mapi PROPERTIES VERSION ${MAPI_VERSION} SOVERSION ${MAPI_VERSION_MAJOR})
