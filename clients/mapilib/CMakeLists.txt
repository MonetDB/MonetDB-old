#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../common/stream ../../common/utils ${CRYPTO_INCLUDE_DIR})

set(MAPI_RC "")
if(WIN32)
	set(MAPI_RC mapi.rc)
endif()

add_library(mapi SHARED mapi.c ${MAPI_RC})
target_link_libraries(mapi PRIVATE ${SOCKET_LIBRARIES} ${CRYPTO_LIBRARIES} moptions mcrypt stream)
set_target_properties(mapi PROPERTIES VERSION ${MAPI_VERSION} SOVERSION ${MAPI_VERSION_MAJOR})
target_compile_definitions(mapi PRIVATE LIBMAPI)

install(TARGETS mapi DESTINATION ${LIBDIR})
install(FILES mapi.h DESTINATION ${INCLUDEDIR}/monetdb)

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:mapi> DESTINATION ${LIBDIR} OPTIONAL)
else()
	if(SOCKET_LIBRARIES)
		set(PKG_SOCKET_LIBS "-l${SOCKET_LIBRARIES}") #TODO fix this
	endif()
	configure_file(monetdb-mapi.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb-mapi.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb-mapi.pc DESTINATION ${PKGCONFIGDIR})
endif()
