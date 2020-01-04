#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CRYPTO_INCLUDE_DIR})

set(MAPI_LINK_LIBRARIES mcrypt mutils)
if(WIN32)
	list(APPEND MAPI_LINK_LIBRARIES stream ${SOCKET_LIBRARIES} ${CRYPTO_LIBRARIES})
endif()

add_library(mapi_headers INTERFACE)
target_include_directories(mapi_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

add_library(mapi SHARED mapi.c mapi.h mapi.rc)
target_include_directories(mapi INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)
target_link_libraries(mapi PUBLIC mapi_headers PRIVATE stream_headers ${MAPI_LINK_LIBRARIES})
set_target_properties(mapi PROPERTIES VERSION ${MAPI_VERSION} SOVERSION ${MAPI_VERSION_MAJOR})
target_compile_definitions(mapi PRIVATE LIBMAPI)

install(TARGETS mapi DESTINATION ${LIBDIR})
install(FILES mapi.h DESTINATION ${INCLUDEDIR}/monetdb)

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:mapi> DESTINATION ${LIBDIR} OPTIONAL)
else()
	configure_file(monetdb-mapi.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb-mapi.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb-mapi.pc DESTINATION ${PKGCONFIGDIR})
endif()
