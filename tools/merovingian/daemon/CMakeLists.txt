#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_subdirectory(config)

include_directories(.. ../../../common/options ../../../common/stream ../../../common/utils ../../../clients/mapilib
					../../../gdk ../../../monetdb5/mal ${ICONV_INCLUDE_DIR})

add_executable(monetdbd merovingian.c merovingian.h argvcmds.c argvcmds.h client.c client.h connections.c connections.h
			   controlrunner.c controlrunner.h discoveryrunner.c discoveryrunner.h forkmserver.c forkmserver.h
			   handlers.c handlers.h multiplex-funnel.c multiplex-funnel.h proxy.c proxy.h)
target_link_libraries(monetdbd PRIVATE meroutil moptions stream mapi monetdb5 gdk ${UUID_LIBRARIES}
					  ${CRYPTO_LIBRARIES} ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES}
					  ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${MATH_LIBRARIES}
					  ${THREAD_LIBRARIES} ${DL_LIBRARIES} ${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES} ${KVM_LIBRARIES})

install(TARGETS monetdbd DESTINATION ${BINDIR})
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:monetdbd> DESTINATION ${BINDIR} OPTIONAL)
endif()

configure_file(monetdbd.1.in ${CMAKE_CURRENT_BINARY_DIR}/monetdbd.1 @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdbd.1 DESTINATION ${MANDIR}/man1)
