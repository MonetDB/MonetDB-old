#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_subdirectory(config)

include_directories(.. ../../../common/stream ../../../common/utils ../../../common/options ../../../clients/mapilib
					../../../gdk ../../../monetdb5/mal)

add_executable(monetdbd merovingian.c merovingian.h argvcmds.c argvcmds.h client.c client.h connections.c connections.h
			   controlrunner.c controlrunner.h discoveryrunner.c discoveryrunner.h forkmserver.c forkmserver.h
			   handlers.c handlers.h multiplex-funnel.c multiplex-funnel.h proxy.c proxy.h)
target_link_libraries(monetdbd PRIVATE meroutil mcrypt msabaoth mutils stream mapi monetdb5 gdk ${UUID_LIBRARIES}
					  ${CRYPTO_LIBRARIES} ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES}
					  ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${MATH_LIBRARIES} ${MALLOC_LIBRARIES}
					  ${PTHREAD_LIBRARIES} ${DL_LIBRARIES} ${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES})
install(TARGETS monetdbd DESTINATION ${BINDIR})

configure_file(monetdbd.1.in ${CMAKE_CURRENT_BINARY_DIR}/monetdbd.1 @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdbd.1 DESTINATION ${MANDIR}/man1)
