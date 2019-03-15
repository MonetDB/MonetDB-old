#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../monetdb5/mal ../../gdk ../../common/stream ../../common/utils ../../common/options
					../../monetdb5/modules/atoms ../../monetdb5/modules/mal ../../sql/backends/monet5 ../../sql/common
					../../sql/include ../../sql/server ../../sql/storage ${OPENSSL_INCS} ${PCRE_INCLUDE_DIR}
					${LIBXML2_INCLUDE_DIRS})

configure_file(monet_version.c.in monet_version.c @ONLY)

add_executable(mserver5 mserver5.c monet_version.h monet_version.c)
add_dependencies(mserver5 monetdb5 mal bat stream mapi)
target_link_libraries(mserver5 monetdb5 mal bat stream mapi ${UUID_LIBRARIES} ${MATH_LIBS} ${SOCKET_LIBRARIES}
					  ${MALLOC_LIBS} ${PTHREAD_LIBRARIES} ${PTHREAD_LIBRARIES} ${CMAKE_DL_LIBS} ${OPENSSL_LIBRARIES}
					  ${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES} ${PSAPILIB_LIBS} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
					  ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY})
install(TARGETS mserver5 DESTINATION ${BINDIR})

if(HAVE_TESTING)
	add_executable(shutdowntest shutdowntest.c)
	add_dependencies(shutdowntest monetdb5 mal bat stream mapi)
	target_link_libraries(shutdowntest monetdb5 mal bat stream mapi ${UUID_LIBRARIES} ${MATH_LIBS} ${SOCKET_LIBRARIES}
			${MALLOC_LIBS} ${PTHREAD_LIBRARIES} ${PTHREAD_LIBRARIES} ${CMAKE_DL_LIBS} ${OPENSSL_LIBRARIES}
			${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES} ${PSAPILIB_LIBS} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
			${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY})
	install(TARGETS mserver5 DESTINATION ${BINDIR})
endif()
