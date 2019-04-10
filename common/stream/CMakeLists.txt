#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${ZLIB_INCLUDE_DIR} ${BZIP2_INCLUDE_DIR} ${SNAPPY_INCLUDE_DIR} ${LZ4_INCLUDE_DIR}
					${LIBLZMA_INCLUDE_DIR} ${CURL_INCLUDE_DIR} ${ICONV_INCLUDE_DIR})

add_library(stream SHARED stream.c stream.h stream_socket.h)
target_link_libraries(stream PRIVATE ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${SOCKET_LIBRARIES})
set_target_properties(stream PROPERTIES VERSION ${STREAM_VERSION} SOVERSION ${STREAM_VERSION_MAJOR})
target_compile_definitions(stream PRIVATE LIBSTREAM)

install(TARGETS stream DESTINATION ${LIBDIR})
install(FILES stream_socket.h stream.h DESTINATION ${INCLUDEDIR}/monetdb)

if(NOT WIN32)
	if(CURL_FOUND)
		set(PKG_CURL "-lcurl")
	endif()
	if(ZLIB_FOUND)
		set(PKG_ZLIB "-lz")
	endif()
	if(LIBLZMA_FOUND)
		set(PKG_LIBLZMA "-llzma")
	endif()
	if(LZ4_FOUND)
		set(PKG_LZ4 "-llz4")
	endif()
	if(BZIP2_FOUND)
		set(PKG_BZIP2 "-lbz2")
	endif()
	if(SNAPPY_FOUND)
		set(PKG_SNAPPY "-lsnappy")
	endif()
	if(SOCKET_LIBRARIES)
		set(PKG_SOCKET_LIBS "-l${SOCKET_LIBRARIES}") #TODO fix this
	endif()
	if(Iconv_FOUND AND NOT Iconv_IS_BUILT_IN)
		set(PKG_LIBICONV "-l${ICONV_LIBRARIES}")
	endif()
	configure_file(monetdb-stream.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb-stream.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb-stream.pc DESTINATION ${PKGCONFIGDIR})
endif()
