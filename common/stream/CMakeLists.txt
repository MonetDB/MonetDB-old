#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${ZLIB_INCLUDE_DIR} ${BZIP2_INCLUDE_DIR} ${SNAPPY_INCLUDEDIR} ${LZ4_INCLUDEDIR}
				    ${LIBLZMA_INCLUDE_DIR} ${CURL_INCLUDE_DIR})

add_library(stream SHARED stream.c stream.h stream_socket.h)
target_link_libraries(stream PRIVATE ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY})
target_include_directories(stream PRIVATE ${Iconv_INCLUDE_DIR})
set_target_properties(stream PROPERTIES VERSION ${STREAM_VERSION} SOVERSION ${STREAM_VERSION_MAJOR})

install(TARGETS stream LIBRARY DESTINATION ${LIBDIR})
install(FILES stream_socket.h stream.h DESTINATION ${INCLUDEDIR})

if(NOT WIN32)
	if(CURL_LIBRARY)
		set(PKG_CURL "-lcurl")
	endif()
	if(ZLIB_LIBRARIES)
		set(PKG_ZLIB "-lz")
	endif()
	if(LIBLZMA_LIBRARY)
		set(PKG_LIBLZMA "-llzma")
	endif()
	if(LZ4_LIBRARIES)
		set(PKG_LZ4 "-llz4")
	endif()
	if(BZIP2_LIBRARIES)
		set(PKG_BZIP2 "-lbz2")
	endif()
	if(SNAPPY_LIBRARIES)
		set(PKG_SNAPPY "-lsnappy")
	endif()
	if(SOCKET_LIBRARIES)
		set(PKG_SOCKET_LIBS "-l${SOCKET_LIBRARIES}") #TODO fix this
	endif()
	if(Iconv_LIBRARY AND NOT Iconv_IS_BUILT_IN)
		set(PKG_LIBICONV "-l${Iconv_LIBRARY}")
	endif()
	configure_file(monetdb-stream.pc.in monetdb-stream.pc @ONLY)
	install(FILES monetdb-stream.pc DESTINATION ${PKGCONFIGDIR})
endif()
