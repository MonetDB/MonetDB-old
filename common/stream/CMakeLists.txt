#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../utils ${ZLIB_INCLUDE_DIR} ${BZIP2_INCLUDE_DIR} ${SNAPPY_INCLUDE_DIR} ${LZ4_INCLUDE_DIR}
					${LIBLZMA_INCLUDE_DIR} ${CURL_INCLUDE_DIR} ${ICONV_INCLUDE_DIR})

set(STREAM_LINK_LIBRARIES )
if(WIN32)
	list(APPEND STREAM_LINK_LIBRARIES ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
		 ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${SOCKET_LIBRARIES})
endif()

add_library(stream SHARED stream.c stream.h stream_socket.h)
target_link_libraries(stream PRIVATE ${STREAM_LINK_LIBRARIES})
set_target_properties(stream PROPERTIES VERSION ${STREAM_VERSION} SOVERSION ${STREAM_VERSION_MAJOR})
target_compile_definitions(stream PRIVATE LIBSTREAM)

install(TARGETS stream DESTINATION ${LIBDIR})
install(FILES stream_socket.h stream.h DESTINATION ${INCLUDEDIR}/monetdb)

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:stream> DESTINATION ${LIBDIR} OPTIONAL)
else()
	if(CURL_FOUND)
		set(PKG_CURL "${CMAKE_SHARED_LIBRARY_PREFIX}curl")
	endif()
	if(ZLIB_FOUND)
		set(PKG_ZLIB "zlib")
	endif()
	if(LIBLZMA_FOUND)
		set(PKG_LIBLZMA "${CMAKE_SHARED_LIBRARY_PREFIX}lzma")
	endif()
	if(LZ4_FOUND)
		set(PKG_LZ4 "${CMAKE_SHARED_LIBRARY_PREFIX}lz4")
	endif()
	if(BZIP2_FOUND)
		set(PKG_BZIP2 "bzip2")
	endif()
	if(SNAPPY_FOUND)
		set(PKG_SNAPPY "snappy")
	endif()
	if(ICONV_FOUND AND NOT ICONV_IS_BUILT_IN)
		get_filename_component(ICONV_LIBRARIES_PATH "${ICONV_LIBRARIES}" DIRECTORY)
		set(PKG_LIBICONV "-L${ICONV_LIBRARIES_PATH} -l${ICONV_LIBRARIES}")
	endif()
	configure_file(monetdb-stream.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb-stream.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb-stream.pc DESTINATION ${PKGCONFIGDIR})
endif()
