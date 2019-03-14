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
target_link_libraries(stream ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY})
target_include_directories(stream PRIVATE ${Iconv_INCLUDE_DIR})
