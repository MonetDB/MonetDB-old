#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../../common/stream ../../../common/utils ../utils)

add_executable(monetdb monetdb.c)
target_link_libraries(monetdb PRIVATE meroutil mcrypt msabaoth mutils stream ${UUID_LIBRARIES} ${OPENSSL_LIBRARIES}
					  ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY})
install(TARGETS monetdb DESTINATION ${BINDIR})
install(FILES monetdb.1 DESTINATION ${DATADIR}/man/man1)
