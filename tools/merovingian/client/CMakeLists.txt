#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../../common/stream ../../../common/utils ../utils ${ICONV_INCLUDE_DIR})

add_executable(monetdb monetdb.c)
target_link_libraries(monetdb PRIVATE meroutil mcrypt msabaoth mutils stream ${UUID_LIBRARIES} ${CRYPTO_LIBRARIES}
					  ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES}
					  ${CURL_LIBRARIES} ${ICONV_LIBRARIES})

install(TARGETS monetdb DESTINATION ${BINDIR})
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:monetdb> DESTINATION ${BINDIR} OPTIONAL)
endif()
install(FILES monetdb.1 DESTINATION ${MANDIR}/man1)
