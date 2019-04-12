#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../mapilib ../../common/options ../../common/stream ../../common/utils ${READLINE_INCLUDE_DIR}
					${ICONV_INCLUDE_DIR})

add_library(mcutil STATIC dump.c dotmonetdb.c dotmonetdb.h eventparser.c eventparser.h)

add_executable(mclient mclient.c ReadlineTools.c ReadlineTools.h mhelp.c mhelp.h)
target_link_libraries(mclient PRIVATE mcutil mutils mcrypt moptions mapi stream ${READLINE_LIBS} ${READLINE_LIBRARIES}
					  ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(msqldump msqldump.c msqldump.h)
target_link_libraries(msqldump PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(stethoscope stethoscope.c)
target_link_libraries(stethoscope PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(tachograph tachograph.c)
target_link_libraries(tachograph PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(tomograph tomograph.c)
target_link_libraries(tomograph PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

install(TARGETS mclient msqldump stethoscope tachograph tomograph DESTINATION ${BINDIR})
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:mclient> $<TARGET_PDB_FILE:msqldump> $<TARGET_PDB_FILE:stethoscope>
			$<TARGET_PDB_FILE:tachograph> $<TARGET_PDB_FILE:tomograph> DESTINATION ${BINDIR} OPTIONAL)
else()
	install(FILES mclient.1 msqldump.1 DESTINATION ${MANDIR}/man1)
endif()
