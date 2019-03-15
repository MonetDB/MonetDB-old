#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../mapilib ../../common/options ../../common/stream ../../common/utils ${READLINE_INCS})

add_library(mcutil STATIC dump.c dotmonetdb.c dotmonetdb.h eventparser.c eventparser.h)

add_executable(mclient mclient.c ReadlineTools.c ReadlineTools.h mhelp.c mhelp.h)
add_dependencies(mclient mcutil mutils mcrypt mapi stream)
target_link_libraries(mclient mcutil mutils mcrypt mapi stream ${READLINE_LIBS} ${READLINE_LIBRARIES}
					  ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY} ${OPENSSL_LIBRARIES})

add_executable(msqldump msqldump.c msqldump.h)
add_dependencies(msqldump mcutil mutils mapi stream)
target_link_libraries(msqldump mcutil mutils mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
					  ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY}
					  ${OPENSSL_LIBRARIES})

add_executable(stethoscope stethoscope.c)
add_dependencies(stethoscope mcutil mutils mapi stream)
target_link_libraries(stethoscope mcutil mutils mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
					  ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY}
					  ${OPENSSL_LIBRARIES})

add_executable(tachograph tachograph.c)
add_dependencies(tachograph mcutil mutils mapi stream)
target_link_libraries(tachograph mcutil mutils mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
					  ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY}
					  ${OPENSSL_LIBRARIES})

add_executable(tomograph tomograph.c)
add_dependencies(tomograph mcutil mutils mapi stream)
target_link_libraries(tomograph mcutil mutils mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
					  ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARY} ${CURL_LIBRARY} ${Iconv_LIBRARY}
					  ${OPENSSL_LIBRARIES})

install(TARGETS mclient msqldump stethoscope tachograph tomograph DESTINATION ${BINDIR})
