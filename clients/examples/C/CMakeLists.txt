#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../mapilib)

if(HAVE_TESTING)
	set(COMMON_LIBRARIES mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES}
						 ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})
	add_executable(sample0 sample0.c)
	target_link_libraries(sample0 PRIVATE ${COMMON_LIBRARIES})
	add_executable(sample1 sample1.c)
	target_link_libraries(sample1 PRIVATE ${COMMON_LIBRARIES})
	add_executable(sample4 sample4.c)
	target_link_libraries(sample4 PRIVATE ${COMMON_LIBRARIES})
	add_executable(smack00 smack00.c)
	target_link_libraries(smack00 PRIVATE ${COMMON_LIBRARIES})
	add_executable(smack01 smack01.c)
	target_link_libraries(smack01 PRIVATE ${COMMON_LIBRARIES})

	install(TARGETS sample0 sample1 sample4 smack00 smack01 DESTINATION ${BINDIR})
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:sample0> $<TARGET_PDB_FILE:sample1> $<TARGET_PDB_FILE:sample4>
				$<TARGET_PDB_FILE:smack00> $<TARGET_PDB_FILE:smack01> DESTINATION ${BINDIR} OPTIONAL)
	endif()
endif()
