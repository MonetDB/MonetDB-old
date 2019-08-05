#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(. ${CRYPTO_INCLUDE_DIR} ${PCRE_INCLUDE_DIR} ${LIBXML2_INCLUDE_DIR} ${ICONV_INCLUDE_DIR})

site_name(BUILD_HOST)
set(builtby "$ENV{USER}@${BUILD_HOST}")
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
set(compilercall "${CMAKE_C_COMPILER}${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
set(linkercall "${CMAKE_LINKER}${CMAKE_C_LINK_FLAGS}")
configure_file(monet_version.c.in ${CMAKE_CURRENT_BINARY_DIR}/monet_version.c @ONLY)

add_executable(mserver5 mserver5.c monet_version.h ${CMAKE_CURRENT_BINARY_DIR}/monet_version.c)
target_link_libraries(mserver5 PRIVATE utils_headers sqlmonet5_headers sql_headers sqlcommon_headers store_headers
					  sqlserver_headers monetdb5 gdk stream mapi moptions ${UUID_LIBRARIES} ${MATH_LIBRARIES}
					  ${SOCKET_LIBRARIES} ${THREAD_LIBRARIES} ${DL_LIBRARIES} ${CRYPTO_LIBRARIES} ${PCRE_LIBRARIES}
					  ${LIBXML2_LIBRARIES} ${PSAPI_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES}
					  ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${KVM_LIBRARIES})
install(TARGETS mserver5 DESTINATION ${BINDIR})

install(FILES tomographintro.docx tomographintro.pdf DESTINATION ${DATADIR}/doc/MonetDB-client-tools)
install(FILES monetdblogo.png DESTINATION ${DATADIR}/doc/MonetDB)

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:mserver5> DESTINATION ${BINDIR} OPTIONAL)
else()
	configure_file(mserver5.1.in ${CMAKE_CURRENT_BINARY_DIR}/mserver5.1 @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mserver5.1 DESTINATION ${MANDIR}/man1)
endif()

if(HAVE_TESTING)
	add_executable(shutdowntest shutdowntest.c)
	target_link_libraries(shutdowntest PRIVATE utils_headers sqlmonet5_headers sql_headers sqlserver_headers
						  sqlcommon_headers store_headers monetdb5 gdk stream mapi moptions ${UUID_LIBRARIES}
						  ${MATH_LIBRARIES} ${SOCKET_LIBRARIES} ${THREAD_LIBRARIES} ${DL_LIBRARIES} ${CRYPTO_LIBRARIES}
						  ${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES} ${PSAPI_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES}
						  ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES}
						  ${KVM_LIBRARIES})
	install(TARGETS shutdowntest DESTINATION ${BINDIR})
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:shutdowntest> DESTINATION ${BINDIR} OPTIONAL)
	endif()
endif()
