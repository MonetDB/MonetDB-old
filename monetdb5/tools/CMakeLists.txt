#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(MONETDB5_PRIVATE_LINK_LIBRARIES )
if(WIN32)
	list(APPEND MONETDB5_PRIVATE_LINK_LIBRARIES ${UUID_LIBRARIES} ${MATH_LIBRARIES} ${SOCKET_LIBRARIES}
		 ${THREAD_LIBRARIES} ${DL_LIBRARIES} ${CRYPTO_LIBRARIES} ${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES} gdk stream)
	if(HAVE_MAPI)
		list(APPEND MONETDB5_PRIVATE_LINK_LIBRARIES mapi)
	endif()
endif()

add_library(monetdb5 SHARED ${MONETDB5_OBJECTS} $<$<PLATFORM_ID:Windows>:libmonetdb5.rc>)
target_link_libraries(monetdb5 PUBLIC mutils mcrypt msabaoth PRIVATE ${MONETDB5_PRIVATE_LINK_LIBRARIES})
set_target_properties(monetdb5 PROPERTIES VERSION ${MONETDB5_VERSION} SOVERSION ${MONETDB5_VERSION_MAJOR})

install(TARGETS monetdb5 DESTINATION ${LIBDIR})

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:monetdb5> DESTINATION ${LIBDIR} OPTIONAL)
else()
	if(LIBXML2_FOUND)
		set(PKG_LIBXML2 "libxml-2.0")
	endif()
	if(COMMONCRYPTO_FOUND OR OPENSSL_FOUND)
		set(PKG_CRYPTO "-lcrypto") # In OpenSSL case, we link only to the crypto library, so it will be always one library to link to
	endif()
	if(HAVE_LIBPCRE)
		set(PKG_PCRE "-lpcre")
	endif()
	configure_file(monetdb5.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb5.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb5.pc DESTINATION ${PKGCONFIGDIR})
endif()
