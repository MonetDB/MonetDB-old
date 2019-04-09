#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(MONETDB5_PRIVATE_LINK_OBJECTS ${UUID_LIBRARIES} ${MATH_LIBRARIES} ${SOCKET_LIBRARIES} ${MALLOC_LIBRARIES}
								  ${PTHREAD_LIBRARIES} ${DL_LIBRARIES} ${CRYPTO_LIBRARIES} ${PCRE_LIBRARIES}
								  ${LIBXML2_LIBRARIES} gdk stream)
set(MONETDB5_PUBLIC_LINK_OBJECTS mal atoms kernel malmodules optimizer scheduler mutils mcrypt msabaoth)
if(HAVE_MAPI)
	list(APPEND MONETDB5_PRIVATE_LINK_OBJECTS mapi)
endif()

if(MSVC)
	add_library(monetdb5 STATIC monetdb.c)
	target_link_libraries(monetdb5 PRIVATE ${MONETDB5_PRIVATE_LINK_OBJECTS} PUBLIC ${MONETDB5_PUBLIC_LINK_OBJECTS})

	add_library(monetdb5shared SHARED monetdb.c)
	target_link_libraries(monetdb5shared PRIVATE ${MONETDB5_PRIVATE_LINK_OBJECTS} PUBLIC ${MONETDB5_PUBLIC_LINK_OBJECTS})
	set_target_properties(monetdb5shared PROPERTIES OUTPUT_NAME monetdb5)

	install(TARGETS monetdb5shared DESTINATION ${LIBDIR})
else()
	add_library(monetdb5 SHARED monetdb.c)
	target_link_libraries(monetdb5 PRIVATE ${MONETDB5_PRIVATE_LINK_OBJECTS} PUBLIC ${MONETDB5_PUBLIC_LINK_OBJECTS})
	set_target_properties(monetdb5 PROPERTIES VERSION ${MONETDB5_VERSION} SOVERSION ${MONETDB5_VERSION_MAJOR})
endif()

install(TARGETS monetdb5 DESTINATION ${LIBDIR})

if(NOT WIN32)
	if(LIBXML2_FOUND)
		set(PKG_LIBXML2 "libxml-2.0")
	endif()
	if(NOT CRYPTO_LIBRARIES STREQUAL "")
		set(PKG_OPENSSL "-l${CRYPTO_LIBRARIES}")
	endif()
	if(HAVE_LIBPCRE)
		set(PKG_PCRE "-l${PCRE_LIBRARIES}")
	endif()
	configure_file(monetdb5.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb5.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb5.pc DESTINATION ${PKGCONFIGDIR})
endif()
