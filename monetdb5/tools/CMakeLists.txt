#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(MONETDB5_DEPENDENCIES atoms kernel malmodules optimizer scheduler msabaoth msabaoth bat stream)
set(MONETDB5_LINK_LIBRARIES atoms kernel malmodules optimizer scheduler msabaoth msabaoth bat stream ${UUID_LIBRARIES}
	${MATH_LIBS} ${SOCKET_LIBRARIES} ${MALLOC_LIBS} ${PTHREAD_LIBRARIES} ${CMAKE_DL_LIBS} ${OPENSSL_LIBRARIES}
	${PCRE_LIBRARIES} ${LIBXML2_LIBRARIES})
if(HAVE_MAPI)
	list(APPEND MONETDB5_DEPENDENCIES mapi)
	list(APPEND MONETDB5_LINK_LIBRARIES mapi)
endif()
if(WIN32)
	list(APPEND MONETDB5_DEPENDENCIES strptime)
	list(APPEND MONETDB5_LINK_LIBRARIES strptime)
endif()

add_library(monetdb5 SHARED monetdb.c)
add_dependencies(monetdb5 ${MONETDB5_DEPENDENCIES})
target_link_libraries(monetdb5 ${MONETDB5_LINK_LIBRARIES})
set_target_properties(monetdb5 PROPERTIES VERSION ${MONETDB5_VERSION} SOVERSION ${MONETDB5_VERSION_MAJOR})
install(TARGETS monetdb5 DESTINATION ${LIBDIR})

if(NOT WIN32)
	if(MATH_LIBS)
		set(PKG_LIBXML2 "libxml-2.0")
	endif()
	if(MALLOC_LIBS)
		set(PKG_OPENSSL "-l${OPENSSL_LIBRARIES}")
	endif()
	if(CMAKE_DL_LIBS)
		set(PKG_PCRE "-l${PCRE_LIBRARIES}")
	endif()
	configure_file(monetdb5.pc.in monetdb5.pc @ONLY)
	install(FILES monetdb5.pc DESTINATION ${PKGCONFIGDIR})
endif()
