#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../mal ../../common/options ../../common/stream ../../common/utils ../../clients/mapilib ../../gdk
					../optimizer ../modules/atoms ../modules/mal ../modules/kernel ${CRYPTO_INCLUDE_DIR}
					${PCRE_INCLUDE_DIR} ${LIBXML2_INCLUDE_DIR})

set(MONETDB5_PRIVATE_LINK_LIBRARIES ${UUID_LIBRARIES} ${MATH_LIBRARIES} ${SOCKET_LIBRARIES} ${MALLOC_LIBRARIES}
									${PTHREAD_LIBRARIES} ${DL_LIBRARIES} ${CRYPTO_LIBRARIES} ${PCRE_LIBRARIES}
									${LIBXML2_LIBRARIES} gdk stream)
if(HAVE_MAPI)
	list(APPEND MONETDB5_PRIVATE_LINK_LIBRARIES mapi)
endif()

file(GLOB MAL_SRC "../mal/*.h" "../mal/*.c")
file(GLOB ATOMS_SRC "../modules/atoms/*.h" "../modules/atoms/*.c")
file(GLOB KERNEL_SRC  "../modules/kernel/*.h" "../modules/kernel/*.c")
file(GLOB MODULES_SRC "../modules/mal/*.h" "../modules/mal/*.c")
file(GLOB OPTIMIZER_SRC "../optimizer/*.h" "../optimizer/*.c")
file(GLOB SCHEDULER_SRC "../scheduler/*.h" "../scheduler/*.c")

add_library(monetdb5 SHARED ${MAL_SRC} ${ATOMS_SRC} ${KERNEL_SRC} ${MODULES_SRC} ${OPTIMIZER_SRC} ${SCHEDULER_SRC})
target_link_libraries(monetdb5 PRIVATE ${MONETDB5_PRIVATE_LINK_LIBRARIES} PUBLIC mutils mcrypt msabaoth)
set_target_properties(monetdb5 PROPERTIES VERSION ${MONETDB5_VERSION} SOVERSION ${MONETDB5_VERSION_MAJOR})
target_compile_definitions(monetdb5 PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)

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
