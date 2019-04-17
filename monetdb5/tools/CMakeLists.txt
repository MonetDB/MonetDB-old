#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../mal ../../common/stream ../../common/utils ../../clients/mapilib ../../gdk ../optimizer
					../modules/atoms ../modules/mal ../modules/kernel ${CRYPTO_INCLUDE_DIR} ${PCRE_INCLUDE_DIR}
					${UUID_INCLUDE_DIR} ${LIBXML2_INCLUDE_DIR})

file(GLOB MAL_SRC "../mal/*.h" "../mal/*.c")
file(GLOB ATOMS_SRC "../modules/atoms/*.h" "../modules/atoms/*.c")
file(GLOB KERNEL_SRC  "../modules/kernel/*.h" "../modules/kernel/*.c")
file(GLOB MODULES_SRC "../modules/mal/*.h" "../modules/mal/*.c")
file(GLOB OPTIMIZER_SRC "../optimizer/*.h" "../optimizer/*.c")
file(GLOB SCHEDULER_SRC "../scheduler/*.h" "../scheduler/*.c")

add_library(monetdb5 SHARED ${MAL_SRC} ${ATOMS_SRC} ${KERNEL_SRC} ${MODULES_SRC} ${OPTIMIZER_SRC} ${SCHEDULER_SRC}
			$<$<PLATFORM_ID:Windows>:libmonetdb5.rc>)
target_link_libraries(monetdb5 PRIVATE mutils mcrypt msabaoth $<$<PLATFORM_ID:Windows>:${UUID_LIBRARIES}>
					  $<$<PLATFORM_ID:Windows>:${MATH_LIBRARIES}> $<$<PLATFORM_ID:Windows>:${SOCKET_LIBRARIES}>
					  $<$<PLATFORM_ID:Windows>:${MALLOC_LIBRARIES}> $<$<PLATFORM_ID:Windows>:${PTHREAD_LIBRARIES}>
					  $<$<PLATFORM_ID:Windows>:${DL_LIBRARIES}> $<$<PLATFORM_ID:Windows>:${CRYPTO_LIBRARIES}>
					  $<$<PLATFORM_ID:Windows>:${PCRE_LIBRARIES}> $<$<PLATFORM_ID:Windows>:${LIBXML2_LIBRARIES}>
					  $<$<PLATFORM_ID:Windows>:gdk> $<$<PLATFORM_ID:Windows>:stream>
					  $<$<AND:$<PLATFORM_ID:Windows>,$<BOOL:HAVE_MAPI>>:mapi>)
set_target_properties(monetdb5 PROPERTIES VERSION ${MONETDB5_VERSION} SOVERSION ${MONETDB5_VERSION_MAJOR})
target_compile_definitions(monetdb5 PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)

install(TARGETS monetdb5 DESTINATION ${LIBDIR})

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:monetdb5> DESTINATION ${LIBDIR} OPTIONAL)
else()
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
