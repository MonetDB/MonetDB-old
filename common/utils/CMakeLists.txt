#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${UUID_INCLUDE_DIR} ${CRYPTO_INCLUDE_DIR})

# FindRevision script must run every time to detect the current revision
add_custom_target(mrevision COMMAND ${CMAKE_COMMAND} ARGS -P ${CMAKE_CURRENT_SOURCE_DIR}/FindRevision.cmake
				  ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/monetdb_hgversion.h)

set(MUTILS_OBJECTS mutils.h mutils.c revision.c)
set(MCRYPT_OBJECTS mcrypt.h mcrypt.c)
set(MSABAOTH_OBJECTS muuid.h muuid.c msabaoth.h msabaoth.c)

if(WIN32)
	add_library(mutils SHARED ${MUTILS_OBJECTS})
	target_compile_definitions(mutils PRIVATE LIBMUTILS LIBGDK LIBMEROUTIL)
	install(TARGETS mutils DESTINATION ${LIBDIR})
	install(FILES $<TARGET_PDB_FILE:mutils> DESTINATION ${LIBDIR} OPTIONAL)

	add_library(mcrypt SHARED ${MCRYPT_OBJECTS})
	target_compile_definitions(mcrypt PRIVATE LIBMAPI LIBMCRYPT)
	target_link_libraries(mcrypt PRIVATE ${CRYPTO_LIBRARIES})
	install(TARGETS mcrypt DESTINATION ${LIBDIR})
	install(FILES $<TARGET_PDB_FILE:mcrypt> DESTINATION ${LIBDIR} OPTIONAL)

	add_library(msabaoth SHARED ${MSABAOTH_OBJECTS})
	target_compile_definitions(msabaoth PRIVATE LIBMSABAOTH LIBMUUID LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER
							   LIBMONETDB5)
	target_link_libraries(msabaoth PRIVATE ${CRYPTO_LIBRARIES} mutils)
	install(TARGETS msabaoth DESTINATION ${LIBDIR})
	install(FILES $<TARGET_PDB_FILE:msabaoth> DESTINATION ${LIBDIR} OPTIONAL)
else()
	add_library(mutils STATIC ${MUTILS_OBJECTS})
	set_target_properties(mutils PROPERTIES POSITION_INDEPENDENT_CODE ON)

	add_library(mcrypt STATIC ${MCRYPT_OBJECTS})
	set_target_properties(mcrypt PROPERTIES POSITION_INDEPENDENT_CODE ON)

	add_library(msabaoth STATIC ${MSABAOTH_OBJECTS})
	set_target_properties(msabaoth PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_dependencies(mutils mrevision)
install(FILES matomic.h DESTINATION ${INCLUDEDIR}/monetdb)
