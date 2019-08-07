#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(HAVE_LIDAR OFF CACHE INTERNAL "lidar extension is available" FORCE) # Disable it and test it again
if(NOT ${ENABLE_LIDAR} STREQUAL "NO")
	include(FindLiblas)

	if(LIBLAS_FOUND)
		set(LIBLAS_VERSION_REQUIRED_MIN "1.8.0")
		if(LIBLAS_VERSION VERSION_LESS "${LIBLAS_MINIMUM_VERSION}")
			if(${WITH_LIBLAS} STREQUAL "YES")
				message(FATAL_ERROR "liblas library found, but the version is too old: ${LIBLAS_VERSION} < ${LIBLAS_MINIMUM_VERSION}")
			else()
				message(STATUS "liblas library found, but the version is too old: ${LIBLAS_VERSION} < ${LIBLAS_MINIMUM_VERSION}")
			endif()
		else()
			set(HAVE_LIDAR ON CACHE INTERNAL "lidar extension is available" FORCE)
		endif()
	elseif(${ENABLE_LIDAR} STREQUAL "YES")
		message(FATAL_ERROR "liblas library required for lidar module")
	endif()
endif()
if(HAVE_LIDAR)
	include_directories(${LIBLAS_INCLUDE_DIR})

	set(LIDAR_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
							 sqlmonet5_headers sql_headers sqlserver_headers sqlcommon_headers store_headers
							 ${LIBLAS_LIBRARIES})
	if(WIN32)
		list(APPEND LIDAR_LINK_LIBRARIES monetdb5 gdk)
	endif()

	add_library(lidar MODULE lidar.c lidar.h)
	target_link_libraries(lidar PRIVATE ${LIDAR_LINK_LIBRARIES})
	set_target_properties(lidar PROPERTIES OUTPUT_NAME _lidar)
	target_compile_definitions(lidar PRIVATE LIBLIDAR)

	install(TARGETS lidar DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:lidar> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES lidar.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 75_lidar.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 75_lidar.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
