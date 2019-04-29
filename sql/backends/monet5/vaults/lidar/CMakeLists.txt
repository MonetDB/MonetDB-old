#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

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
			set(HAVE_LIDAR ON CACHE INTERNAL "lidar extension is available")
		endif()
	elseif(${ENABLE_LIDAR} STREQUAL "YES")
		message(FATAL_ERROR "liblas library required for lidar module")
	endif()
endif()
if(HAVE_LIDAR)
	include_directories(.. ../.. ../../../../include ../../../../common ../../../../storage ../../../../server
						../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/kernel
						../../../../../monetdb5/mal ../../../../../monetdb5/modules/mal ../../../../../monetdb5/optimizer
						../../../../../monetdb5/scheduler ../../../../../common/stream ../../../../../common/utils
						../../../../../gdk ${LIBLAS_INCLUDE_DIR})

	set(LIDAR_LINK_LIBRARIES ${LIBLAS_LIBRARIES})
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
