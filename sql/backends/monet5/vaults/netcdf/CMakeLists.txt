#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(HAVE_NETCDF OFF CACHE INTERNAL "netcdf extension is available" FORCE) # Disable it and test it again
if(NOT ${ENABLE_NETCDF} STREQUAL "NO")
	include(FindNetCDF)

	if(NETCDF_FOUND)
		set(NETCDF_MINIMUM_VERSION "4.2")
		if(NETCDF_VERSION VERSION_LESS "${NETCDF_MINIMUM_VERSION}")
			if(${WITH_NETCDF} STREQUAL "YES")
				message(FATAL_ERROR "netcdf library found, but the version is too old: ${NETCDF_VERSION} < ${NETCDF_MINIMUM_VERSION}")
			else()
				message(STATUS "netcdf library found, but the version is too old: ${NETCDF_VERSION} < ${NETCDF_MINIMUM_VERSION}")
			endif()
		else()
			set(HAVE_NETCDF ON CACHE INTERNAL "netcdf extension is available" FORCE)
		endif()
	elseif(${ENABLE_NETCDF} STREQUAL "YES")
		message(FATAL_ERROR "netcdf library required for NetCDF support")
	endif()
endif()
if(HAVE_NETCDF)
	include_directories(${NETCDF_INCLUDE_DIR})

	set(NETCDF_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
							  sqlmonet5_headers sql_headers sqlserver_headers sqlcommon_headers store_headers
							  ${NETCDF_LIBRARIES})
	if(WIN32)
		list(APPEND NETCDF_LINK_LIBRARIES monetdb5 gdk)
	endif()

	# the netcdf library is also called netcdf, so cmake gets confused. Give it another name
	add_library(netcdff MODULE netcdf.c netcdf_vault.h)
	target_link_libraries(netcdff PRIVATE ${NETCDF_LINK_LIBRARIES})
	set_target_properties(netcdff PROPERTIES OUTPUT_NAME _netcdf)
	target_compile_definitions(netcdff PRIVATE LIBNCDF)

	install(TARGETS netcdff DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:netcdff> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES netcdf.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 74_netcdf.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 74_netcdf.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
