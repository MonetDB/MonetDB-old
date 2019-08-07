#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(HAVE_FITS OFF CACHE INTERNAL "fits extension is available" FORCE) # Disable it and test it again
if(NOT ${ENABLE_FITS} STREQUAL "NO")
	include(FindCfitsio)

	if(CFITSIO_FOUND)
		set(HAVE_FITS ON CACHE INTERNAL "fits extension is available" FORCE)
	elseif(${ENABLE_FITS} STREQUAL "YES")
		message(FATAL_ERROR "cfitsio library required for FITS support")
	endif()
endif()
if(HAVE_FITS)
	include_directories(${CFITSIO_INCLUDE_DIR})

	set(FITS_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
							sqlmonet5_headers sql_headers sqlserver_headers sqlcommon_headers store_headers
							${CFITSIO_LIBRARIES})
	if(WIN32)
		list(APPEND FITS_LINK_LIBRARIES monetdb5 gdk)
	endif()

	add_library(fits MODULE fits.c fits.h)
	target_link_libraries(fits PRIVATE ${FITS_LINK_LIBRARIES})
	set_target_properties(fits PROPERTIES OUTPUT_NAME _fits)
	target_compile_definitions(fits PRIVATE LIBFITS)

	install(TARGETS fits DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:fits> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES fits.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 72_fits.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 72_fits.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
