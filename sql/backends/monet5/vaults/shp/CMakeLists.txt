#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(NOT ${ENABLE_SHP} STREQUAL "NO")
	if(NOT HAVE_GEOM AND ${ENABLE_SHP} STREQUAL "YES")
		message(FATAL_ERROR "geom module required for ESRI Shapefile vault")
	else()
		include(FindGDAL)

		if(GDAL_FOUND)
			set(HAVE_SHP ON CACHE INTERNAL "shp extension is available")
		elseif(${ENABLE_SHP} STREQUAL "YES")
			message(FATAL_ERROR "gdal library required for ESRI Shapefile vault")
		endif()
	endif()
endif()
if(HAVE_SHP)
	include_directories(.. ../.. ../../../../include ../../../../common ../../../../storage ../../../../server
						../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/kernel
						../../../../../monetdb5/mal ../../../../../monetdb5/modules/mal ../../../../../monetdb5/optimizer
						../../../../../monetdb5/scheduler ../../../../../common/stream ../../../../../common/utils
						../../../../../gdk ../../../../../geom/lib ../../../../../geom/monetdb5 ${GDAL_INCLUDE_DIR})

	set(SHP_LINK_LIBRARIES ${GDAL_LIBRARIES})
	if(WIN32)
		list(APPEND SHP_LINK_LIBRARIES monetdb5 gdk)
	endif()

	add_library(shp MODULE shp.c shp.h)
	target_link_libraries(shp PRIVATE ${SHP_LINK_LIBRARIES})
	set_target_properties(shp PROPERTIES OUTPUT_NAME _shp)
	target_compile_definitions(shp PRIVATE LIBGTIFF)

	install(TARGETS shp DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:shp> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES shp.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 75_shp.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 75_shp.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
