#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(.. ../.. ../../../../include ../../../../common ../../../../storage ../../../../server
					../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/kernel
					../../../../../monetdb5/mal ../../../../../monetdb5/modules/mal ../../../../../monetdb5/optimizer
					../../../../../monetdb5/scheduler ../../../../../common/stream ../../../../../common/utils
					../../../../../gdk ../../../../../geom/lib ../../../../../geom/monetdb5 ${GDAL_INCLUDE_DIR})

add_library(shp MODULE shp.c shp.h)
target_link_libraries(shp PRIVATE $<$<PLATFORM_ID:Windows>:monetdb5> $<$<PLATFORM_ID:Windows>:gdk> ${GDAL_LIBRARIES})
set_target_properties(shp PROPERTIES OUTPUT_NAME _shp)
target_compile_definitions(shp PRIVATE LIBGTIFF)

install(TARGETS shp DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:shp> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES shp.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 75_shp.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 75_shp.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
