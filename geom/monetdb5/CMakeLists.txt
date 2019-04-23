#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../lib ../../gdk ../../common/stream ../../common/utils ../../monetdb5/mal ${GEOS_INCLUDE_DIR}
					${PROJ_INCLUDE_DIR})

set(GEOM_LINK_LIBRARIES ${GEOS_LIBRARIES} ${PROJ_LIBRARIES})
if(WIN32)
	list(APPEND GEOM_LINK_LIBRARIES monetdb5 gdk stream)
endif()

add_library(geom MODULE geom.h geom.c geomBulk.c geom_upgrade.c libgeom.c libgeom.h)
target_link_libraries(geom PRIVATE ${GEOM_LINK_LIBRARIES})
set_target_properties(geom PROPERTIES OUTPUT_NAME _geom)
target_compile_definitions(geom PRIVATE LIBGEOM)

install(TARGETS geom DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:geom> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES geom.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 30_geom.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
