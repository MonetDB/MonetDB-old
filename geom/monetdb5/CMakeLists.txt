#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../lib ../../gdk ../../common/stream ../../common/options ../../monetdb5/mal ${GEOS_INCS}
					${PROJ_INCLUDE_DIR})

add_library(geom MODULE geom.h geom.c geomBulk.c geom_upgrade.c)
target_link_libraries(geom PRIVATE geomobj gdk stream monetdb5 ${GEOS_LIBS} ${PROJ_LIBRARIES})
set_target_properties(geom PROPERTIES RUNTIME_OUTPUT_NAME _geom LIBRARY_OUTPUT_NAME _geom)
target_compile_definitions(geom PRIVATE LIBGEOM)

install(TARGETS geom DESTINATION ${LIBDIR}/monetdb5)
install(FILES geom.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 30_geom.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
