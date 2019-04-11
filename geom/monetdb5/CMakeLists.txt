#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../lib ../../gdk ../../common/stream ../../common/options ../../monetdb5/mal ${GEOS_INCLUDE_DIR}
					${PROJ_INCLUDE_DIR})

add_library(geom_module MODULE geom.h geom.c geomBulk.c geom_upgrade.c)
target_link_libraries(geom_module PRIVATE geom gdk stream monetdb5 ${GEOS_LIBRARIES} ${PROJ_LIBRARIES})
set_target_properties(geom_module PROPERTIES OUTPUT_NAME geom)

install(TARGETS geom_module DESTINATION ${LIBDIR}/monetdb5)
install(FILES geom.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 30_geom.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
