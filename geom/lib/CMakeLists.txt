#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../common/options ../../common/stream ../../gdk ${GEOS_INCLUDE_DIR} ${PROJ_INCLUDE_DIR})

set(GEOM_OBJECTS libgeom.c libgeom.h)

if(WIN32)
	add_library(geom SHARED ${GEOM_OBJECTS})
	target_compile_definitions(geom PRIVATE LIBGEOM)
	install(TARGETS geom DESTINATION ${LIBDIR})
else()
	add_library(geom STATIC ${GEOM_OBJECTS})
	set_target_properties(geom PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(geom PRIVATE gdk ${GEOS_LIBRARIES} ${PROJ_LIBRARIES})
