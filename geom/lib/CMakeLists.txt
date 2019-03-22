#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../common/options ../../common/stream ../../gdk ${GEOS_INCS} ${PROJ_INCLUDE_DIR})

add_library(geomobj STATIC libgeom.c libgeom.h)
set_target_properties(geomobj PROPERTIES POSITION_INDEPENDENT_CODE ON)
