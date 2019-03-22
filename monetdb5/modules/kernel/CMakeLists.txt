#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../mal ../atoms ../../../common/options ../../../common/stream ../../../gdk)

add_library(kernel ${LIBMONETDB5_LINKING}
			aggr.c
			alarm.c
			algebra.c algebra.h
			bat5.c bat5.h
			batcolor.c batcolor.h
			batmmath.c batmmath.h
			batstr.c
			group.c group.h
			logger.c
			microbenchmark.c microbenchmark.h
			mmath.c mmath.h mmath_private.h
			status.c status.h)
set_target_properties(kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(kernel PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)

install(FILES
		bat5.mal algebra.mal status.mal
		mmath.mal alarm.mal batstr.mal
		batcolor.mal batmmath.mal
		group.mal aggr.mal
		logger.mal microbenchmark.mal
		DESTINATION ${LIBDIR}/monetdb5)

if(HAVE_HGE)
	install(FILES 00_aggr_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
