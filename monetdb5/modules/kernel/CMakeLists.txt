#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_library(kernel_headers INTERFACE)
target_include_directories(kernel_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

add_library(kernel OBJECT
			aggr.c
			alarm.c
			algebra.c algebra.h
			bat5.c bat5.h
			batcolor.c batcolor.h
			batmmath.c batmmath.h
			batstr.c
			group.c group.h
			microbenchmark.c microbenchmark.h
			mmath.c mmath.h mmath_private.h
			status.c status.h)
target_compile_definitions(kernel PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)
target_link_libraries(kernel PUBLIC kernel_headers PRIVATE utils_headers stream_headers gdk_headers mal_headers
					  atoms_headers)
set_target_properties(kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:kernel> PARENT_SCOPE)

install(FILES
		bat5.mal algebra.mal status.mal
		mmath.mal alarm.mal batstr.mal
		batcolor.mal batmmath.mal
		group.mal aggr.mal
		microbenchmark.mal
		DESTINATION ${LIBDIR}/monetdb5)

if(HAVE_HGE)
	install(FILES 00_aggr_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
