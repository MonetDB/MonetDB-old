#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../common/options ../common/stream ../common/utils ${VALGRIND_INCLUDE_DIR})

add_library(gdk SHARED
			gdk_select.c
			gdk_calc.c gdk_calc.h gdk_calc_compare.h gdk_calc_private.h
			gdk_ssort.c gdk_ssort_impl.h
			gdk_aggr.c
			gdk.h gdk_cand.h gdk_atomic.h gdk_batop.c
			gdk_search.c gdk_hash.c gdk_hash.h gdk_tm.c
			gdk_orderidx.c
			gdk_align.c gdk_bbp.c gdk_bbp.h
			gdk_heap.c gdk_utils.c gdk_utils.h
			gdk_atoms.c gdk_atoms.h gdk_string.c
			gdk_qsort.c gdk_qsort_impl.h
			gdk_storage.c gdk_bat.c
			gdk_delta.c gdk_cross.c gdk_system.c gdk_value.c
			gdk_posix.c gdk_logger.c gdk_sample.c xoshiro256starstar.h
			gdk_private.h gdk_delta.h gdk_logger.h gdk_posix.h
			gdk_system.h gdk_system_private.h gdk_tm.h gdk_storage.h
			gdk_group.c
			gdk_imprints.c gdk_imprints.h
			gdk_join.c gdk_project.c
			gdk_unique.c
			gdk_interprocess.c gdk_interprocess.h
			gdk_firstn.c
			gdk_analytic_bounds.c
			gdk_analytic_func.c gdk_analytic.h)
add_dependencies(gdk moptions mutils stream)
target_link_libraries(gdk moptions mutils stream ${MATH_LIBS} ${SOCKET_LIBRARIES} ${MALLOC_LIBS} ${PTHREAD_LIBRARIES}
					  ${DL_LIBS})
set_target_properties(gdk PROPERTIES COMPILE_DEFINITIONS "LIBGDK")
set_target_properties(gdk PROPERTIES VERSION ${GDK_VERSION} SOVERSION ${GDK_VERSION_MAJOR})
