#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../common/options ../common/stream ../common/utils ${VALGRIND_INCLUDE_DIR})

set(GDK_LINK_LIBRARIES moptions mutils)
if(WIN32)
	list(APPEND GDK_LINK_LIBRARIES stream ${MATH_LIBRARIES} ${SOCKET_LIBRARIES} ${MALLOC_LIBRARIES} ${PTHREAD_LIBRARIES}
		 ${DL_LIBRARIES} ${PSAPI_LIBRARIES})
endif()

add_library(gdk SHARED
			gdk_select.c
			gdk_calc.c gdk_calc.h gdk_calc_compare.h gdk_calc_private.h
			gdk_ssort.c gdk_ssort_impl.h
			gdk_aggr.c
			gdk.h gdk_cand.h gdk_batop.c
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
			gdk_analytic_func.c gdk_analytic.h $<$<PLATFORM_ID:Windows>:libbat.rc>)
target_link_libraries(gdk PRIVATE ${GDK_LINK_LIBRARIES})
set_target_properties(gdk PROPERTIES VERSION ${GDK_VERSION} SOVERSION ${GDK_VERSION_MAJOR} OUTPUT_NAME bat)
target_compile_definitions(gdk PRIVATE LIBGDK)

install(TARGETS gdk DESTINATION ${LIBDIR})
install(FILES
		gdk.h
		gdk_analytic.h
		gdk_atoms.h
		gdk_bbp.h
		gdk_calc.h
		gdk_cand.h
		gdk_delta.h
		gdk_posix.h
		gdk_hash.h
		gdk_system.h
		gdk_utils.h
		DESTINATION ${INCLUDEDIR}/monetdb)

if(WIN32)
	install(FILES $<TARGET_PDB_FILE:gdk> DESTINATION ${LIBDIR} OPTIONAL)
else()
	if(NOT MATH_LIBRARIES STREQUAL "")
		set(PKG_MATH_LIBRARIES "-l${MATH_LIBRARIES}")
	endif()
	if(NOT MALLOC_LIBRARIES STREQUAL "")
		set(PKG_MALLOC_LIBRARIES "-l${MALLOC_LIBRARIES}")
	endif()
	if(NOT PTHREAD_LIBRARIES STREQUAL "")
		set(PKG_PTHREAD_LIBRARIES "-${PTHREAD_LIBRARIES}")
	endif()
	if(NOT DL_LIBRARIES STREQUAL "")
		set(PKG_DL_LIBS "-l${DL_LIBRARIES}")
	endif()
	configure_file(monetdb-gdk.pc.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb-gdk.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb-gdk.pc DESTINATION ${PKGCONFIGDIR})
endif()
