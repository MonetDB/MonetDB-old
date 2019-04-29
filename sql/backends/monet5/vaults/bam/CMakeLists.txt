#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(NOT ${ENABLE_SAMTOOLS} STREQUAL "NO")
	cmake_push_check_state()
	set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${MATH_LIBRARIES};${THREAD_LIBRARIES};${ZLIB_LIBRARIES};bam")
	check_include_file("samtools/bam.h" HAVE_SAMTOOLS_BAM_H)
	check_include_file("bam/bam.h" HAVE_BAM_BAM_H)
	check_include_file("samtools/sam.h" HAVE_SAMTOOLS_SAM_H)
	check_include_file("bam/sam.h" HAVE_BAM_SAM_H)
	check_include_file("samtools/bgzf.h" HAVE_SAMTOOLS_BGZF_H)
	check_include_file("bam/bgzf.h" HAVE_BAM_BGZF_H)
	check_include_file("samtools/kstring.h" HAVE_SAMTOOLS_KSTRING_H)
	check_include_file("bam/kstring.h" HAVE_BAM_KSTRING_H)
	check_symbol_exists("bam_header_read" "samtools/bam.h" HAVE_BAM_HEADER_READ)
	if(HAVE_BAM_HEADER_READ)
		set(BAM_LINK_LIBRARIES "bam")
		set(HAVE_SAMTOOLS ON CACHE INTERNAL "bam extension is available")
	elseif(${ENABLE_SAMTOOLS} STREQUAL "YES")
		message(FATAL_ERROR "bam_header_read symbol not found")
	else()
		message(STATUS "bam_header_read symbol not found")
	endif()
	cmake_pop_check_state()
endif()
if(HAVE_SAMTOOLS)
	include_directories(.. ../.. ../../../../include ../../../../common ../../../../storage ../../../../server
						../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/kernel
						../../../../../monetdb5/mal ../../../../../monetdb5/modules/mal ../../../../../monetdb5/optimizer
						../../../../../common/stream ../../../../../common/utils ../../../../../gdk
						../../../../../tools/merovingian ../../../../../tools/merovingian/daemon)

	if(WIN32)
		list(APPEND BAM_LINK_LIBRARIES monetdb5 gdk)
	endif()

	# the bam library is also called bam, so cmake gets confused. Give it another name
	add_library(bamm MODULE bam_loader.c bam_loader.h bam_wrapper.c bam_wrapper.h bam_db_interface.c bam_db_interface.h
				bam_globals.c bam_globals.h bam_lib.c bam_lib.h bam_export.c bam_export.h mykstring.h)
	target_link_libraries(bamm PRIVATE ${BAM_LINK_LIBRARIES})
	set_target_properties(bamm PROPERTIES OUTPUT_NAME _bam)
	target_compile_definitions(bamm PRIVATE LIBBAM)

	install(TARGETS bamm DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:bamm> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	endif()
	install(FILES bam.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 85_bam.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 85_bam.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
