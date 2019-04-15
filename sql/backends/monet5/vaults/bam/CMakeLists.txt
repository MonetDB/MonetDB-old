#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(.. ../.. ../../../../include ../../../../common ../../../../storage ../../../../server
					../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/kernel
					../../../../../monetdb5/mal ../../../../../monetdb5/modules/mal ../../../../../monetdb5/optimizer
					../../../../../common/stream ../../../../../common/utils ../../../../../gdk
					../../../../../tools/merovingian ../../../../../tools/merovingian/daemon ${SAMTOOLS_INCLUDE_DIR})

# the bam library is also called bam, so cmake gets confused. Give another name
add_library(bamm MODULE
			bam_loader.c bam_loader.h
			bam_wrapper.c bam_wrapper.h
			bam_db_interface.c bam_db_interface.h
			bam_globals.c bam_globals.h
			bam_lib.c bam_lib.h
			bam_export.c bam_export.h
			mykstring.h)
target_link_libraries(bamm PRIVATE monetdb5 gdk ${SAMTOOLS_LIBRARIES})
set_target_properties(bamm PROPERTIES OUTPUT_NAME _bam)
target_compile_definitions(bamm PRIVATE LIBBAM)

install(TARGETS bamm DESTINATION ${LIBDIR}/monetdb5)
install(FILES bam.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 85_bam.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 85_bam.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
