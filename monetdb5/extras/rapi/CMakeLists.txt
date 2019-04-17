#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../../common/stream ../../../common/utils ../../../gdk ../../mal ../../modules/kernel
					../../modules/atoms ../../modules/mal ../../../sql/common ../../../sql/include
					../../../sql/server ../../../sql/storage ../../../sql/backends/monet5 ${R_INCLUDE_DIR})

add_library(rapi MODULE rapi.c rapi.h converters.c.h)
target_link_libraries(rapi PRIVATE $<$<PLATFORM_ID:Windows>:monetdb5> $<$<PLATFORM_ID:Windows>:gdk> ${R_LIBRARIES})
set_target_properties(rapi PROPERTIES OUTPUT_NAME _rapi)
target_compile_definitions(rapi PRIVATE LIBRAPI)
if(NOT MSVC)
	target_compile_options(rapi PRIVATE -Wno-redundant-decls)
endif()

install(TARGETS rapi DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:rapi> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES rapi.mal rapi.R DESTINATION ${LIBDIR}/monetdb5)
install(FILES 50_rapi.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
