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
					../../../../../monetdb5/scheduler ../../../../../common/stream ../../../../../common/utils
					../../../../../gdk ${CFITSIO_INCLUDE_DIR})

add_library(fits MODULE fits.c fits.h)
target_link_libraries(fits PRIVATE $<$<PLATFORM_ID:Windows>:monetdb5> $<$<PLATFORM_ID:Windows>:gdk>
					  ${CFITSIO_LIBRARIES})
set_target_properties(fits PROPERTIES OUTPUT_NAME _fits)
target_compile_definitions(fits PRIVATE LIBFITS)

install(TARGETS fits DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:fits> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES fits.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 72_fits.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 72_fits.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
