#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(GENERATOR_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers kernel_headers atoms_headers
							 optimizer_headers)
if(WIN32)
	list(APPEND GENERATOR_LINK_LIBRARIES monetdb5 gdk)
endif()

add_library(generator MODULE generator.c generator.h)
target_link_libraries(generator PRIVATE ${GENERATOR_LINK_LIBRARIES})
set_target_properties(generator PROPERTIES OUTPUT_NAME _generator)
target_compile_definitions(generator PRIVATE LIBGENERATOR)

install(TARGETS generator DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:generator> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES generator.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 90_generator.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 90_generator.sql DESTINATION ${LIBDIR}/monetdb5/createdb)

if(HAVE_HGE)
	install(FILES generator_hge.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 90_generator_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 90_generator_hge.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
