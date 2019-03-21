#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../../include ../../../common ../../../storage ../../../server ../../../../monetdb5/modules/atoms
					../../../../monetdb5/modules/kernel ../../../../monetdb5/mal ../../../../monetdb5/modules/mal
					../../../../monetdb5/optimizer ../../../../monetdb5/scheduler ../../../../common/options
					../../../../common/stream ../../../../common/utils ../../../../gdk)


add_library(generator MODULE generator.c  generator.h)
target_link_libraries(generator PRIVATE monetdb5 gdk)
set_target_properties(generator PROPERTIES RUNTIME_OUTPUT_NAME _generator LIBRARY_OUTPUT_NAME _generator)
target_compile_definitions(generator PRIVATE LIBGENERATOR)

install(TARGETS generator DESTINATION ${LIBDIR}/monetdb5)
install(FILES generator.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 90_generator.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 90_generator.sql DESTINATION ${LIBDIR}/monetdb5/createdb)

if(HAVE_HGE)
	install(FILES generator_hge.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 90_generator_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 90_generator_hge.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
