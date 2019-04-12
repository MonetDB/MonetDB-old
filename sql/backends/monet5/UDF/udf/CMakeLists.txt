#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../.. ../../../../include ../../../../common ../../../../storage ../../../../server
					../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/kernel
					../../../../../monetdb5/mal ../../../../../monetdb5/modules/mal ../../../../../monetdb5/optimizer
					../../../../../common/utils ../../../../../common/stream ../../../../../gdk)

add_library(udf MODULE udf.c udf.h udf_impl.h)
target_link_libraries(udf PRIVATE monetdb5 gdk)
set_target_properties(udf PROPERTIES OUTPUT_NAME _udf)
target_compile_definitions(udf PRIVATE LIBUDF)

install(TARGETS udf DESTINATION ${LIBDIR}/monetdb5)
install(FILES udf.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 80_udf.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 80_udf.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:udf> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()

if(HAVE_HGE)
	install(FILES udf_hge.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 80_udf_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
	install(FILES 80_udf_hge.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
endif()
