#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(UDF_LINK_LIBRARIES utils_headers stream_headers gdk_headers mal_headers atoms_headers malmodules_headers
					   sqlmonet5_headers sql_headers sqlserver_headers sqlcommon_headers store_headers)
if(WIN32)
	list(APPEND UDF_LINK_LIBRARIES monetdb5 gdk)
endif()

add_library(udf MODULE udf.c udf.h udf_impl.h)
target_link_libraries(udf PRIVATE ${UDF_LINK_LIBRARIES})
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
