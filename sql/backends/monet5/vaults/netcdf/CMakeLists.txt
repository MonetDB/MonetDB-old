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
					../../../../../gdk ${NETCDF_INCLUDE_DIR})

set(NETCDF_LINK_LIBRARIES ${NETCDF_LIBRARIES})
if(WIN32)
	list(APPEND NETCDF_LINK_LIBRARIES monetdb5 gdk)
endif()

add_library(netcdf MODULE netcdf.c netcdf_vault.h)
target_link_libraries(netcdf PRIVATE ${NETCDF_LINK_LIBRARIES})
set_target_properties(netcdf PROPERTIES OUTPUT_NAME _netcdf)
target_compile_definitions(netcdf PRIVATE LIBNCDF)

install(TARGETS netcdf DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:netcdf> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES netcdf.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 74_netcdf.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 74_netcdf.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
