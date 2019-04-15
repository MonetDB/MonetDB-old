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
					../../../../../gdk ${LIBLAS_INCLUDE_DIR})

add_library(lidar MODULE lidar.c lidar.h)
target_link_libraries(lidar PRIVATE monetdb5 gdk ${LIBLAS_LIBRARIES})
set_target_properties(lidar PROPERTIES OUTPUT_NAME _lidar)
target_compile_definitions(lidar PRIVATE LIBLIDAR)

install(TARGETS lidar DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:lidar> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES lidar.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 75_lidar.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
install(FILES 75_lidar.sql DESTINATION ${LIBDIR}/monetdb5/createdb)
