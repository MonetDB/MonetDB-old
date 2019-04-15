#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../../../../clients/mapilib ../../../../../common/utils ../../../../../common/stream
					../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/mal ../../../../../gdk
					../../../../../monetdb5/mal ../../../../../sql/backends/monet5 ../../../../../sql/common
					../../../../../sql/include ../../../../../sql/server ../../../../../sql/storage
					${CMAKE_CURRENT_BINARY_DIR})

add_library(capi MODULE capi.c capi.h cheader.h ${CMAKE_CURRENT_BINARY_DIR}/cheader.text.h)
target_link_libraries(capi PRIVATE monetdb5 gdk)
set_target_properties(capi PROPERTIES OUTPUT_NAME _capi)
target_compile_definitions(capi PRIVATE LIBCUDF)

# Generate cheader.text.h file
add_custom_command(OUTPUT cheader COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/boot.py
				   ${CMAKE_CURRENT_SOURCE_DIR}/cheader.h ${CMAKE_CURRENT_BINARY_DIR}/cheader.text.h
				   BYPRODUCTS cheader.text.h)
add_custom_target(do_cheader DEPENDS cheader)
add_dependencies(capi do_cheader)

install(TARGETS capi DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:capi> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
endif()
install(FILES capi.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 50_capi.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
