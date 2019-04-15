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
					../../../../../sql/include ../../../../../sql/server ../../../../../sql/storage ../pyapi
					${PY3_INCLUDE_DIR})

# TODO check XDEPS
add_library(pyapi3 MODULE
			pyapi3.c unicode3.c pytypes3.c type_conversion3.c formatinput3.c connection3.c pyloader3.c emit3.c
			conversion3.c)
target_link_libraries(pyapi3 PRIVATE monetdb5 gdk ${PY3_LIBRARIES})
set_target_properties(pyapi3 PROPERTIES OUTPUT_NAME _pyapi3)
target_compile_definitions(pyapi3 PRIVATE LIBPYAPI LIBPYAPI3)

install(TARGETS pyapi3 DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:pyapi3> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	install(FILES pyapi_locatepython3.bat DESTINATION ${PREFIX})
endif()
install(FILES pyapi3.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 50_pyapi3.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
