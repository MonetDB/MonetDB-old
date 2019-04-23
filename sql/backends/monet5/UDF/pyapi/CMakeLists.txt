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
					../../../../../sql/include ../../../../../sql/server ../../../../../sql/storage ${PY2_INCLUDE_DIR}
					${NUMPY2_INCLUDE_DIR})

set(PYAPI_LINK_LIBRARIES ${PY2_LIBRARIES})
if(WIN32)
	list(APPEND PYAPI_LINK_LIBRARIES monetdb5 gdk)
endif()

add_library(pyapi MODULE
			pyapi.c pyapi.h unicode.c unicode.h pytypes.c pytypes.h type_conversion.c type_conversion.h formatinput.c
			formatinput.h connection.c connection.h undef.h pyloader.c emit.h emit.c conversion.c conversion.h
			convert_loops.h pyheader.h)
target_link_libraries(pyapi PRIVATE ${PYAPI_LINK_LIBRARIES})
set_target_properties(pyapi PROPERTIES OUTPUT_NAME _pyapi)
target_compile_definitions(pyapi PRIVATE LIBPYAPI LIBPYAPI3)

install(TARGETS pyapi DESTINATION ${LIBDIR}/monetdb5)
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:pyapi> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
	install(FILES pyapi_locatepython2.bat DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()
install(FILES pyapi.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 50_pyapi.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
