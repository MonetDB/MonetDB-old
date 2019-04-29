#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(PY3_INCLUDE_DIR "")
set(PY3_LIBRARIES "")
if(HAVE_PYTHON3)
	set(PY3_INCLUDE_DIR "${Python3_INCLUDE_DIRS}")
	set(PY3_LIBRARIES "${Python3_LIBRARIES}")

	message(STATUS "Checking for numpy version")
	exec_program("${Python3_EXECUTABLE}" ARGS "-c \"import numpy; print(int(numpy.__version__.split('.').__getitem__(1)) >= 7)\"" OUTPUT_VARIABLE NUMPY_OUTPUT RETURN_VALUE NUMPY_RETURN_CODE)
	exec_program("${Python3_EXECUTABLE}" ARGS "-c \"import numpy; print(numpy.get_include())\"" OUTPUT_VARIABLE NUMPY_CPP_FLAGS RETURN_VALUE NUMPY_CPP_RC)
	if(NUMPY_RETURN_CODE EQUAL 0 AND NUMPY_CPP_RC EQUAL 0 AND "${NUMPY_OUTPUT}" STREQUAL "True")
		set(HAVE_LIBPY3 ON CACHE INTERNAL "pyapi3 extension is available")
		set(NUMPY3_INCLUDE_DIR ${NUMPY_CPP_FLAGS})
		message(STATUS "numpy version >= 1.7.0 for Python 3 found")
	elseif("${ENABLE_PY3INTEGRATION}" STREQUAL "YES")
		message(FATAL_ERROR "numpy version >= 1.7.0 required for Python 3 integration support")
	else()
		message(STATUS "numpy version >= 1.7.0 required for Python 3 integration support")
	endif()
elseif("${ENABLE_PY3INTEGRATION}" STREQUAL "YES")
	message(FATAL_ERROR "A valid Python 3 interpreter is required for Python 3 integration support")
endif()

if(HAVE_LIBPY3)
	include_directories(../../../../../clients/mapilib ../../../../../common/utils ../../../../../common/stream
						../../../../../monetdb5/modules/atoms ../../../../../monetdb5/modules/mal ../../../../../gdk
						../../../../../monetdb5/mal ../../../../../sql/backends/monet5 ../../../../../sql/common
						../../../../../sql/include ../../../../../sql/server ../../../../../sql/storage ../pyapi
						${PY3_INCLUDE_DIR} ${NUMPY3_INCLUDE_DIR})

	set(PYAPI3_LINK_LIBRARIES ${PY3_LIBRARIES})
	if(WIN32)
		list(APPEND PYAPI3_LINK_LIBRARIES monetdb5 gdk)
	endif()

	add_library(pyapi3 MODULE
				pyapi3.c unicode3.c pytypes3.c type_conversion3.c formatinput3.c connection3.c pyloader3.c emit3.c
				conversion3.c)
	target_link_libraries(pyapi3 PRIVATE ${PYAPI3_LINK_LIBRARIES})
	set_target_properties(pyapi3 PROPERTIES OUTPUT_NAME _pyapi3)
	target_compile_definitions(pyapi3 PRIVATE LIBPYAPI LIBPYAPI3)

	install(TARGETS pyapi3 DESTINATION ${LIBDIR}/monetdb5)
	if(WIN32)
		install(FILES $<TARGET_PDB_FILE:pyapi3> DESTINATION ${LIBDIR}/monetdb5 OPTIONAL)
		install(FILES pyapi_locatepython3.bat DESTINATION ${CMAKE_INSTALL_PREFIX})
	endif()
	install(FILES pyapi3.mal DESTINATION ${LIBDIR}/monetdb5)
	install(FILES 50_pyapi3.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
