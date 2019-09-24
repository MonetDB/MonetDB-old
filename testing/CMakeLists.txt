#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_executable(Mdiff difflib.c helpers.c Mdiff.c difflib.h helpers.h)
install(TARGETS Mdiff DESTINATION ${BINDIR})

string(REPLACE "\\" "\\\\" QXbindir "${BINDIR}")
string(REPLACE "\\" "\\\\" QXBUILD "${CMAKE_BINARY_DIR}")
string(REPLACE "\\" "\\\\" QXdatadir "${DATADIR}")
string(REPLACE "\\" "\\\\" QXdatarootdir "${DATAROOTDIR}")
string(REPLACE "\\" "\\\\" QXexec_prefix "${CMAKE_INSTALL_PREFIX}")
string(REPLACE "\\" "\\\\" QXincludedir "${INCLUDEDIR}")
string(REPLACE "\\" "\\\\" QXinfodir "${INFODIR}")
string(REPLACE "\\" "\\\\" QXlibdir "${LIBDIR}")
string(REPLACE "\\" "\\\\" QXlibexecdir "${LIBEXECDIR}")
string(REPLACE "\\" "\\\\" QXlocalstatedir "${LOCALSTATEDIR}")
string(REPLACE "\\" "\\\\" QXmandir "${MANDIR}")
string(REPLACE "\\" "\\\\" QXprefix "${CMAKE_INSTALL_PREFIX}")
string(REPLACE "\\" "\\\\" QXPYTHON3 "${Python3_EXECUTABLE}")
string(REPLACE "\\" "\\\\" QXPYTHON3_LIBDIR "${PYTHON3_LIBDIR}")
string(REPLACE "\\" "\\\\" QXSOURCE "${CMAKE_SOURCE_DIR}")
string(REPLACE "\\" "\\\\" QXsysconfdir "${SYSCONFDIR}")

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(BITS32_FALSE "#")
else()
	set(BITS64_FALSE "#")
endif()
if(WIN32)
	set(NATIVE_WIN32_FALSE "#")
else()
	set(NOT_WIN32_FALSE "#")
endif()
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(HAVE_MAL_DEBUGGER ON)
endif()

set(HAVE_LIST CUDF CURL FITS GEOM HGE LIBBZ2 LIBLZ4 LIBLZMA LIBPCRE LIBR LIBPY3 LIBXML LIBZ LIDAR MAL_DEBUGGER SHP 
	NETCDF ODBC PROJ SAMTOOLS)
foreach(loop_var IN LISTS HAVE_LIST)
	if(HAVE_${loop_var})
		set(HAVE_${loop_var}_FALSE "#")
	endif()
endforeach()

configure_file(Mtest.py.in ${CMAKE_CURRENT_BINARY_DIR}/Mtest.py @ONLY)
configure_file(listexports.py.in ${CMAKE_CURRENT_BINARY_DIR}/listexports.py @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Mtest.py PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION ${BINDIR})
if(PYTHON3_LIBDIR)
	install(FILES Mfilter.py process.py __init__.py ${CMAKE_CURRENT_BINARY_DIR}/listexports.py exportutils.py malcheck.py
			DESTINATION ${PYTHON3_LIBDIR}/MonetDBtesting)
endif()

if(WIN32)
	install(FILES Mlog.bat Mtest.py.bat DESTINATION ${BINDIR})
	install(FILES Mtest.py.bat DESTINATION ${BINDIR} RENAME Mapprove.py.bat)
	install(FILES $<TARGET_PDB_FILE:Mdiff> DESTINATION ${BINDIR} OPTIONAL)
else()
	if(NOT BASH)
		message(FATAL_ERROR "A bash interpreter was not found")
	endif()
	configure_file(Mlog.in ${CMAKE_CURRENT_BINARY_DIR}/Mlog @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Mlog PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION ${BINDIR})
	install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${BINDIR})")
	install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink Mtest.py ${BINDIR}/Mapprove.py)")
endif()
