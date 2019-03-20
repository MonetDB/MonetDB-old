#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_executable(Mdiff difflib.c helpers.c Mdiff.c difflib.h helpers.h)
target_link_libraries(Mdiff PRIVATE ${MALLOC_LIBS})
install(TARGETS Mdiff DESTINATION ${BINDIR})

string(REPLACE "\\" "\\\\" SOURCE "${CMAKE_SOURCE_DIR}")
string(REPLACE "\\" "\\\\" QXSOURCE "${CMAKE_SOURCE_DIR}") # TODO the QX variables are wrong!
string(REPLACE "\\" "\\\\" QXSOURCE "${CMAKE_INSTALL_PREFIX}")
string(REPLACE "\\" "\\\\" QXprefix "${CMAKE_INSTALL_PREFIX}")
string(REPLACE "\\" "\\\\" QXexec_prefix "${CMAKE_INSTALL_PREFIX}")
string(REPLACE "\\" "\\\\" QXlibexecdir "${LIBEXECDIR}")
string(REPLACE "\\" "\\\\" QXdatadir "${DATADIR}")
string(REPLACE "\\" "\\\\" QXdatarootdir "${DATAROOTDIR}")
string(REPLACE "\\" "\\\\" QXbindir "${BINDIR}")
string(REPLACE "\\" "\\\\" QXsysconfdir "${SYSCONFDIR}")
string(REPLACE "\\" "\\\\" QXlocalstatedir "${LOGDIR}")
string(REPLACE "\\" "\\\\" QXmandir "${MANDIR}")
string(REPLACE "\\" "\\\\" QXinfodir "${INFODIR}")
string(REPLACE "\\" "\\\\" QXlibdir "${LIBDIR}")
string(REPLACE "\\" "\\\\" QXincludedir "${INCLUDEDIR}")

string(REPLACE "\\" "\\\\" QXPYTHON2 "${PYTHON2_INTERPRETER}")
string(REPLACE "\\" "\\\\" QXPYTHON2_LIBDIR "${PYTHON2_LIBDIR}")
string(REPLACE "\\" "\\\\" QXPYTHON3 "${PYTHON3_INTERPRETER}")
string(REPLACE "\\" "\\\\" QXPYTHON3_LIBDIR "${PYTHON3_LIBDIR}")

find_program(SH sh)
if(NOT SH)
	message(FATAL_ERROR "A shell interpreter was not found")
endif()
set(BASH "${SH}")
find_program(PYTHON2_INT python2)
if(NOT SH)
	message(FATAL_ERROR "A shell interpreter was not found")
endif()

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
if(CMAKE_CROSSCOMPILING)
	set(CROSS_COMPILING_FALSE "#")
endif()
if(HAVE_LIBPCRE)
	set(HAVE_PCRE_FALSE "#")
endif()

set(HAVE_LIST
	CURL DEVELOPER FITS GDK GEOM HGE LIBBZ2 LIBLZ4 LIBLZMA LIBR LIBPY LIBPY3 LIBXML LIBZ LIDAR SHP MONETDB5 NETCDF ODBC
	PROJ PYTHON PYTHON2 PYTHON3 SAMTOOLS SQL TESTING)
foreach(loop_var IN LISTS HAVE_LIST)
	if(HAVE_${loop_var})
		set(HAVE_${loop_var}_FALSE "#")
	endif()
endforeach(loop_var)

configure_file(Mtest.py.in ${CMAKE_CURRENT_BINARY_DIR}/Mtest.py @ONLY)
configure_file(monet_options.py.in ${CMAKE_CURRENT_BINARY_DIR}/monet_options.py @ONLY)
configure_file(listexports.py.in ${CMAKE_CURRENT_BINARY_DIR}/listexports.py @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Mtest.py PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION ${BINDIR})
install(FILES Mfilter.py process.py ${CMAKE_CURRENT_BINARY_DIR}/monet_options.py __init__.py
		${CMAKE_CURRENT_BINARY_DIR}/listexports.py exportutils.py malcheck.py
		DESTINATION ${CMAKE_INSTALL_PREFIX}${PYTHON_LIBDIR}/MonetDBtesting)

if(WIN32)
	install(FILES Mlog.bat Mtest.py.bat DESTINATION ${BINDIR})
else()
	configure_file(Mlog.in ${CMAKE_CURRENT_BINARY_DIR}/Mlog @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Mlog PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION ${BINDIR})
	install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${BINDIR})")
	install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink Mtest.py ${BINDIR}/Mapprove.py)")
endif()
