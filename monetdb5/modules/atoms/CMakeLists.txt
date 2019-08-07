#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_library(atoms_headers INTERFACE)
target_include_directories(atoms_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						   $<INSTALL_INTERFACE:${INCLUDEDIR}/monetdb>)

if(HAVE_MONETDB5)
	include_directories(${LIBXML2_INCLUDE_DIR} ${CRYPTO_INCLUDE_DIR} ${UUID_INCLUDE_DIR})

	add_library(atoms OBJECT
				batxml.c
				blob.c blob.h
				color.c color.h
				identifier.c
				inet.c
				json.c json.h
				mtime.c mtime_analytic.c mtime.h
				str.c str.h
				streams.c streams.h
				strptime.c
				url.c url.h
				uuid.c
				xml.c xml.h)
	target_compile_definitions(atoms PRIVATE LIBMONETDB5)
	target_link_libraries(atoms PUBLIC atoms_headers PRIVATE utils_headers stream_headers gdk_headers mal_headers)
	set_target_properties(atoms PROPERTIES POSITION_INDEPENDENT_CODE ON)
	set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:atoms> PARENT_SCOPE)
endif()
