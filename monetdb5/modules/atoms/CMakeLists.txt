#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../mal ../../../common/options ../../../common/stream ../../../common/utils ../../../gdk
					${LIBXML2_INCLUDE_DIRS} ${OPENSSL_INCS})

set(ATOM_ITEMS
	batxml.c
	blob.c blob.h
	color.c color.h
	identifier.c
	inet.c
	json.c json.h
	mtime.c mtime_analytic.c mtime.h
	str.c str.h
	streams.c streams.h
	url.c url.h
	uuid.c
	xml.c xml.h)
if(WIN32)
	list(APPEND ATOM_ITEMS strptime.c)
endif()
add_library(atoms OBJECT ${ATOM_ITEMS})
set_target_properties(atoms PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(FILES
		batxml.mal
		blob.mal
		color.mal
		identifier.mal
		inet.mal
		json.mal
		mtime.mal
		streams.mal
		str.mal
		url.mal
		uuid.mal
		xml.mal
		DESTINATION ${LIBDIR}/monetdb5)

if(HAVE_HGE)
	install(FILES 00_json_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
