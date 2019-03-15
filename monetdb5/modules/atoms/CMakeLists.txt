#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../mal ../../../common/options ../../../common/stream ../../../common/utils ../../../gdk
					${LIBXML2_INCLUDE_DIRS} ${OPENSSL_INCS})

add_library(atoms STATIC
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
	add_library(strptime STATIC strptime.c)
endif()
