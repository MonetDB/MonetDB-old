#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

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
