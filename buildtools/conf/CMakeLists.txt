#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

install(FILES Maddlog PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION ${BINDIR})

if(WIN32)
	install(FILES banner.bmp license.rtf monetdb.ico website.html DESTINATION ${BINDIR})
else()
	install(FILES monetdbd.conf DESTINATION ${SYSCONFDIR}/tmpfiles.d)
endif()
