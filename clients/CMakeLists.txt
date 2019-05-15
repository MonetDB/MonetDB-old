#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(${ENABLE_MAPI} STREQUAL "YES")
	set(HAVE_MAPI ON CACHE INTERNAL "mapi extension is available" FORCE)
	add_subdirectory(mapilib)
	add_subdirectory(mapiclient)
	add_subdirectory(examples)
else()
	set(HAVE_MAPI OFF CACHE INTERNAL "mapi extension is available" FORCE)
endif()
add_subdirectory(odbc)
if(WIN32)
	add_subdirectory(NT)
endif()
