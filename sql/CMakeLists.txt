#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_subdirectory(include)
add_subdirectory(common)
add_subdirectory(storage)
add_subdirectory(server)
add_subdirectory(backends)
if(WIN32)
	add_subdirectory(NT)
endif()

install(FILES dump-restore.txt dump-restore.html DESTINATION ${DATADIR}/doc/MonetDB-SQL)
