#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_subdirectory(udf)
if(NOT WIN32)
	add_subdirectory(capi)
endif()
if(HAVE_LIBPY)
	add_subdirectory(pyapi)
endif()
if(HAVE_LIBPY3)
	add_subdirectory(pyapi3)
endif()
