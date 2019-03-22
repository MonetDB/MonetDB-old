#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(WIN32) # hack to link libmonetdb5, cmake has trouble linking static libraries into a single shared library while exposing symbols on Linux..
	set(LIBMONETDB5_LINKING STATIC)
else()
	set(LIBMONETDB5_LINKING OBJECT)
endif()

add_subdirectory(mal)
add_subdirectory(modules)
add_subdirectory(optimizer)
add_subdirectory(scheduler)
add_subdirectory(tools)
add_subdirectory(extras)
#if(WIN32)
#	add_subdirectory(NT)
#endif()
