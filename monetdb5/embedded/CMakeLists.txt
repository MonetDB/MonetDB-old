#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../common/stream ../../common/utils ../../gdk ../mal ../modules/atoms ../modules/kernel
					../modules/mal ../optimizer ../scheduler)

add_library(malembedded OBJECT mal_embedded.c mal_embedded.h)
target_compile_definitions(malembedded PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)
set_target_properties(malembedded PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:malembedded> PARENT_SCOPE)
