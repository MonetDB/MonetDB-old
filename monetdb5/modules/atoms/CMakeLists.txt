#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../mal ../../../common/stream ../../../common/utils ../../../gdk ${LIBXML2_INCLUDE_DIR}
					${CRYPTO_INCLUDE_DIR} ${UUID_INCLUDE_DIR})

add_library(atoms OBJECT
			batxml.c
			blob.c blob.h
			color.c color.h
			identifier.c
			inet.c
			json.c json.h
			mtime.c mtime_analytic.c mtime.h mtime_private.h
			str.c str.h
			streams.c streams.h
			strptime.c
			url.c url.h
			uuid.c
			xml.c xml.h)
target_compile_definitions(atoms PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)
set_target_properties(atoms PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:atoms> PARENT_SCOPE)
