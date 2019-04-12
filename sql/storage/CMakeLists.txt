#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_subdirectory(bat)

include_directories(. ../include ../common ../../common/stream ../../common/utils ../../gdk)

add_library(store STATIC
			store_dependency.c store_sequence.c
			store.c sql_catalog.c
			sql_storage.h store_dependency.h
			store_sequence.h)
set_target_properties(store PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(store PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
