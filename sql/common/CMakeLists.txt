#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(.. ../include ../storage ../../common/options ../../common/stream ../../monetdb5/mal ../../gdk)

add_library(sqlcommon STATIC
			sql_mem.c sql_list.c sql_hash.c sql_stack.c sql_backend.c
			sql_keyword.c sql_changeset.c sql_types.c sql_string.c
			sql_backend.h sql_string.h sql_types.h)
set_target_properties(sqlcommon PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlcommon PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
