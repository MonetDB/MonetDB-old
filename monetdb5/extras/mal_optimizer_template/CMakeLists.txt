#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../optimizer ../../mal ../../../gdk ../../../common/stream ../../../common/options)

add_library(opt_sql_append MODULE opt_sql_append.c opt_sql_append.h)
add_dependencies(opt_sql_append monetdb5 bat)
target_link_libraries(opt_sql_append monetdb5 bat)
