#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../optimizer ../../mal ../../../gdk ../../../common/stream ../../../common/options)

add_library(opt_sql_append MODULE opt_sql_append.c opt_sql_append.h)
target_link_libraries(opt_sql_append PRIVATE monetdb5 gdk)
set_target_properties(opt_sql_append PROPERTIES RUNTIME_OUTPUT_NAME _opt_sql_append LIBRARY_OUTPUT_NAME _opt_sql_append
					  COMPILE_DEFINITIONS "LIBOPT_SQL_APPEND")

install(TARGETS opt_sql_append DESTINATION ${LIBDIR}/monetdb5)
install(FILES opt_sql_append.mal DESTINATION ${LIBDIR}/monetdb5)
install(FILES 91_opt_sql_append.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
