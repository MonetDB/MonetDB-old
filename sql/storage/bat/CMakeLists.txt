#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(.. ../../include ../../common ../../../common/stream ../../../common/utils ../../../gdk
					../../../monetdb5/modules/atoms ../../../monetdb5/modules/kernel ../../../monetdb5/mal
					../../../monetdb5/modules/mal ../../../monetdb5/optimizer ../../../monetdb5/scheduler)

add_library(batstore STATIC
			bat_table.c bat_storage.c bat_logger.c bat_utils.c
			res_table.c
			bat_logger.h bat_storage.h bat_table.h bat_utils.h res_table.h)
set_target_properties(batstore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(batstore PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
