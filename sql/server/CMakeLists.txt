#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(. ../include ../common ../storage ../../monetdb5/mal ../../monetdb5/modules/atoms
					../../monetdb5/modules/mal ../../monetdb5/extras/rapi ../../clients/mapilib
					../../common/stream ../../common/utils ../../gdk ${CMAKE_CURRENT_BINARY_DIR})

bison_target(sqlparser sql_parser.y ${CMAKE_CURRENT_BINARY_DIR}/sql_parser.tab.c
			COMPILE_FLAGS "-d -p sql -r all" DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/sql_parser.tab.h)
add_library(bison_obj OBJECT ${BISON_sqlparser_OUTPUTS})
set_target_properties(bison_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
check_c_compiler_flag("-Wno-unreachable-code" COMPILER_-Wno-unreachable-code_FLAG)
if(COMPILER_-Wno-unreachable-code_FLAG)
	target_compile_options(bison_obj PRIVATE -Wno-unreachable-code) # use this flag only to compile the bison output
endif()

add_library(sqlserver STATIC
			sql_atom.c
			sql_datetime.c
			sql_decimal.c
			sql_semantic.c
			sql_qc.c
			sql_symbol.c
			sql_scan.c
			sql_partition.c
			sql_mvc.c
			sql_env.c
			sql_privileges.c
			rel_semantic.c
			rel_select.c
			rel_updates.c
			rel_trans.c
			rel_schema.c
			rel_sequence.c
			rel_prop.c
			rel_exp.c
			rel_rel.c
			rel_optimizer.c
			rel_partition.c
			rel_planner.c rel_planner.h
			rel_distribute.c
			rel_remote.c rel_remote.h
			rel_propagate.c rel_propagate.h
			rel_psm.c
			rel_xml.c
			rel_dump.c
			rel_dump.h rel_exp.h rel_rel.h
			rel_optimizer.h rel_partition.h rel_distribute.h rel_prop.h rel_schema.h rel_select.h
			rel_semantic.h rel_sequence.h rel_trans.h
			rel_updates.h rel_psm.h rel_xml.h sql_atom.h sql_datetime.h
			sql_decimal.h sql_env.h sql_mvc.h sql_parser.h
			sql_privileges.h sql_qc.h sql_scan.h
			sql_semantic.h sql_symbol.h sql_tokens.h sql_partition.h)
target_link_libraries(sqlserver PRIVATE bison_obj)
set_target_properties(sqlserver PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlserver PRIVATE LIBSQL LIBSQLSERVER LIBSQLCOMMON LIBBATSTORE LIBSTORE)
