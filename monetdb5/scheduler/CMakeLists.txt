#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../mal ../optimizer ../modules/atoms ../modules/mal ../modules/kernel ../../clients/mapilib
					../../common/stream ../../common/utils ../../gdk)

add_library(scheduler OBJECT
			run_adder.c run_adder.h
			run_isolate.c run_isolate.h
			run_memo.c run_memo.h
			run_pipeline.c run_pipeline.h)
target_compile_definitions(scheduler PRIVATE LIBMONETDB5)
set_target_properties(scheduler PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:scheduler> PARENT_SCOPE)
