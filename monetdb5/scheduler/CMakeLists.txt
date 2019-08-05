#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

add_library(scheduler OBJECT
			run_adder.c run_adder.h
			run_isolate.c run_isolate.h
			run_memo.c run_memo.h
			run_pipeline.c run_pipeline.h)
target_compile_definitions(scheduler PRIVATE LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)
target_link_libraries(scheduler PRIVATE utils_headers stream_headers gdk_headers mal_headers optimizer_headers)
set_target_properties(scheduler PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:scheduler> PARENT_SCOPE)

install(FILES run_adder.mal run_isolate.mal run_memo.mal DESTINATION ${LIBDIR}/monetdb5)
