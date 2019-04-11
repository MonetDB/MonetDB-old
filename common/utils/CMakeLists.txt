#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CRYPTO_INCLUDE_DIR})

add_library(mutils SHARED mutils.h mutils.c prompt.c mprompt.h revision.c)
target_compile_definitions(mutils PRIVATE LIBMUTILS LIBGDK LIBMEROUTIL)
install(TARGETS mutils DESTINATION ${LIBDIR})

add_library(mcrypt SHARED mcrypt.h mcrypt.c)
target_link_libraries(mcrypt PRIVATE ${CRYPTO_LIBRARIES})
target_compile_definitions(mcrypt PRIVATE LIBMAPI LIBMCRYPT)
install(TARGETS mcrypt DESTINATION ${LIBDIR})

add_library(msabaoth SHARED muuid.h muuid.c msabaoth.h msabaoth.c)
target_link_libraries(msabaoth PRIVATE ${CRYPTO_LIBRARIES} mutils)
target_compile_definitions(msabaoth PRIVATE LIBMSABAOTH LIBMUUID LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER LIBMONETDB5)
install(TARGETS msabaoth DESTINATION ${LIBDIR})
