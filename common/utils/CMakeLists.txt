#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CRYPTO_INCLUDE_DIR})

set(MUTILS_OBJECTS mutils.h mutils.c prompt.c mprompt.h revision.c)
set(MCRYPT_OBJECTS mcrypt.h mcrypt.c)
set(MSABAOTH_OBJECTS muuid.h muuid.c msabaoth.h msabaoth.c)

if(WIN32)
	add_library(mutils SHARED ${MUTILS_OBJECTS})
	target_compile_definitions(mutils PRIVATE LIBMUTILS LIBGDK LIBMEROUTIL)
	install(TARGETS mutils DESTINATION ${LIBDIR})

	add_library(mcrypt SHARED ${MCRYPT_OBJECTS})
	target_compile_definitions(mcrypt PRIVATE LIBMAPI LIBMCRYPT)
	install(TARGETS mcrypt DESTINATION ${LIBDIR})

	add_library(msabaoth SHARED ${MSABAOTH_OBJECTS})
	target_compile_definitions(msabaoth PRIVATE LIBMSABAOTH LIBMUUID LIBMAL LIBATOMS LIBKERNEL LIBOPTIMIZER LIBSCHEDULER
							   LIBMONETDB5)
	install(TARGETS msabaoth DESTINATION ${LIBDIR})
else()
	add_library(mutils STATIC ${MUTILS_OBJECTS})
	set_target_properties(mutils PROPERTIES POSITION_INDEPENDENT_CODE ON)

	add_library(mcrypt STATIC ${MCRYPT_OBJECTS})
	set_target_properties(mcrypt PROPERTIES POSITION_INDEPENDENT_CODE ON)

	add_library(msabaoth STATIC ${MSABAOTH_OBJECTS})
	set_target_properties(msabaoth PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(mcrypt PRIVATE ${CRYPTO_LIBRARIES})
target_link_libraries(msabaoth PRIVATE ${CRYPTO_LIBRARIES} mutils)
