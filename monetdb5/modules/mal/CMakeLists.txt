#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../mal ../atoms ../kernel ../../../clients/mapilib ../../../common/stream ../../../common/utils
					../../../gdk ${PCRE_INCLUDE_DIR} ${CRYPTO_INCLUDE_DIR})

add_library(malmodules OBJECT
			batExtensions.c batExtensions.h
			bbp.c bbp.h
			clients.c clients.h
			factories.c factories.h
			groupby.c groupby.h
			inspect.c inspect.h
			orderidx.c orderidx.h
			iterator.c  iterator.h
			projectionpath.c  projectionpath.h
			language.c language.h
			mal_io.c mal_io.h
			mal_mapi.c mal_mapi.h
			manual.c manual.h
			mat.c mat.h
			mdb.c mdb.h
			mkey.c mkey.h
			manifold.c manifold.h
			oltp.c oltp.h
			wlc.c wlc.h
			pcre.c
			profiler.c profiler.h
			querylog.c querylog.h
			remote.c remote.h
			sysmon.c sysmon.h
			tablet.c tablet.h
			tokenizer.c tokenizer.h
			transaction.c
			txtsim.c txtsim.h
			sample.c sample.h
			json_util.c json_util.h
			calc.c batcalc.c)
target_compile_definitions(malmodules PRIVATE LIBMONETDB5)
set_target_properties(malmodules PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MONETDB5_OBJECTS ${MONETDB5_OBJECTS} $<TARGET_OBJECTS:malmodules> PARENT_SCOPE)

install(FILES mal_init.mal DESTINATION ${LIBDIR}/monetdb5)
