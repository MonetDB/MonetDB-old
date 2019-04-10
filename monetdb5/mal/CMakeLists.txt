#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

install(FILES
		mal.h
		mal_client.h
		mal_errors.h
		mal_exception.h
		mal_factory.h
		mal_function.h
		mal_instruction.h
		mal_interpreter.h
		mal_listing.h
		mal_module.h
		mal_namespace.h
		mal_profiler.h
		mal_resolve.h
		mal_stack.h
		mal_type.h
		DESTINATION ${INCLUDEDIR}/monetdb)
