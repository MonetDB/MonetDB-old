/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2020 MonetDB B.V.
 */

/*
 * H. Muehleisen, M. Kersten
 * The R interface
 */
#ifndef _RAPI_LIB_
#define _RAPI_LIB_

#include "mal.h"
#include "mal_exception.h"
#include "mal_interpreter.h"

#ifdef WIN32
#ifndef LIBRAPI
#define rapi_export extern __declspec(dllimport)
#else
#define rapi_export extern __declspec(dllexport)
#endif
#else
#define rapi_export extern
#endif

#define RAPI_MAX_TUPLES 2147483647L

rapi_export str RAPIeval(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
		InstrPtr pci, bit grouped);
rapi_export str RAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
		InstrPtr pci);
rapi_export str RAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
		InstrPtr pci);
rapi_export void* RAPIloopback(void *query);
rapi_export str RAPIprelude(void *ret);

rapi_export void writeConsoleEx(const char * buf, int buflen, int foo);
rapi_export void writeConsole(const char * buf, int buflen);
rapi_export void clearRErrConsole(void);

char* rtypename(int rtypeid);

#if defined(WIN32) && !defined(HAVE_EMBEDDED)
// On Windows we need to dynamically load any SQL functions we use
// For embedded, this is not necessary because we create one large shared object
#define CREATE_SQL_FUNCTION_PTR(retval, fcnname)                               \
	typedef retval (*fcnname##_ptr_tpe)();                                     \
	fcnname##_ptr_tpe fcnname##_ptr = NULL;

#define LOAD_SQL_FUNCTION_PTR(fcnname)                                         \
	fcnname##_ptr = (fcnname##_ptr_tpe)getAddress(#fcnname);                   \
	if (fcnname##_ptr == NULL)                                                 \
		e = createException(MAL, "rapi.initialize", SQLSTATE(PY000) "Failed to load function %s", #fcnname);
#else
#define CREATE_SQL_FUNCTION_PTR(retval, fcnname)                               \
	typedef retval (*fcnname##_ptr_tpe)();                                     \
	fcnname##_ptr_tpe fcnname##_ptr = (fcnname##_ptr_tpe)fcnname;

#define LOAD_SQL_FUNCTION_PTR(fcnname) (void)fcnname
#endif

#endif /* _RAPI_LIB_ */
