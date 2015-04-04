/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * H. Muehleisen
 * The Python interface
 */
#ifndef _PYPI_LIB_
#define _PYPI_LIB_

#include "mal.h"
#include "mal_exception.h"
#include "mal_interpreter.h"

#define pyapi_export extern

pyapi_export str PyAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
pyapi_export str PyAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

pyapi_export str PyAPIprelude(void *ret);

int PyAPIEnabled(void);

#endif /* _PYPI_LIB_ */
