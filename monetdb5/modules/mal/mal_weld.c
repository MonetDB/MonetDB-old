/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "gdk.h"
#include "mal_exception.h"
#include "mal_interpreter.h"

mal_export str WeldInitState(ptr *retval);
str
WeldInitState(ptr *retval)
{
	(void) retval;
	return MAL_SUCCEED;
}

mal_export str WeldRun(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldRun(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldGetResult(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldGetResult(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldAggrSum(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldAggrSum(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldAlgebraProjection(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldAlgebraProjection(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldAlgebraSelect1(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldAlgebraSelect1(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldAlgebraSelect2(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldAlgebraSelect2(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldAlgebraThetaselect1(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldAlgebraThetaselect1(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldAlgebraThetaselect2(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldAlgebraThetaselect2(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldBatcalcMULsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
str
WeldBatcalcMULsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

mal_export str WeldLanguagePass(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

str
WeldLanguagePass(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}
