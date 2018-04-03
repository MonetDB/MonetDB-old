/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */
#ifndef _MAL_WELD_H_
#define _MAL_WELD_H_

#include "mal.h"
#include "mal_interpreter.h"

#define OP_GET 0
#define OP_SET 1

typedef struct {
	char *program;
	InstrPtr *groupDeps;
	bit *cudfOutputs;
	size_t programMaxLen;
} weldState;

mal_export void getOrSetStructMember(char **addr, int type, const void *value, int op);
mal_export str WeldInitState(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldRun(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAggrSum(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraProjection(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldSqlProjectDelta(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraSelect1(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraSelect2(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraThetaselect1(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraThetaselect2(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcANDsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcADDsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcSUBsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcMULsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcDIVsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcMODsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcLTsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcLEsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcEQsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcGTsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcGEsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcNEsignal(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcIsNil(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcIfThenElse(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldGroup(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAggrSubSum(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAggrSubProd(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAggrSubMin(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAggrSubMax(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAggrSubCount(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatMtimeYear(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatMergeCand(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatMirror(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldBatcalcIdentity(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraJoin(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraDifference(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldAlgebraIntersect(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str WeldLanguagePass(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

#endif
