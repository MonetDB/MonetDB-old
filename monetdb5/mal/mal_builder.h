/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _MAL_BUILDER_
#define _MAL_BUILDER_

#include "mal.h"
#include "mal_instruction.h"

mal5_export InstrPtr newStmt(MalBlkPtr mb, const char *module, const char *name);
mal5_export InstrPtr newAssignment(MalBlkPtr mb);
mal5_export InstrPtr newComment(MalBlkPtr mb, const char *val);
mal5_export InstrPtr newCatchStmt(MalBlkPtr mb, str nme);
mal5_export InstrPtr newRaiseStmt(MalBlkPtr mb, str nme);
mal5_export InstrPtr newExitStmt(MalBlkPtr mb, str nme);
mal5_export InstrPtr newReturnStmt(MalBlkPtr mb);
mal5_export InstrPtr newFcnCall(MalBlkPtr mb, char *mod, char *fcn);
mal5_export InstrPtr pushSht(MalBlkPtr mb, InstrPtr q, sht val);
mal5_export InstrPtr pushEndInstruction(MalBlkPtr mb);
mal5_export InstrPtr pushInt(MalBlkPtr mb, InstrPtr q, int val);
mal5_export InstrPtr pushLng(MalBlkPtr mb, InstrPtr q, lng val);
#ifdef HAVE_HGE
mal5_export InstrPtr pushHge(MalBlkPtr mb, InstrPtr q, hge val);
#endif
mal5_export InstrPtr pushBte(MalBlkPtr mb, InstrPtr q, bte val);
mal5_export InstrPtr pushOid(MalBlkPtr mb, InstrPtr q, oid val);
mal5_export InstrPtr pushVoid(MalBlkPtr mb, InstrPtr q);
mal5_export InstrPtr pushDbl(MalBlkPtr mb, InstrPtr q, dbl val);
mal5_export InstrPtr pushFlt(MalBlkPtr mb, InstrPtr q, flt val);
mal5_export InstrPtr pushStr(MalBlkPtr mb, InstrPtr q, const char *val);
mal5_export InstrPtr pushBit(MalBlkPtr mb, InstrPtr q, bit val);
mal5_export InstrPtr pushNil(MalBlkPtr mb, InstrPtr q, int tpe);
mal5_export InstrPtr pushType(MalBlkPtr mb, InstrPtr q, int tpe);
mal5_export InstrPtr pushNilType(MalBlkPtr mb, InstrPtr q, char *tpe);
mal5_export InstrPtr pushZero(MalBlkPtr mb, InstrPtr q, int tpe);
mal5_export InstrPtr pushEmptyBAT(MalBlkPtr mb, InstrPtr q, int tpe);
mal5_export InstrPtr pushValue(MalBlkPtr mb, InstrPtr q, ValPtr cst);

mal5_export int getIntConstant(MalBlkPtr mb, int val);
mal5_export int getLngConstant(MalBlkPtr mb, lng val);
mal5_export int getShtConstant(MalBlkPtr mb, sht val);
mal5_export int getBteConstant(MalBlkPtr mb, bte val);
mal5_export int getOidConstant(MalBlkPtr mb, oid val);
mal5_export int getDblConstant(MalBlkPtr mb, dbl val);
mal5_export int getFltConstant(MalBlkPtr mb, flt val);
mal5_export int getStrConstant(MalBlkPtr mb, str val);
mal5_export int getBitConstant(MalBlkPtr mb, bit val);
#ifdef HAVE_HGE
mal5_export int getHgeConstant(MalBlkPtr mb, hge val);
#endif
#endif /* _MAL_BUILDER_ */

