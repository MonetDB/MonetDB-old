/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _MAL_FCN_H
#define _MAL_FCN_H

#include "mal_instruction.h"
#include "mal_module.h"
#include "mal_resolve.h"

#define getLastUpdate(L,I)	((L)->var[I].updated)
#define getEndScope(L,I)	((L)->var[I].eolife)
#define getBeginScope(L,I)	((L)->var[I].declared)

/* #define DEBUG_MAL_FCN */
/* #define DEBUG_CLONE */

mal5_export Symbol   newFunction(str mod, str nme,int kind);
mal5_export int      getPC(MalBlkPtr mb, InstrPtr p);

mal5_export Symbol   getFunctionSymbol(Module scope, InstrPtr p);
mal5_export void chkFlow(MalBlkPtr mb);
mal5_export void chkDeclarations(MalBlkPtr mb);
mal5_export void clrDeclarations(MalBlkPtr mb);
mal5_export int isLoopBarrier(MalBlkPtr mb, int pc);
mal5_export int getBlockExit(MalBlkPtr mb,int pc);
mal5_export int getBlockBegin(MalBlkPtr mb,int pc);
mal5_export void setVariableScope(MalBlkPtr mb);

mal5_export void printFunction(stream *fd, MalBlkPtr mb, MalStkPtr stk, int flg);
mal5_export void fprintFunction(FILE *fd, MalBlkPtr mb, MalStkPtr stk, int flg);
mal5_export void debugFunction(stream *fd, MalBlkPtr mb, MalStkPtr stk, int flg, int first, int size);

#include "mal_exception.h"

#define MAXDEPTH 32
#endif /*  _MAL_FCN_H*/
