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

mal_export void dumpWeldProgram(str program, FILE *f);
mal_export void getOrSetStructMember(char **addr, int type, const void *value, int op);
mal_export str getWeldType(int type);
mal_export str WeldRun(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

#endif
