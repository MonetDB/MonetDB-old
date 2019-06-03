/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _OIDX_H
#define _OIDX_H

#include "mal.h"
#include "mal_builder.h"
#include "mal_instruction.h"
#include "mal_interpreter.h"
#include "mal_namespace.h"

//#define _DEBUG_OIDX_
mal_export str OIDXcreate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str OIDXcreateImplementation(Client cntxt, int tpe, BAT *b, int pieces);
mal_export str OIDXdropImplementation(Client cntxt, BAT *b);
mal_export str OIDXorderidx(bat *ret, const bat *bid, const bit *stable);
mal_export str OIDXmerge(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str OIDXhasorderidx(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str OIDXgetorderidx(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
#endif /* _OIDX_H */
