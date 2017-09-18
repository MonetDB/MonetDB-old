/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

#ifndef _BITCANDIDATES_
#define _BITCANDIDATES_
#include "mal.h"
#include "mal_client.h"
#include "mal_interpreter.h"

mal_export str BCLcompress(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
mal_export str BCLdecompress(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
#endif
