/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2020 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "opt_hitchhiker.h"
#include "mal_interpreter.h"

str
OPThitchhikerImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) cntxt;
    (void) mb;
    (void) stk;
    (void) pci;

    str msg = MAL_SUCCEED;
    return msg;
}