/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2020 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "sql_hitchhiker.h"


str
hh_tid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    return SQLtid(cntxt, mb, stk, pci);
}


str
hh_bind(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    return mvc_bind_wrap(cntxt, mb, stk, pci);
}
