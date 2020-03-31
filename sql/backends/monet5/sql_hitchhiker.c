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
hh_move(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) cntxt;
    (void) mb;
    (void) stk;
    (void) pci;

    str *home_node, *landscape;
    int *next_node_idx, idx;

    home_node = getArgReference_str(stk, pci, 1);
    next_node_idx = getArgReference_int(stk, pci, 2);
    landscape = getArgReference_str(stk, pci, 3);

    // arguments start from 1!
    // jump over home_node, next_node_idx and landscape
    // and get the node that should be visited next
    idx = 3 + *next_node_idx;

    // modify the next_node_idx in the stack
    // so the next nodes knows where to jump
    *next_node_idx += 1;
    VALset(&stk->stk[pci->argv[1]], TYPE_int, &next_node_idx);
    next_node_idx = getArgReference_int(stk, pci, 2);

    // connect to the next node 
    // TODO

    
    return MAL_SUCCEED;
}
