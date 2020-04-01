/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2020 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "sql_hitchhiker.h"
#include "mapi.h"

str
hh_move(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) mb;
    (void) cntxt;

    const str delim = ":";
    str username = "monetdb";
    str password = "monetdb";
    str lang = "sql";
    str dbalias = "mdb1";
    str msg = MAL_SUCCEED;

    str *home_node, *landscape, *next_node, host, token;
    int *next_node_idx, port, idx;
    Mapi dbh;

    home_node = getArgReference_str(stk, pci, 1);
    next_node_idx = getArgReference_int(stk, pci, 2);
    landscape = getArgReference_str(stk, pci, 3);

    // arguments start from 1!
    // jump over home_node, next_node_idx and landscape
    // and get the node that should be visited next
    idx = 3 + *next_node_idx;
    next_node = getArgReference_str(stk, pci, idx);
    token = strtok(*next_node, delim);
    host = token;
    if(!host)
    {
        fprintf(stderr, "Could not parse connection string\n");
        return msg;
    }

    while(token != NULL) {
        token = strtok(NULL, delim);
        if(token)
            port = atoi(token);
    }

    if(!port)
    {
        fprintf(stderr, "Could not parse connection string\n");
        return msg;
    }

    // connect to the next node 
    // TODO: change dbname
    dbh = mapi_connect(host, port, username, password, lang, dbalias);
    if(mapi_error(dbh)) 
        fprintf(stderr, "Failed to connect to node %s:%d\n", host, port);

    fprintf(stderr, "Connect to node %s:%d\n", host, port);

    mapi_destroy(dbh); 
    fprintf(stderr, "Disconnected from node %s:%d\n", host, port);



    // modify the next_node_idx in the stack
    // so the next nodes knows where to jump
    *next_node_idx += 1;
    VALset(&stk->stk[pci->argv[1]], TYPE_int, &next_node_idx);
    next_node_idx = getArgReference_int(stk, pci, 2);
    
    return msg;
}
