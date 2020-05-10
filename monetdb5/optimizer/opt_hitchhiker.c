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
    (void) stk;
    (void) pci;

    str msg;
    char buf[256];
    int i, limit, slimit, updates = 0, actions = 0;
    size_t j;
    InstrPtr p, q, *old;
    lng clk = GDKusec();

    // Initialize the landscape - this is hardcoded it needs to be removed
    // It also assumes that the tables of the query have a replicated schema
    // and all tables exist on all nodes (see the Python script for more details)
    str home_node = "localhost:50000";
    int next_node_idx = 2;
    // const char* landscape[3] = {
    //     "localhost:50000:mdb1",
    //     "localhost:50001:mdb2",
    //     "localhost:50002:mdb3",
    // };

    const char* landscape[2] = {
        "localhost:50000:mdb1",
        "localhost:50001:mdb2"
    };
    size_t landscape_size = sizeof landscape / sizeof landscape[0];

    // check if optimizer has been applied
    if(optimizerIsApplied(mb, "hh"))
        return MAL_SUCCEED;

    limit = mb->stop;
    slimit = mb->ssize;
    old = mb->stmt;

    // count the number statements that need to be inserted
    // in practice, for every sql.tid we need a hh.move
    for(i = 0; i < limit; i++)
        if(getModuleId(mb->stmt[i]) == sqlRef && getFunctionId(mb->stmt[i]) == tidRef)
            updates++;

    if(updates)
    {
        // malloc new MAL block statement
        if (newMalBlkStmt(mb, mb->ssize + updates) < 0)
            throw(MAL, "hitchhiker.optimizer", SQLSTATE(HY013) MAL_MALLOC_FAIL);

        // locate sql.tid and inject hh.move calls before that
        for(i = 0; i < limit; i++)
        {
            p = old[i];

            // if instruction IS sql.tid first inject the new instruction first
            if(getModuleId(p) == sqlRef && getFunctionId(p) == tidRef)
            {
                // SOS: if table s1 is included - this needs to be removed 
                if(strcmp(getVarConstant(mb, getArg(p, 3)).val.sval, "s1") == 0) {

                    // do it if table is s1 - for now
                    // create a new instruction and push it
                    q = newInstruction(mb, hitchhikerRef, moveRef);

                    // fill home node
                    q = pushStr(mb, q, home_node);
                    
                    // next_arg
                    q = pushInt(mb, q, next_node_idx);

                    // landscape-fmt
                    // q = pushStr(mb, q, "sss");

                    // fill landscape info
                    for(j = 0; j < landscape_size; j++)
                        q = pushStr(mb, q, landscape[j]);

                    // getArg(q, 0) = newTmpVariable(mb, TYPE_any);    
                    setDestVar(q, newTmpVariable(mb, TYPE_any));
                    pushInstruction(mb, q);
                    actions++;
                }
            }

            // push the original instructions
            if(p)
                pushInstruction(mb, p);
        }

        // free old
        for (; i < slimit; i++)
            if (old[i])
                freeInstruction(old[i]);
        GDKfree(old);
    }

    // defense line against incorrect plans
    msg = chkTypes(cntxt->usermodule, mb, FALSE);
    if(msg == MAL_SUCCEED) msg = chkFlow(mb);
    if(msg == MAL_SUCCEED) msg = chkDeclarations(mb);

    clk = GDKusec() - clk;
    snprintf(buf, 256, "%-20s actions=%2d time=" LLFMT " usec", "hitchhiker", actions, clk);
    newComment(mb, buf);
    addtoMalBlkHistory(mb);

    return msg;
}
