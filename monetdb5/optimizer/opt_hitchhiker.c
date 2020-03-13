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
    InstrPtr p, q, *old;
    lng clk = GDKusec();

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
                // create a new instruction and push it
                q = newInstruction(mb, hitchhikerRef, moveRef);
                getArg(q, 0) = newTmpVariable(mb, TYPE_any);
                // setDestVar(q, newTmpVariable(mb, Typ));
                pushInstruction(mb, q);
                actions++;
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
    snprintf(buf, 256, "%-20s actions=%2d time=" LLFMT " usec", "optimizer.hitchhiker", actions, clk);
    newComment(mb, buf);
    addtoMalBlkHistory(mb);

    return msg;
}
