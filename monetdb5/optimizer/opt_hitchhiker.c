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
    int i, limit, slimit, actions = 0;
    // InstrPtr p;
    lng clk = GDKusec();

    // if (newMalBlkStmt(mb, mb->ssize) < 0)
    //     throw(MAL, "optimizer.hitchhiker", SQLSTATE(HY013) MAL_MALLOC_FAIL);

    limit = mb->stop;
    slimit = mb->ssize;
    // old = mb->stmt;
    
    // sql in bindRef and tidRef
    for(i = 0; i < limit; i++)
    {
        if(getModuleId(mb->stmt[i]) == sqlRef && (getFunctionId(mb->stmt[i]) == bindRef || getFunctionId(mb->stmt[i]) == tidRef))
        {
            // if((q = copyInstruction(p)) == NULL) {
            //     for (; i < slimit; i++)
            //         if (old[i])
            //             freeInstruction(old[i]);
            //     GDKfree(old);

            //     return createException(MAL, "optimizer.hitchhiker", SQLSTATE(HY013) MAL_MALLOC_FAIL);
            // }

            setModuleId(mb->stmt[i], hitchhikerRef);
            actions++;
        }

        // if(p)
        //     pushInstruction(mb, p);
    }

    // free old
    // for (; i < slimit; i++)
    //     if (old[i])
    //         freeInstruction(old[i]);
    // GDKfree(old);

    if(mb->errors)
        throw(MAL, "optimizer.hitchhiker", SQLSTATE(42000) PROGRAM_GENERAL);

    // defense line against incorrect plans
    msg = chkTypes(cntxt->usermodule, mb, FALSE);
    if(msg == MAL_SUCCEED) msg = chkFlow(mb);
    if(msg == MAL_SUCCEED) msg = chkDeclarations(mb);

    clk = GDKusec() - clk;
    snprintf(buf, 256, "%-20s actions=%2d time=" LLFMT " usec", "optimizer.hitchhiker", actions, clk);
    newComment(mb,buf);
    addtoMalBlkHistory(mb);

    return msg;
}
