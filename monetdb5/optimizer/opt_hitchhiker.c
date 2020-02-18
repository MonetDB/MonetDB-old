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



static int 
OPThitchhiker(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) cntxt;
	(void) pci;
	(void) stk;
    (void) mb;
    return 0;
}


str
OPThitchhikerImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) cntxt;

    str modnme;
	str fcnnme;
    str msg;
    Symbol s = NULL;
    int actions = 0;
    char buf[256];
    lng clk= GDKusec();

    if(pci)
        removeInstruction(mb, pci);

    if(pci && pci->argc > 1){
        if( getArgType(mb, pci, 1) != TYPE_str ||
            getArgType(mb, pci, 2) != TYPE_str ||
            !isVarConstant(mb, getArg(pci, 1)) ||
            !isVarConstant(mb, getArg(pci, 2))
        ) {
            throw(MAL, "optimizer.hitchhiker", ILLARG_CONSTANTS);
        }
        if(stk != 0) {
            modnme = *getArgReference_str(stk, pci, 1);
            fcnnme = *getArgReference_str(stk, pci, 2);
        } else {
            modnme = getArgDefault(mb, pci, 1);
            fcnnme = getArgDefault(mb, pci, 2);
        }
        s = findSymbol(cntxt->usermodule, putName(modnme), putName(fcnnme));

        if(s == NULL) {
            char buf[1024];
            snprintf(buf, 1024, "%s.%s", modnme, fcnnme);
            throw(MAL, "optimizer.hitchhiker", RUNTIME_OBJECT_UNDEFINED ":%s", buf);
        }

        mb = s->def;
        stk = 0;
    }

    if(mb->errors) {
        // when we have errors, we still want to see them
        addtoMalBlkHistory(mb);
        return MAL_SUCCEED;
    }

    // number of successfull changes to the code
    actions = OPThitchhiker(cntxt, mb, stk, pci);

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
