/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "opt_datavaults.h"
#include "mal_interpreter.h"	/* for showErrors() */
#include "mal_builder.h"
/*
 * The instruction sent is produced with a variation of call2str
 * from the debugger.
 */

#define DEFAULT_NUM_TABLES 512

typedef enum {
    NOT_LOADED,
    SUBMITTED,
    LOADED
} STATUS;

typedef struct TABLE_ {
    str sname;
    str tname;
    int arg0;
} TABLE;

static str
checkTable(int *res, int *action, MalBlkPtr mb, InstrPtr p, TABLE **tabs, int num_tabs, str sname, str tname) {
    int i = 0, j = 0;
    InstrPtr c, a, r;
    int upd = (getFunctionId(p)==bindRef) && (p->argc == 7 || p->argc == 9);
    (void) *action;

    if ( num_tabs && ((num_tabs % DEFAULT_NUM_TABLES) == 0)) {
		*tabs = (TABLE*) GDKrealloc(tabs, sizeof(TABLE) * (num_tabs*2));
        if (!tabs) {
            throw(MAL, "optimizer.datavaults", "Realloc failed");
        }
    }
    for(i = 0; i < num_tabs; i++) {
        if ( tabs[i]->sname && (strcmp(tabs[i]->sname, sname) == 0) && (strcmp(tabs[i]->tname, tname) == 0)) { 
            r = newInstruction(mb,ASSIGNsymbol);
            setModuleId(r, vaultRef);
            setFunctionId(r, getFunctionId(p));
            getArg(r,0) = getArg(p,0);
            for (j = 1; j < p->retc; j++) {
                r = pushArgument(mb, r, getArg(p,j));
            }
            r= pushArgument(mb, r, tabs[i]->arg0);
            for (j = p->retc; j < p->argc; j++) {
                r = pushArgument(mb, r, getArg(p,j));
            }
            r->retc = p->retc;
            pushInstruction(mb,r);
            *res = num_tabs;
            *action = *action+1;
            return MAL_SUCCEED;
        }
    }

    c = newInstruction(mb,ASSIGNsymbol);
    setModuleId(c, vaultRef);
    setFunctionId(c, checktableRef);
    getArg(c,0) = newTmpVariable(mb, TYPE_int);
	setVarFixed(mb,getArg(c,0));
    c = pushArgument(mb, c, newTmpVariable(mb, TYPE_int));
    c = pushArgument(mb, c, getArg(p,1+upd));
    c = pushArgument(mb, c, getArg(p,3+upd));
    c->retc = 2;
    c->argc = 4;
    pushInstruction(mb,c);

    a = newInstruction(mb,ASSIGNsymbol);
    setModuleId(a, vaultRef);
    setFunctionId(a, analyzetableRef);
    getArg(a,0) = newTmpVariable(mb, TYPE_int);
    a = pushArgument(mb, a, getArg(c,0));
    a = pushArgument(mb, a, getArg(c,1));
    a = pushArgument(mb, a, getArg(p,3+upd));
    a->retc = 1;
    pushInstruction(mb,a);

    r = newInstruction(mb,ASSIGNsymbol);
    setModuleId(r, vaultRef);
    setFunctionId(r, getFunctionId(p));
    getArg(r,0) = getArg(p,0);
    for (j = 1; j < p->retc; j++) {
        r = pushArgument(mb, r, getArg(p,j));
    }
    r= pushArgument(mb, r, getArg(a,0));
    for (j = p->retc; j < p->argc; j++) {
        r = pushArgument(mb, r, getArg(p,j));
    }
    //r->retc = p->retc;
    printf("R argc %d p argc %d \n", r->argc, p->argc);
    pushInstruction(mb,r);

    /*Add info about the table*/
    if (!tabs[num_tabs])
        tabs[num_tabs] = (TABLE*) GDKmalloc(sizeof(TABLE));
    tabs[num_tabs]->sname = GDKstrdup(sname);
    tabs[num_tabs]->tname = GDKstrdup(tname);
    tabs[num_tabs]->arg0 = getArg(a,0);

    *res = ++num_tabs;
    *action = *action+2;

    return MAL_SUCCEED;
}

int
OPTdatavaultsImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    InstrPtr p, *old;
    int i, limit, slimit, action=0;
    TABLE **tabs = NULL;
    int num_tabs = 0;

	mnstr_printf(cntxt->fdout, "Datavaults optimizer started\n");
	(void) cntxt;
	(void) stk;
	(void) pci;

	limit = mb->stop;
	slimit = mb->ssize;
	old = mb->stmt;

	if ( newMalBlkStmt(mb, mb->ssize) < 0){
		return 0;
	}

    /*Set auxiliar structures*/
    tabs = (TABLE**) GDKzalloc(sizeof(TABLE*) * DEFAULT_NUM_TABLES);

    for (i = 0; i < limit; i++) {
        p = old[i];
        if( getModuleId(p)== sqlRef && ((getFunctionId(p)==bindRef) || (getFunctionId(p)==tidRef)) ){
            int upd = (getFunctionId(p)==bindRef) && (p->argc == 7 || p->argc == 9);
            str sname = getVarConstant(mb, getArg(p,2 + upd)).val.sval;
            str tname = getVarConstant(mb, getArg(p,3 + upd)).val.sval;
            checkTable(&num_tabs, &action, mb, p, tabs, num_tabs, sname, tname);
            continue;
        }
        pushInstruction(mb,p);
    }

    for(; i<slimit; i++)
        if( old[i])
            freeInstruction(old[i]);
    GDKfree(old);

    if (tabs) {
        for (i = 0; i < num_tabs; i++) {
            if (tabs[i]) {
				GDKfree(tabs[i]->sname);
				GDKfree(tabs[i]->tname);
                GDKfree(tabs[i]);
            }
        }
        GDKfree(tabs);
    }


#ifdef DEBUG_OPT_DATAVAULTS
	if (0 && action) {
		mnstr_printf(cntxt->fdout, "datavaults %d\n", action);
		printFunction(cntxt->fdout, mb, 0, LIST_MAL_ALL);
	}
#endif
	return action;
}
