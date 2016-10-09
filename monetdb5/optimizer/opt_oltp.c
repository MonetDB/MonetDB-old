/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/* author M.Kersten
 * This optimizer prepares a MAL block for locking
 */
#include "monetdb_config.h"
#include "opt_oltp.h"

static InstrPtr
addLock(InstrPtr lcks, MalBlkPtr mb, InstrPtr p, int sch, int tbl)
{
	int i;
	char buf[2 * IDLENGTH];

	snprintf(buf, 2 * IDLENGTH, "%s#%s", (sch?getVarConstant(mb, getArg(p,sch)).val.sval : "global"), (tbl? getVarConstant(mb, getArg(p,tbl)).val.sval : ""));
	// add unique table names only
	for( i=1; i< lcks->argc; i++)
		if( strcmp(buf, getVarConstant(mb,getArg(lcks,i)).val.sval) == 0)
			return lcks;
	lcks= pushStr(mb,lcks, buf);
	return lcks;
}

int
OPToltpImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	int i,j,k,limit,slimit;
	InstrPtr p, q, lcks;
	int actions = 0;
	InstrPtr *old;
	lng usec = GDKusec();
	char buf[256];

	//if ( !optimizerIsApplied(mb,"multiplex") )
		//return 0;
	(void) pci;
	(void) cntxt;
	(void) stk;		/* to fool compilers */

	old= mb->stmt;
	limit= mb->stop;
	slimit = mb->ssize;
	
	// First check and collect the binds needed.
	lcks= newInstruction(mb, ASSIGNsymbol);
	setModuleId(lcks, oltpRef);
	setFunctionId(lcks, lockRef);
	getArg(lcks,0)= newTmpVariable(mb, TYPE_void);

	for (i = 0; i < limit; i++) {
		p = old[i];
		if( getModuleId(p) == sqlRef && getFunctionId(p) == appendRef )
			lcks = addLock(lcks, mb, p, p->retc + 1, p->retc + 2);
		if( getModuleId(p) == sqlRef && getFunctionId(p) == updateRef )
			lcks = addLock(lcks, mb, p, p->retc + 1, p->retc + 2);
		if( getModuleId(p) == sqlRef && getFunctionId(p) == deleteRef )
			lcks = addLock(lcks, mb, p, p->retc + 1, p->retc + 2);
		if( getModuleId(p) == sqlcatalogRef )
			lcks = addLock(lcks, mb, p, 0,0);
	}
	
	if( lcks->argc == 1){
		freeInstruction(lcks);
		return 0;
	}

	// the lock names should be sorted
	for(i=1; i< lcks->argc; i++)
		for(j=1; j< lcks->argc; j++)
			if(strcmp(getVarConstant(mb,getArg(lcks,i)).val.sval, getVarConstant(mb,getArg(lcks,j)).val.sval) > 0){
				k = getArg(p,i);
				getArg(p,i)= getArg(p,j);
				getArg(p,j) = k;
			}
	
	// Now optimize the code
	if ( newMalBlkStmt(mb,mb->stop) < 0)
		return 0;
	pushInstruction(mb,0);
	pushInstruction(mb,lcks);
	for (i = 1; i < limit; i++) {
		p = old[i];
		if( p->token == ENDsymbol){
			// unlock all if there is an error
			q= newCatchStmt(mb,"MALexception");
			q= newExitStmt(mb,"MALexception");
			q= newCatchStmt(mb,"SQLexception");
			q= newExitStmt(mb,"SQLexception");
			q= copyInstruction(lcks);
			setFunctionId(q, releaseRef);
			pushInstruction(mb,q);
		}
		pushInstruction(mb,p);
	} 
	for(; i<slimit; i++)
		if (old[i]) 
			freeInstruction(old[i]);
	GDKfree(old);

    /* Defense line against incorrect plans */
	chkTypes(cntxt->fdout, cntxt->nspace, mb, FALSE);
	//chkFlow(cntxt->fdout, mb);
	//chkDeclarations(cntxt->fdout, mb);
    /* keep all actions taken as a post block comment */
    snprintf(buf,256,"%-20s actions=%2d time=" LLFMT " usec","emptybind",actions,GDKusec() - usec);
    newComment(mb,buf);
	return 1;
}
