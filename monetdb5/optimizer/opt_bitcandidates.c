/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

/*
 * Use bit compressed candidate lists on selected operations.
 */

#include "monetdb_config.h"
#include "mal_instruction.h"
#include "opt_bitcandidates.h"

str
OPTbitcandidatesImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i, j, actions =0, limit=0, slimit=0;
	InstrPtr *old, p,q;
	lng usec = GDKusec();
	char buf[256];

	(void) cntxt;
	(void) stk;
	(void) pci;

	limit = mb->stop;
	slimit = mb->ssize;
	old = mb->stmt;

	if ( newMalBlkStmt(mb,mb->ssize ) < 0){
		throw(MAL, "optimizer.bitcandidates", MAL_MALLOC_FAIL);
	}

	/* decompress all MSK columns before they are used */
	for (i = 0; i < limit; i++) {
		p = old[i];
		for(j = p->retc; j< p->argc; j++)
			// decompress mask candidates before use
			if(  getBatType(getArgType(mb,p,j)) == TYPE_msk){
				q= newFcnCall(mb, candidatesRef, decompressRef);
				setVarType( mb, getArg(q,0), newBatType(TYPE_oid));
				q= pushArgument(mb,q, getArg(p,j));
				getArg(p,j) = getArg(q,0);
				actions++;
			}
		// thetaselect is the first operation to produce a msk
		pushInstruction(mb,p);
		if( getFunctionId(p) == thetaselectRef && getModuleId(p) == algebraRef){
			setVarType( mb, getArg(p,0), newBatType(TYPE_msk));
			actions++;
		}
		if( getFunctionId(p) == selectRef && getModuleId(p) == algebraRef){
			setVarType( mb, getArg(p,0), newBatType(TYPE_msk));
			actions++;
		}
	}
	for( ; i<slimit; i++)
		if( old[i])
			freeInstruction(old[i]);
#ifdef _OPT_BITCANDIDATES_DEBUG_
	fprintf(stderr,"#result of bitcandidates\n");
	fprintFunction(stderr,mb, 0, LIST_MAL_ALL);
#endif

    /* Defense line against incorrect plans */
	if( actions){
		chkTypes(cntxt->usermodule, mb, FALSE);
		chkFlow(mb);
		chkDeclarations(mb);
	}
    /* keep all actions taken as a post block comment */
	usec = GDKusec()- usec;
    snprintf(buf,256,"%-20s actions=%d time=" LLFMT " usec","bitcandidates", actions, usec);
    newComment(mb,buf);
	addtoMalBlkHistory(mb);

	return MAL_SUCCEED;
}
