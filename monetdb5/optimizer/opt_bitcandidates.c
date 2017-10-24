/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

/*
 * Use bit compressed candidate lists.
 */

#include "monetdb_config.h"
#include "mal_instruction.h"
#include "opt_bitcandidates.h"

str
OPTbitcandidatesImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i, j, k, actions =0, limit=0, slimit=0;
	InstrPtr *old, p,q;
	int *alias = 0;
	lng usec = GDKusec();
	char buf[256];

	(void) cntxt;
	(void) stk;
	(void) pci;

	limit = mb->stop;
	slimit = mb->ssize;
	old = mb->stmt;

	// first check number of Candidate use cases
	for (k= i = 0; i < mb->stop; i++) {
		p = old[i];
		for(j=0; j< p->argc; j++)
			k += isVarCList(mb, getArg(p,j)) > 0;
	}
	if ( k == 0)
		goto wrapup;

	alias = (int*) GDKzalloc( ( k + mb->vsize) * sizeof(int));
	if( alias == NULL)
		throw(MAL, "optimizer.bitcandidates", MAL_MALLOC_FAIL);
	for( i = 0; i < mb->vtop ; i++)
		alias[i] = i;

	if ( newMalBlkStmt(mb,mb->ssize + k) < 0){
		GDKfree(alias);
		throw(MAL, "optimizer.bitcandidates", MAL_MALLOC_FAIL);
	}

	for (i = 0; i < limit; i++) {
		p = old[i];
		for(j = p->retc; j< p->argc; j++){
		// decompress before use
			if ( isVarCList(mb, getArg(p,j)) && getArg(p,j) != alias[getArg(p,j)] ){
				q= newFcnCall(mb,candidatesRef,decompressRef);
				getArg(q,0) = getArg(p,j);
				q= pushArgument(mb,q, alias[getArg(p,j)]);
				alias[getArg(p,j)] = getArg(p,j);
			}
		}
		pushInstruction(mb,p);
		for(j=0; j< p->retc; j++){
		// compress after creation, avoid this step when you immediately  can use it
			if ( isVarCList(mb, getArg(p,j)) ){
				k = newTmpVariable(mb,getArgType(mb,p,j));
				setVarFixed(mb, k);
				q= newFcnCall(mb,candidatesRef,compressRef);
				setVarType(mb,getArg(q,0), getArgType(mb,p,j));
				setVarFixed(mb, getArg(q,0));
				q= pushArgument(mb,q, k);
				alias[getArg(p,j)] = getArg(q,0);
				getArg(p,j) = k;
				actions++;
			}
		}
	}
	for( ; i<slimit; i++)
		if( old[i])
			freeInstruction(old[i]);
#ifdef _OPT_BITCANDIDATES_DEBUG_
	fprintf(stderr,"#result of bitcandidates\n");
	fprintFunction(stderr,mb, 0, LIST_MAL_ALL);
#endif

wrapup:
	GDKfree(alias);
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
