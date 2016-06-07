/*
 * The contents of this file are subject to the MonetDB Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.monetdb.org/Legal/MonetDBLicense
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is the MonetDB Database System.
 *
 * The Initial Developer of the Original Code is CWI.
 * Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
 * Copyright August 2008-2015 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * (author) M. Kersten
 * Assume simple queries . Clear out all non-iot schema related sql statements, except
 * for the bare minimum.
 */
/*
 * We keep a flow dependency table to detect.
 */
#include "monetdb_config.h"
#include "opt_iot.h"
#include "opt_deadcode.h"
#include "mal_interpreter.h"    /* for showErrors() */
#include "mal_builder.h"
#include "opt_statistics.h"
#include "opt_dataflow.h"

#define MAXBSKT 64
#define isstream(S,T) \
	for(fnd=0, k= 0; k< btop; k++) \
	if( strcmp(schemas[k], S)== 0 && strcmp(tables[k], T)== 0 ){ \
		fnd= 1; break;\
	}

int
OPTiotImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i, j, k, fnd, limit, slimit;
	InstrPtr r, p, *old;
	int *alias;
	str  schemas[MAXBSKT];
	str  tables[MAXBSKT];
	int  mvc[MAXBSKT];
	int done[MAXBSKT]= {0};
	int btop=0;
	int noerror=0;
	int cq;

	(void) pci;

	cq= strncmp(getFunctionId(getInstrPtr(mb,0)),"cq",2) == 0;
	old = mb->stmt;
	limit = mb->stop;
	slimit = mb->ssize;

	/* first analyse the query for streaming tables */
	for (i = 1; i < limit && btop <MAXBSKT; i++){
		p = old[i];
		if( getModuleId(p)== basketRef && (getFunctionId(p)== registerRef || getFunctionId(p)== bindRef || getFunctionId(p)== clear_tableRef)  ){
			OPTDEBUGiot mnstr_printf(cntxt->fdout, "#iot stream table %s.%s\n", getModuleId(p), getFunctionId(p));
			schemas[btop]= getVarConstant(mb, getArg(p,1)).val.sval;
			tables[btop]= getVarConstant(mb, getArg(p,2)).val.sval;
			mvc[btop] = getArg(p,0);
			for( j =0; j< btop ; j++)
			if( strcmp(schemas[j], schemas[j+1])==0  && strcmp(tables[j],tables[j+1]) ==0)
				break;
			mvc[j] = getArg(p,0);
			done[j]= done[j] || getFunctionId(p)== registerRef;
			if( j == btop)
				btop++;
		}
		if( getModuleId(p)== basketRef && (getFunctionId(p) == appendRef || getFunctionId(p) == deleteRef )){
			OPTDEBUGiot mnstr_printf(cntxt->fdout, "#iot stream table %s.%s\n", getModuleId(p), getFunctionId(p));
			schemas[btop]= getVarConstant(mb, getArg(p,2)).val.sval;
			tables[btop]= getVarConstant(mb, getArg(p,3)).val.sval;
			mvc[btop] = getArg(p,0);
			for( j =0; j< btop ; j++)
			if( strcmp(schemas[j], schemas[j+1])==0  && strcmp(tables[j],tables[j+1]) ==0)
				break;

			mvc[j] = getArg(p,0);
			if( j == btop)
				btop++;
		}
		if( getModuleId(p)== iotRef && getFunctionId(p) == basketRef){
			OPTDEBUGiot mnstr_printf(cntxt->fdout, "#iot stream table %s.%s\n", getModuleId(p), getFunctionId(p));
			schemas[btop]= getVarConstant(mb, getArg(p,1)).val.sval;
			tables[btop]= getVarConstant(mb, getArg(p,2)).val.sval;
			for( j =0; j< btop ; j++)
			if( strcmp(schemas[j], schemas[j+1])==0  && strcmp(tables[j],tables[j+1]) ==0)
				break;
			if( j == btop)
				btop++;
		}
	}
	if( btop == MAXBSKT || btop == 0)
		return 0;

	OPTDEBUGiot {
		mnstr_printf(cntxt->fdout, "#iot optimizer started\n");
		printFunction(cntxt->fdout, mb, stk, LIST_MAL_DEBUG);
	}
		(void) stk;

	alias = (int *) GDKzalloc(mb->vtop * 2 * sizeof(int));
	if (alias == 0)
		return 0;

	if (newMalBlkStmt(mb, slimit) < 0)
		return 0;

	pushInstruction(mb, old[0]);
	// register all baskets used
	for( j=0; j<btop; j++)
	if( done[j]==0) {
		p= newStmt(mb,basketRef,registerRef);
		p= pushStr(mb,p, schemas[j]);
		p= pushStr(mb,p, tables[j]);
	}
	for (i = 1; i < limit; i++)
		if (old[i]) {
			p = old[i];

			if(getModuleId(p) == sqlRef && getFunctionId(p)== transactionRef){
				freeInstruction(p);
				continue;
			}
			if (getModuleId(p) == sqlRef && getFunctionId(p) == tidRef ){
				isstream(getVarConstant(mb,getArg(p,2)).val.sval, getVarConstant(mb,getArg(p,3)).val.sval );
				if( fnd){
					alias[getArg(p,0)] = -1;
					freeInstruction(p);
				}
				continue;
			}
			if (getModuleId(p) == algebraRef && getFunctionId(p) == projectionRef && alias[getArg(p,1)] < 0){
				alias[getArg(p,0)] = getArg(p,2);
				freeInstruction(p);
				continue;
			}

			if (getModuleId(p) == sqlRef && getFunctionId(p) == affectedRowsRef ){
				for(j = 0; j < btop; j++){
					r =  newStmt(mb, basketRef, commitRef);
					if (alias[mvc[j]] > 0)
						r =  pushArgument(mb,r, alias[mvc[j]]);
					else
						r =  pushArgument(mb,r, mvc[j]);
					r =  pushStr(mb,r, schemas[j]);
					r =  pushStr(mb,r, tables[j]);
				}
				freeInstruction(p);
				continue;
			}

			if( getModuleId(p)== iotRef && getFunctionId(p)==errorRef)
				noerror++;
			if (p->token == ENDsymbol && btop > 0 && noerror==0) {
				// empty all baskets used only when we are optimizing a cq
				for( j=0; cq && j<btop; j++)
				if( done[j]==0) {
					p= newStmt(mb,basketRef,finishRef);
					p= pushStr(mb,p, schemas[j]);
					p= pushStr(mb,p, tables[j]);
				}

				/* catch any exception left behind */
				r = newAssignment(mb);
				j = getArg(r, 0) = newVariable(mb, GDKstrdup("SQLexception"), TYPE_str);
				setVarUDFtype(mb, j);
				r->barrier = CATCHsymbol;

				r = newStmt(mb,iotRef, errorRef);
				r = pushStr(mb, r, getModuleId(old[0]));
				r = pushStr(mb, r, getFunctionId(old[0]));
				r = pushArgument(mb, r, j);

				r = newAssignment(mb);
				getArg(r, 0) = j;
				r->barrier = EXITsymbol;
				r = newAssignment(mb);
				j = getArg(r, 0) = newVariable(mb, GDKstrdup("MALexception"), TYPE_str);
				setVarUDFtype(mb, j);
				r->barrier = CATCHsymbol;

				r = newStmt(mb,iotRef, errorRef);
				r = pushStr(mb, r, getModuleId(old[0]));
				r = pushStr(mb, r, getFunctionId(old[0]));
				r = pushArgument(mb, r, j);

				r = newAssignment(mb);
				getArg(r, 0) = j;
				r->barrier = EXITsymbol;
				break;
			}


			for (j = 0; j < p->argc; j++)
				if (alias[getArg(p, j)] > 0)
					getArg(p, j) = alias[getArg(p, j)];

			if (getModuleId(p) == sqlRef && getFunctionId(p) == appendRef ){
				isstream(getVarConstant(mb,getArg(p,2)).val.sval, getVarConstant(mb,getArg(p,3)).val.sval );
				/* the appends come in multiple steps.
				   The first initializes an basket update statement,
				   which is triggered when we commit the transaction.
				 */
			}
			pushInstruction(mb, p);
		}

    /* take the remainder as is */
    for (; i<limit; i++)
        if (old[i])
            pushInstruction(mb,old[i]);

    /* Defense line against incorrect plans */
	chkTypes(cntxt->fdout, cntxt->nspace, mb, FALSE);
	chkFlow(cntxt->fdout, mb);
	chkDeclarations(cntxt->fdout, mb);

	OPTDEBUGiot {
		mnstr_printf(cntxt->fdout, "#iot optimizer final\n");
		printFunction(cntxt->fdout, mb, stk, LIST_MAL_DEBUG);
	} 
	GDKfree(alias);
	return btop > 0;
}

