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
 * Copyright August 2008-2016 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * The interface from SQL passes through here.
 */

#include "monetdb_config.h"
#include "sql_optimizer.h"
#include "sql_gencode.h"
#include "opt_timetrails.h"
#include "basket.h"
#include "petrinet.h"

MT_Lock ttrLock MT_LOCK_INITIALIZER("timetrailsLock");

// locate the SQL procedure in the catalog
static str
TTRprocedureStmt(Client cntxt, MalBlkPtr mb, str schema, str nme)
{
	mvc *m = NULL;
	str msg = MAL_SUCCEED;
	sql_schema  *s;
	backend *be;
	node *o;
	sql_func *f;
	/*sql_trans *tr;*/

	msg = getSQLContext(cntxt, mb, &m, NULL);
	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
		return msg;
	s = mvc_bind_schema(m, schema);
	if (s == NULL)
		throw(SQL, "ttr.query", "Schema missing");
	/*tr = m->session->tr;*/
	for (o = s->funcs.set->h; o; o = o->next) {
		f = o->data;
		if (strcmp(f->base.name, nme) == 0) {
			be = (void *) backend_create(m, cntxt);
			if ( be->mvc->sa == NULL)
				be->mvc->sa = sa_create();
			//TODO fix result type
			backend_create_func(be, f, f->res,NULL);
			return MAL_SUCCEED;
		}
	}
	throw(SQL, "ttr.query", "Procedure missing");
}

/* locate the MAL representation of this operation and extract the flow */
/* If the operation is not available yet, it should be compiled from its
   definition retained in the SQL catalog */
str
TTRquery(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str nme= NULL;
	str def= NULL;
	int calls = -1;

	Symbol s = NULL;
	MalBlkPtr qry;
	str msg = NULL;
	InstrPtr p;
	char buf[BUFSIZ], name[IDLENGTH];
	static int ttrquerycnt=0;
	int i;

	if( pci->argc > 2)
		calls = *getArgReference_int(stk,pci,2);

	_DEBUG_TTR_ fprintf(stderr,"#ttr: register the continues query %s.%s()\n",userRef,nme);

	// pre-create the new procedure
	snprintf(name, IDLENGTH,"cquery_%d",ttrquerycnt++);
	def = *getArgReference_str(stk, pci, 1);
	// package it as a procedure in the current schema [todo]
	snprintf(buf,BUFSIZ,"create procedure %s.%s() begin %s; end",userRef,name,def);
	_DEBUG_TTR_ fprintf(stderr,"#ttr.compile: %s\n",buf);
	nme = name;
	msg = SQLstatementIntern(cntxt, &def, nme, 1, 0, 0);
	if (msg)
		return msg;
	qry = cntxt->curprg->def;

	chkProgram(cntxt->fdout,cntxt->nspace,qry);
	if( qry->errors)
		msg = createException(SQL,"ttr.query","Error in timetrails query");

	_DEBUG_TTR_ fprintf(stderr,"#ttr: register a new continuous query plan\n");
	s = newFunction(userRef, putName(nme), FUNCTIONsymbol);
	if (s == NULL)
		msg = createException(SQL, "ttr.query", "Procedure code does not exist.");

	freeMalBlk(s->def);
	s->def = copyMalBlk(qry);
	p = getInstrPtr(s->def, 0);
	setModuleId(p, userRef);
	setFunctionId(p, putName(nme));
	insertSymbol(cntxt->nspace, s);
	_DEBUG_TTR_ printFunction(cntxt->fdout, s->def, 0, LIST_MAL_ALL);
	/* optimize the code and register at scheduler */
	if (msg == MAL_SUCCEED) 
		addtoMalBlkHistory(mb);
	if (msg == MAL_SUCCEED) {
		_DEBUG_TTR_ fprintf(stderr,"#ttr: continuous query plan\n");
		msg = PNregisterInternal(cntxt, s->def,calls);
	}

	// register all the baskets mentioned in the plan
	for( i=1; i< s->def->stop;i++){
		p= getInstrPtr(s->def,i);
		if( getModuleId(p) == basketRef && getFunctionId(p)== registerRef){
			BSKTregister(cntxt,s->def,0,p);
		}
	}
	return msg;
}
str
TTRqueryProc(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch= NULL;
	str nme= NULL;
	int calls = -1;

	Symbol s = NULL;
	MalBlkPtr qry;
	str msg = NULL;
	InstrPtr p;
	//Module scope;
	int i;
	char name[IDLENGTH];

	_DEBUG_TTR_ fprintf(stderr,"#ttr: register the continues query %s.%s()\n",sch,nme);

	/* check existing of the pre-compiled and activated function */
	sch = *getArgReference_str(stk, pci, 1);
	nme = *getArgReference_str(stk, pci, 2);
	snprintf(name,IDLENGTH,"%s_%s",sch,nme);

	if( pci->argc > 3)
		calls = *getArgReference_int(stk,pci,3);

	/* check existing of the pre-compiled function */
	_DEBUG_TTR_ fprintf(stderr,"#ttr: locate a SQL procedure %s.%s()\n",sch,nme);
	msg = TTRprocedureStmt(cntxt, mb, sch, nme);
	if (msg)
		return msg;
	s = findSymbolInModule(cntxt->nspace, putName(nme));
	if (s == NULL)
		throw(SQL, "ttr.query", "Definition missing");
	qry = s->def;

	chkProgram(cntxt->fdout,cntxt->nspace,qry);
	if( qry->errors)
		msg = createException(SQL,"ttr.query","Error in timetrails query");

	_DEBUG_TTR_ fprintf(stderr,"#ttr: register a new continuous query plan\n");
	s = newFunction(userRef, putName(name), FUNCTIONsymbol);
	if (s == NULL)
		msg = createException(SQL, "ttr.query", "Procedure code does not exist.");

	freeMalBlk(s->def);
	s->def = copyMalBlk(qry);
	p = getInstrPtr(s->def, 0);
	setModuleId(p,userRef);
	setFunctionId(p, putName(name));
	insertSymbol(cntxt->nspace, s);
	_DEBUG_TTR_ printFunction(cntxt->fdout, s->def, 0, LIST_MAL_ALL);
	/* optimize the code and register at scheduler */
	if (msg == MAL_SUCCEED) 
		addtoMalBlkHistory(mb);
	if (msg == MAL_SUCCEED) {
		_DEBUG_TTR_ fprintf(stderr,"#ttr: continuous query plan\n");
		msg = PNregisterInternal(cntxt, s->def, calls);
	}

	// register all the baskets mentioned in the plan
	for( i=1; i< s->def->stop;i++){
		p= getInstrPtr(s->def,i);
		if( getModuleId(p) == basketRef && getFunctionId(p)== registerRef){
			BSKTregister(cntxt,s->def,0,p);
		}
	}
	return msg;
}

str
TTRactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	return PNresume(cntxt,mb,stk,pci);
}

str
TTRdeactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	return PNpause(cntxt,mb,stk,pci);
}
