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
#include "iot.h"
#include "opt_iot.h"
#include "sql_optimizer.h"
#include "sql_gencode.h"

MT_Lock iotLock MT_LOCK_INITIALIZER("iotLock");

// locate the SQL procedure in the catalog
static str
IOTprocedureStmt(Client cntxt, MalBlkPtr mb, str schema, str nme)
{
	mvc *m = NULL;
	str msg = getSQLContext(cntxt, mb, &m, NULL);
	sql_schema  *s;
	backend *be;
	node *o;
	sql_func *f;
	/*sql_trans *tr;*/

	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
		return msg;
	s = mvc_bind_schema(m, schema);
	if (s == NULL)
		throw(SQL, "iot.query", "Schema missing");
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
	throw(SQL, "iot.query", "Procedure missing");
}

/* locate the MAL representation of this operation and extract the flow */
/* If the operation is not available yet, it should be compiled from its
   definition retained in the SQL catalog */
str
IOTquery(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch= NULL;
	str nme= NULL;
	str def= NULL;
	Symbol s = NULL;
	MalBlkPtr qry;
	str msg = NULL;
	InstrPtr p;
	Module scope;
	char buf[BUFSIZ], name[IDLENGTH];
	static int iotquerycnt=0;


	_DEBUG_IOT_ fprintf(stderr,"#iot: register the continues query %s.%s()\n",sch,nme);

	/* check existing of the pre-compiled and activated function */
	// if( pci->argc == 3&&  PNisregistered(sch,nme) ) return MAL_SUCCEED;
		//throw(SQL, "iot.query", "already activated");

	if (pci->argc == 3) {
		sch = *getArgReference_str(stk, pci, 1);
		nme = *getArgReference_str(stk, pci, 2);
		/* check existing of the pre-compiled function */
		_DEBUG_IOT_ fprintf(stderr,"#iot: locate a SQL procedure %s.%s()\n",sch,nme);
		msg = IOTprocedureStmt(cntxt, mb, sch, nme);
		if (msg)
			return msg;
		s = findSymbolInModule(cntxt->nspace, putName(nme));
		if (s == NULL)
			throw(SQL, "iot.query", "Definition missing");
		qry = s->def;
	} else if (pci->argc == 2){
		// pre-create the new procedure
		sch = "user";
		snprintf(name, IDLENGTH,"cquery_%d",iotquerycnt++);
		def = *getArgReference_str(stk, pci, 1);
		// package it as a procedure in the current schema [todo]
		snprintf(buf,BUFSIZ,"create procedure %s.%s() begin %s; end",sch,name,def);
		_DEBUG_IOT_ fprintf(stderr,"#iot.compile: %s\n",buf);
		nme = name;
		msg = SQLstatementIntern(cntxt, &def, nme, 1, 0, 0);
		if (msg)
			return msg;
		qry = cntxt->curprg->def;
	}
	chkProgram(cntxt->fdout,cntxt->nspace,qry);
	if( qry->errors)
		msg = createException(SQL,"iot.query","Error in iot query");

	_DEBUG_IOT_ fprintf(stderr,"#iot: register a new continuous query plan\n");
	scope = findModule(cntxt->nspace, putName(sch));
	s = newFunction(putName(sch), putName(nme), FUNCTIONsymbol);
	if (s == NULL)
		msg = createException(SQL, "iot.query", "Procedure code does not exist.");

	freeMalBlk(s->def);
	s->def = copyMalBlk(qry);
	p = getInstrPtr(s->def, 0);
	setModuleId(p, putName(sch));
	setFunctionId(p, putName(nme));
	insertSymbol(scope, s);
	_DEBUG_IOT_ printFunction(cntxt->fdout, s->def, 0, LIST_MAL_ALL);
	/* optimize the code and register at scheduler */
	if (msg == MAL_SUCCEED) 
		addtoMalBlkHistory(mb);
	if (msg == MAL_SUCCEED) {
		_DEBUG_IOT_ fprintf(stderr,"#iot: continuous query plan\n");
		msg = PNregisterInternal(cntxt, s->def);
	}
	return msg;
}

str
IOTactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch, tbl;
	int idx = 0;

	if( pci->argc == pci->retc){
		(void) BSKTactivate(cntxt,mb,stk,pci);
	} else {
		sch = *getArgReference_str(stk, pci, 1);
		tbl = *getArgReference_str(stk, pci, 2);
		_DEBUG_IOT_ fprintf(stderr,"#iot: activate %s.%s\n",sch,tbl);
		/* check for registration */
		idx = BSKTlocate(sch, tbl);
		if( idx )
		(void) BSKTactivate(cntxt,mb,stk,pci);
	}
	return PNactivate(cntxt,mb,stk,pci);
}

str
IOTdeactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch, tbl;
	int idx = 0;

	if( pci->argc == pci->retc){
		(void) BSKTdeactivate(cntxt,mb,stk,pci);
	} else {
		sch = *getArgReference_str(stk, pci, 1);
		tbl = *getArgReference_str(stk, pci, 2);
		_DEBUG_IOT_ fprintf(stderr,"#iot: deactivate %s.%s\n",sch,tbl);
		/* check for registration */
		idx = BSKTlocate(sch, tbl);
		if( idx )
			(void) BSKTdeactivate(cntxt,mb,stk,pci);
	}
	return PNdeactivate(cntxt,mb,stk,pci);
}

str
IOTcycles(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	return PNcycles(cntxt,mb,stk,pci);
}
