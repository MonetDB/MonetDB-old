/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

/*
 *  Martin Kersten
 * Language Extensions
 * Iterators over scalar ranges are often needed, also at the MAL level.
 * The barrier and control primitives are sufficient to mimic them directly.
 *
 * The modules located in the kernel directory should not
 * rely on the MAL datastructures. That's why we have to deal with
 * some bat operations here and delegate the signature to the
 * proper module upon loading.
 *
 * Running a script is typically used to initialize a context.
 * Therefore we need access to the runtime context.
 * For the call variants we have
 * to determine an easy way to exchange the parameter/return values.
 */

#include "monetdb_config.h"
#include "language.h"

str
CMDraise(str *ret, str *msg)
{
	*ret = GDKstrdup(*msg);
	return GDKstrdup(*msg);
}

str
MALassertBit(void *ret, bit *val, str *msg){
	(void) ret;
	if( *val == 0 || *val == bit_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}
str
MALassertInt(void *ret, int *val, str *msg){
	(void) ret;
	if( *val == 0 || *val == int_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}
str
MALassertLng(void *ret, lng *val, str *msg){
	(void) ret;
	if( *val == 0 || *val == lng_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}
#ifdef HAVE_HGE
str
MALassertHge(void *ret, hge *val, str *msg){
	(void) ret;
	if( *val == 0 || *val == hge_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}
#endif
str
MALassertSht(void *ret, sht *val, str *msg){
	(void) ret;
	if( *val == 0 || *val == sht_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}
str
MALassertOid(void *ret, oid *val, str *msg){
	(void) ret;
	if( *val == oid_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}
str
MALassertStr(void *ret, str *val, str *msg){
	(void) ret;
	if( *val == str_nil)
		throw(MAL, "mal.assert", "%s", *msg);
	return MAL_SUCCEED;
}

str
MALassertTriple(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p){
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) p;
	throw(MAL, "mal.assert", PROGRAM_NYI);
}

/*
 * Printing
 * The print commands are implemented as single instruction rules,
 * because they need access to the calling context.
 * At a later stage we can look into the issues related to
 * parsing the format string as part of the initialization phase.
 * The old method in V4 essentially causes a lot of overhead
 * because you have to prepare for the worst (e.g. mismatch format
 * identifier and argument value)
 *
 * Input redirectionrs
 */
str
CMDcallString(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str *s;

	(void) mb;		/* fool compiler */
	s = getArgReference_str(stk, pci, 1);
	if (strlen(*s) == 0)
		return MAL_SUCCEED;
	return callString(cntxt, *s);
}

str
CMDcallFunction(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str mod = *getArgReference_str(stk,pci,1);
	str fcn = *getArgReference_str(stk,pci,2);
	char buf[BUFSIZ];

	(void) mb;		/* fool compiler */
	if (strlen(mod) == 0 || strlen(fcn) ==0)
		return MAL_SUCCEED;
	// lazy implementation of the call
	snprintf(buf,BUFSIZ,"%s.%s();",mod,fcn);
	return callString(cntxt, buf);
}

str
MALstartDataflow( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bit *ret = getArgReference_bit(stk,pci,0);
	int pc = getPC(mb,pci);

	if ( pc <0 || pc > pci->jump)
		throw(MAL,"language.dataflow","Illegal statement range");
	*ret = 0;	/* continue at end of block */
	return runMALdataflow(cntxt, mb, pc, pci->jump, stk);
}

/*
 * Garbage collection over variables can be postponed by grouping
 * all dependent ones in a single sink() instruction.
 */
str
MALgarbagesink( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

str
MALpass( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

str 
CMDregisterFunction(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str *mod = getArgReference_str(stk,pci,1);
	str *fcn = getArgReference_str(stk,pci,2);
	str *code = getArgReference_str(stk,pci,3);
	str *help = getArgReference_str(stk,pci,4);
	InstrPtr sig;
	str msg;
	Symbol sym;

	(void) mb;
	msg= compileString(cntxt,*code);
	mnstr_printf(cntxt->fdout,"#register FUNCTION %s.%s\n",
		getModuleId(cntxt->curprg->def->stmt[0]), getFunctionId(cntxt->curprg->def->stmt[0]));
	if( help)
		cntxt->curprg->def->help= GDKstrdup(*help);
	sym = newSymbol(putName(*fcn), 0);
	if( sym == NULL)
		throw(MAL,"language.register", MAL_MALLOC_FAIL);
	sym->def = copyMalBlk(cntxt->curprg->def);
	if( sym->def == NULL)
		throw(MAL,"language.register", MAL_MALLOC_FAIL);
	sig= getSignature(sym);
	sym->name= putName(*fcn);
	setModuleId(sig, putName(*mod));
	setFunctionId(sig, sym->name);
	insertSymbol(findModule(cntxt->usermodule, getModuleId(sig)), sym);
	return msg;
}

str
CMDsourceFile(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str s = *getArgReference_str(stk,pci,1);
	char *msg = NULL;
	size_t l;

	(void) mb;

	if (s == 0) 
		throw(MAL, "language.source", RUNTIME_FILE_NOT_FOUND "missing file name");

	if (*s != DIR_SEP) {
		char *buf = GDKmalloc(strlen(monet_cwd) + strlen(s) + 2);
		if ( buf == NULL)
			throw(MAL,"language.source", MAL_MALLOC_FAIL);

		strcpy(buf, monet_cwd);
		buf[ l =strlen(buf)] = DIR_SEP;
		buf[ l+1] = 0;
		strcat(buf, s);
		msg = evalFile(cntxt, buf, 0, 0);
		GDKfree(buf);
	} else 
		msg = evalFile(cntxt, s, 0, 0);
	return msg;
}
