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
 * Martin Kersten
 * Petri-net query scheduler
   The Iot scheduler is based on the long-standing and mature Petri-net technology. For completeness, we
   recap its salient points taken from Wikipedia. For more detailed information look at the science library.

   The Iot scheduler is a fair implementation of a Petri-net interpreter. It models all continuous queries as transitions,
   and the stream tables represent the places with all token events. Unlike the pure Petri-net model, all tokens in a place
   are taken out on each firing. They may result into placing multiple tokens into receiving baskets.

   The scheduling amongst the transistions is currently deterministic. Upon each round of the scheduler, it determines all
   transitions eligble to fire, i.e. have non-empty baskets, which are then actived one after the other.
   Future implementations may relax this rigid scheme using a parallel implementation of the scheduler, such that each 
   transition by itself can decide to fire. However, when resources are limited to handle all complex continuous queries, 
   it may pay of to invest into a domain specif scheduler.

   The current implementation is limited to a fixed number of transitions. The scheduler can be stopped and restarted
   at any time. Even selectively for specific baskets. This provides the handle to debug a system before being deployed.
   In general, event processing through multiple layers of continous queries is too fast to trace them one by one.
   Some general statistics about number of events handled per transition is maintained, as well as the processing time
   for each continous query step. This provides the information to re-design the event handling system.
 */

#include "monetdb_config.h"
#include "iot.h"
#include "petrinet.h"
#include "basket.h"
#include "mal_builder.h"
#include "opt_prelude.h"

#define MAXPN 200           /* it is the minimum, if we need more space GDKrealloc */
#define PNDELAY 20			/* forced delay between PN scheduler cycles */

static str statusname[6] = { "init", "running", "waiting", "paused","stopping"};

/* keep track of running tasks */
static int PNcycle;

static void
PNstartScheduler(void);

typedef struct {
	str modname;	/* the MAL query block */
	str fcnname;
	MalBlkPtr mb;       /* Query block */
	MalStkPtr stk;    	/* might be handy */

	int status;     /* query status waiting/running/paused */
	int enabled;	/* all baskets are available */
	int inputs[MAXBSKT], outputs[MAXBSKT];
	
	int limit;		/* limit the number of invocations before dying */
	lng heartbeat;		/* heart beat for procedures */

	MT_Id	tid;
	int delay;      /* maximum delay between calls */
	timestamp seen; /* last executed */

	int runs;     /* number of invocations of the continuous query */
	//int events;     /* number of events consumed by all (sum of tumblings) */
	str error;      /* last error seen */
	lng time;       /* total time spent for all invocations */
} PNnode;

PNnode pnet[MAXPN];
int pnettop = 0;

int enabled[MAXPN];     /*array that contains the id's of all queries that are enable to fire*/

static int pnstatus = PNINIT;
static int cycleDelay = 50; /* be careful, it affects response/throughput timings */

str PNperiod(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int period = *getArgReference_int(stk, pci, 1);
	
	(void) cntxt;
	(void) mb;

	if (period < 0)
		throw(MAL,"iot.period","The period should be >= 0\n");
	cycleDelay = period;
	return MAL_SUCCEED;
}

str
PNgetheartbeat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    lng *ret = getArgReference_lng(stk,pci,0);
    str mod = *getArgReference_str(stk,pci,1);
    str fcn = *getArgReference_str(stk,pci,2);
    int i;
	char buf[IDLENGTH];

	(void) cntxt;
	(void) mb;
	snprintf(buf,IDLENGTH,"%s_%s",mod,fcn);
	for(i = 0; i < pnettop; i++) 
		if(strcmp(pnet[i].modname,userRef) == 0 && strcmp(pnet[i].fcnname,buf) == 0) {
			*ret = pnet[i].heartbeat;
			return MAL_SUCCEED;
		}
	throw(MAL,"iot.getheartbeat","Stream table or query '%s.%s' not found",mod,fcn);
}

str 
PNheartbeat(Client cntxt,str mod, str fcn, lng ticks)
{
	Module scope;
	Symbol s=  NULL;
	int i;
	str msg;
	char buf[IDLENGTH];

	if (ticks < 0)
		throw(MAL,"iot.heartbeat","The heartbeat should be >= 0\n");
	snprintf(buf,IDLENGTH,"%s_%s",mod,fcn);
	for(i = 0; i < pnettop; i++) {
		if(strcmp(pnet[i].modname,userRef) == 0 && strcmp(pnet[i].fcnname,buf) == 0) {
			pnet[i].heartbeat = ticks;
			return MAL_SUCCEED;
		}
	}

	scope = findModule(cntxt->nspace, putName(mod));
	if (scope)
		s = findSymbolInModule(scope, putName(fcn));

	if (s == NULL)
		throw(MAL, "iot.heartbeat", "Could not find function %s.%s\n",mod,fcn);

	msg = PNregisterInternal(cntxt,s->def,0);
	if( msg){
		GDKfree(msg);
		throw(MAL,"iot.heartbeat","Cannot access stream, nor active query\n");
	}
	// try it again
	snprintf(buf,IDLENGTH,"%s_%s",mod,fcn);
	for(i = 0; i < pnettop; i++) {
		if(strcmp(pnet[i].modname,userRef) == 0 && strcmp(pnet[i].fcnname,buf) == 0) {
			pnet[i].heartbeat = ticks;
			return MAL_SUCCEED;
		}
	}
	throw(MAL, "iot.heartbeat", "Could not find function %s.%s\n",mod,fcn);
}

str PNregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	Module scope;
	Symbol s = 0;
	str modnme = *getArgReference_str(stk, pci, 1);
	str fcnnme = *getArgReference_str(stk, pci, 2);
	int calls = 1;

	(void) mb;
	scope = findModule(cntxt->nspace, putName(modnme));
	if (scope)
		s = findSymbolInModule(scope, putName(fcnnme));

	if (s == NULL)
		throw(MAL, "petrinet.register", "Could not find function\n");

	if( pci->argc > 3)
		calls = *getArgReference_int(stk,pci,2);
	return PNregisterInternal(cntxt,s->def,calls);
}

static int
PNlocate(str modname, str fcnname)
{
	int i;
	char name[IDLENGTH];
	snprintf(name,IDLENGTH,"%s_%s",modname,fcnname);
	for (i = 0; i < pnettop; i++)
		if (strcmp(pnet[i].modname, userRef) == 0 && strcmp(pnet[i].fcnname, name) == 0)
			return i;
	return i;
}

str
PNshow(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch = *getArgReference_str(stk,pci,1);
	str fcn = *getArgReference_str(stk,pci,2);
	int idx;
	int i;
	InstrPtr p;
	Symbol s;
	char name[IDLENGTH];

	(void) cntxt;
	(void) mb;

	snprintf(name,IDLENGTH,"%s_%s",sch,fcn);
	idx = PNlocate(sch, fcn);
	if( idx == pnettop)
		throw(SQL,"basket.commit","Continous query %s.%s not accessible\n",sch,fcn);
	/* release the basket lock */
	printFunction(cntxt->fdout, pnet[idx].mb, 0, LIST_MAL_NAME | LIST_MAL_VALUE  | LIST_MAL_MAPI);
	for( i= 1; i< pnet[idx].mb->stop; i++){
		p= getInstrPtr(pnet[idx].mb,i);
		if(getFunctionId(p) && strcmp(getFunctionId(p), name) ==0){
			s = findSymbol(cntxt->nspace,userRef, getFunctionId(p));
			if( s) {
				printFunction(cntxt->fdout, s->def, 0, LIST_MAL_NAME | LIST_MAL_VALUE  | LIST_MAL_MAPI);
				return MAL_SUCCEED;
			}
		}
	}
	throw(SQL,"basket.commit","Continous query %s.%s not accessible\n",sch,fcn);
}

/* A transition is only allowed when all inputs are privately used */
str
PNregisterInternal(Client cntxt, MalBlkPtr mb, int calls)
{
	int i, init = pnettop == 0;
	InstrPtr sig,q;
	str msg = MAL_SUCCEED;
	MalBlkPtr nmb;
	Symbol s;
	char buf[IDLENGTH];

#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#registerInternal status %d\n", init);
#endif
	if (pnettop == MAXPN) 
		GDKerror("petrinet.register:Too many transitions");

	sig = getInstrPtr(mb,0);
	i = PNlocate(userRef, getFunctionId(sig));
	if (i != pnettop){
		// restart the  query
		pnet[pnettop].status = PNWAIT;
		pnet[pnettop].limit = calls; 
		return MAL_SUCCEED;
	}

	memset((void*) (pnet+pnettop), 0, sizeof(PNnode));
	pnet[pnettop].modname = GDKstrdup(userRef);
	pnet[pnettop].fcnname = GDKstrdup(getFunctionId(sig));
	snprintf(buf,IDLENGTH,"petri_%d",pnettop);
	s = newFunction(userRef, putName(buf), FUNCTIONsymbol);
	nmb = s->def;
	setArgType(nmb, nmb->stmt[0],0, TYPE_void);
    (void) newStmt(nmb, sqlRef, transactionRef);
	(void) newStmt(nmb, userRef,getFunctionId(sig));
    q = newStmt(nmb, sqlRef, commitRef);
	setArgType(nmb,q, 0, TYPE_void);
	pushEndInstruction(nmb);
	chkProgram(cntxt->fdout, cntxt->nspace, nmb);
#ifdef DEBUG_PETRINET
	printFunction(cntxt->fdout, nmb, 0, LIST_MAL_SCHEDULER);
#endif

	pnet[pnettop].mb = nmb;
	pnet[pnettop].stk = prepareMALstack(nmb, nmb->vsize);

	pnet[pnettop].status = PNWAIT;
	pnet[pnettop].limit = calls; 
	pnet[pnettop].seen = *timestamp_nil;
	/* all the rest is zero */

	msg = PNanalysis(cntxt, mb, pnettop);
	/* start the scheduler if analysis does not show errors */
	if(msg == MAL_SUCCEED) {
		if(init) {
			PNstartScheduler();
		}
		pnettop++;
	}
	return msg;
}

static str
PNstatus( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, int newstatus){
	str modname = NULL;
	str fcnname = NULL;
	int i;

	(void) cntxt;
	(void) mb;
	MT_lock_set(&iotLock);
	if ( pci->argc == 3){
		modname= *getArgReference_str(stk,pci,1);
		fcnname= *getArgReference_str(stk,pci,2);
		i = PNlocate(modname,fcnname);
		if ( i == pnettop){
			MT_lock_unset(&iotLock);
			throw(SQL,"iot.pause","Continuous query %s.%s not found\n", modname, fcnname);
		}
		pnet[i].status = newstatus;
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#scheduler status %s.%s %s\n", modname, fcnname, statusname[newstatus]);
#endif
		MT_lock_unset(&iotLock);
		return MAL_SUCCEED;
	}
	for ( i = 0; i < pnettop; i++){
		pnet[i].status = newstatus;
#ifdef DEBUG_PETRINET
		mnstr_printf(GDKout, "#scheduler status %s.%s: %s\n", pnet[i].modname, pnet[i].fcnname, statusname[newstatus]);
#endif
	}
	MT_lock_unset(&iotLock);
	return MAL_SUCCEED;
}

str
PNresume(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#resume scheduler\n");
#endif
	return PNstatus(cntxt, mb, stk, pci, PNWAIT);
}

str
PNwait(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
#ifdef DEBUG_PETRINET
	int old = PNcycle;
#endif
	int cnt=0, i;
	int delay= *getArgReference_int(stk,pci,1);

	(void) cntxt;
	(void) mb;
#ifdef DEBUG_PETRINET
	mnstr_printf(cntxt->fdout, "#scheduler wait %d ms\n",delay);
#endif
	delay = delay < PNDELAY? PNDELAY:delay;
	MT_sleep_ms(delay);
	for(cnt=0,  i = 0; i < pnettop; i++)
		cnt += pnet[i].status != PNWAIT;
#ifdef DEBUG_PETRINET
	mnstr_printf(cntxt->fdout, "#pnstop %d waiting for  %d queries \n", pnettop, cnt);
	mnstr_printf(cntxt->fdout, "#wait finished after %d queries ran\n",PNcycle -old );
#endif
	return MAL_SUCCEED;
}

str
PNpause(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#pause scheduler or individual queries\n");
#endif
	return PNstatus(cntxt, mb, stk, pci, PNPAUSED);
}


/* safely stop a single CQ at the end of a round */
static str
PNstopInternal(int idx){
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#scheduler stops %d\n",idx);
#endif

	MT_lock_set(&iotLock);
	while(pnet[idx].status != PNWAIT && pnet[idx].status != PNPAUSED){
		MT_sleep_ms(1000);
	}
	pnet[idx].status = PNPAUSED;
	MT_lock_unset(&iotLock);
	return MAL_SUCCEED;
}

str
PNstop(void){
	int i, cnt;
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#scheduler being stopped\n");
#endif

	pnstatus = PNSTOP; // avoid starting new continuous queries
	// warn all queries to stop
	for(i = 0; i < pnettop; i++)
		PNstopInternal(i);

	// wait for any leftover running queries
	do{
		for(cnt=0,  i = 0; i < pnettop; i++)
			cnt += (pnet[i].status != PNWAIT && pnet[i].status != PNPAUSED);
#ifdef DEBUG_PETRINET
		mnstr_printf(GDKout, "#pnstop waiting for  %d queries \n", cnt);
#endif
		if(cnt)
			MT_sleep_ms(1000);
	} while(pnstatus != PNINIT && cnt > 0);

	return MAL_SUCCEED;
}

/*Remove a specific continuous query from the scheduler */
static void
PNderegisterInternal(int i){
	MT_lock_set(&iotLock);
	GDKfree(pnet[i].modname);
	GDKfree(pnet[i].fcnname);
	memset((void*) (pnet+i), 0, sizeof(PNnode));
	for( ; i<pnettop-1; i++)
		pnet[i] = pnet[i+1];
	pnettop--;
	MT_lock_unset(&iotLock);
}

str
PNderegister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	str modname= NULL;
	str fcnname= NULL;
	int i;

	(void) cntxt;
	(void) mb;
	PNdump(&i);
	if ( pci->argc == 3){
		modname= *getArgReference_str(stk,pci,1);
		fcnname= *getArgReference_str(stk,pci,2);
		i = PNlocate(modname,fcnname);
		if ( i == pnettop){
			MT_lock_unset(&iotLock);
			throw(SQL,"iot.pause","Continuous query %s.%s not found\n", modname, fcnname);
		}
		PNderegisterInternal(i);
#ifdef DEBUG_PETRINET
		mnstr_printf(GDKout, "#scheduler deregistered %s.%s\n", modname, fcnname);
#endif
		return MAL_SUCCEED;
	}
	PNstop();
	for ( i = pnettop-1; i >= 0 ; i--)
		PNderegisterInternal(i);
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#scheduler deregistered all\n");
#endif
	return MAL_SUCCEED;
}

str
PNcycles(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	str modname= *getArgReference_str(stk,pci,1);
	str fcnname= *getArgReference_str(stk,pci,2);
	int limit= *getArgReference_int(stk,pci,3);
	int i = PNlocate(modname,fcnname);

#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#scheduler set cycle limit %s.%s %d\n",modname,fcnname,limit);
#endif
	(void) cntxt;
	(void) mb;
	if( i != pnettop)
		pnet[i].limit = limit;
	else 
		throw(SQL,"iot.limit","Continuous query not found\n");
	return MAL_SUCCEED;
}

// remove a transition

str PNdump(void *ret)
{
	int i, k, idx;
	mnstr_printf(GDKout, "#scheduler status %s\n", statusname[pnstatus]);
	for (i = 0; i < pnettop; i++) {
		mnstr_printf(GDKout, "#[%d]\t%s.%s %s delay %d runs %d time " LLFMT " ms\n",
				i, pnet[i].modname, pnet[i].fcnname, statusname[pnet[i].status], pnet[i].delay, pnet[i].runs, pnet[i].time / 1000);
		if (pnet[i].error)
			mnstr_printf(GDKout, "#%s\n", pnet[i].error);
		for (k = 0; k < MAXBSKT && pnet[i].inputs[k]; k++){
			idx = pnet[i].inputs[k];
			mnstr_printf(GDKout, "#<--\t%s basket "BUNFMT" %d\n",
					baskets[idx].table_name,
					baskets[idx].count,
					baskets[idx].status);
		}
		for (k = 0; k <MAXBSKT &&  pnet[i].outputs[k]; k++){
			idx = pnet[i].outputs[k];
			mnstr_printf(GDKout, "#-->\t%s basket "BUNFMT" %d\n",
					baskets[idx].table_name,
					baskets[idx].count,
					baskets[idx].status);
		}
	}
	(void) ret;
	return MAL_SUCCEED;
}

/* Collect all input/output basket roles */
/* Make sure we do not re-use the same source more than once */
/* Avoid any concurreny conflict */
str
PNanalysis(Client cntxt, MalBlkPtr mb, int pn)
{
	int i, j, idx, k=0,role;
	InstrPtr p;
	str msg= MAL_SUCCEED, sch,tbl;
	(void) pn;

	for (i = 0; msg== MAL_SUCCEED && i < mb->stop; i++) {
		p = getInstrPtr(mb, i);
		if (getModuleId(p) == basketRef && getFunctionId(p) == registerRef){
			sch = getVarConstant(mb, getArg(p,2)).val.sval;
			tbl = getVarConstant(mb, getArg(p,3)).val.sval;
			role = getVarConstant(mb, getArg(p,4)).val.ival;
			msg =BSKTregister(cntxt,mb,0,p);
			idx =  BSKTlocate(sch, tbl);
			// make sure we have only one reference
			if( role == 0 ){
				for(j=0; j< k; j++)
					if( pnet[pn].inputs[j] == idx)
						break;
				if ( j == k)
					pnet[pn].inputs[k++]= idx;
			} else {
				for(j=0; j< k; j++)
					if( pnet[pn].outputs[j] == idx)
						break;
				if ( j == k)
					pnet[pn].outputs[k++]= idx;
			}
		}
		if (getModuleId(p) == basketRef && getFunctionId(p) == appendRef){
			sch = getVarConstant(mb, getArg(p,2)).val.sval;
			tbl = getVarConstant(mb, getArg(p,3)).val.sval;
			idx =  BSKTlocate(sch, tbl);
			for(j=0; j< k; j++)
				if( pnet[pn].outputs[j] == idx)
					break;
			if ( j == k)
				pnet[pn].outputs[k++]= idx;
		}
	}
	return msg;
}

/*
 * The PetriNet controller lives in an separate thread.
 * It cycles through all transition nodes, hunting for paused queries that can fire.
 * The current policy is a simple round-robin. Later we will
 * experiment with more advanced schemes, e.g., priority queues.
 *
 * During each step cycle we first enable the transformations.
 *
 * Locking the streams is necessary to avoid concurrent changes.
 * Using a fixed order over the basket table, ensure no deadlock.
 */
static void
PNexecute( Client cntxt, int idx)
{
	PNnode *node= pnet+ idx;
	int j;
	lng t = GDKusec();
	timestamp ts;

	if( pnstatus != PNRUNNING)
		return;
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#petrinet.execute %s.%s\n",node->modname, node->fcnname);
#endif
	// first grab exclusive access to all streams.
	/*
	for (j = 0; j < MAXBSKT &&  node->inputs[j]; j++) 
		MT_lock_set(&baskets[node->inputs[j]].lock);
	for (j = 0; j < MAXBSKT &&  node->outputs[j]; j++) 
		MT_lock_set(&baskets[node->outputs[j]].lock);
*/

#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#petrinet.execute %s.%s all locked\n",node->modname, node->fcnname);
	printFunction(cntxt->fdout, node->mb, 0, LIST_MAL_NAME | LIST_MAL_VALUE  | LIST_MAL_MAPI);
#endif

	(void)runMALsequence(cntxt, node->mb, 1, 0, node->stk, 0, 0);

#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#petrinet.execute %s.%s transition done:\n", node->modname, node->fcnname);
#endif

	// remember the time last accessed
	(void) MTIMEcurrent_timestamp(&ts);
	for (j = 0; j < MAXBSKT &&  (idx = node->inputs[j]); j++) 
	if (baskets[idx].heartbeat ) 
		baskets[idx].seen= ts;
	for (j = 0; j < MAXBSKT &&  (idx = node->outputs[j]); j++) 
	if (baskets[idx].heartbeat ) 
		baskets[idx].seen= ts;
	node->seen = ts;

	 // empty the baskets according to their policy
 /*
	for (j = 0; j < MAXBSKT &&  node->inputs[j]; j++) 
		MT_lock_unset(&baskets[node->inputs[j]].lock);
	for (j = 0; j < MAXBSKT &&  node->outputs[j]; j++) 
		MT_lock_unset(&baskets[node->outputs[j]].lock);
*/

	pnet[node->inputs[0]].time += GDKusec() - t;   /* keep around in microseconds */
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#petrinet.execute %s.%s all unlocked\n",node->modname, node->fcnname);
#endif
	MT_lock_set(&iotLock);
	if( node->status != PNPAUSED)
		node->status = PNWAIT;
	MT_lock_unset(&iotLock);
}

static void
PNscheduler(void *dummy)
{
	int idx = -1, i, j;
	int k = -1;
	int m = 0, pntasks;
	Client cntxt;
	str msg = MAL_SUCCEED;
	lng t, analysis, now;
	char claimed[MAXBSKT];
	timestamp ts, tn;
	int force=0;

	stream *fin, *fout;
	fin =  open_rastream("/tmp/fin_petri_sched");
	fout =  open_wastream("/tmp/fout_petri_sched");

#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#petrinet.controller started\n");
#endif
	cntxt = MCinitClient(0,bstream_create(fin,0),fout);
	if( cntxt){
		if( SQLinitClient(cntxt) != MAL_SUCCEED){
			GDKerror("Could not initialize PNscheduler");
			mnstr_printf(GDKout, "#petrinet.controller could not initalize\n");
		}
	}else{
		GDKerror("Could not initialize PNscheduler");
		return;
	}
		
	MT_lock_set(&iotLock);
	pnstatus = PNRUNNING; // global state 
	MT_lock_unset(&iotLock);

	while( pnstatus != PNSTOP ){
		/* Determine which continuous query are eligble to run
  		   Collect latest statistics, note that we don't need a lock here,
		   because the count need not be accurate to the usec. It will simply
		   come back. We also only have to check the places that are marked
		   non empty. You can only trigger on empty baskets using a heartbeat */
		memset((void*) claimed, 0, MAXBSKT);
		now = GDKusec();
		pntasks=0;
		MT_lock_set(&iotLock); // analysis should be done with exclusive access
		for (k = i = 0; i < pnettop; i++) 
		if ( pnet[i].status == PNWAIT ){
			pnet[i].enabled = 1;

			if( pnet[i].limit == 0){
				 pnet[i].enabled = 0;// reached end of sequence
				 pnet[i].status = PNPAUSED; 
			}

			/* queries coming with a heartbeat overrule those in baskets */
			if( pnet[i].heartbeat > 0){
				(void) MTIMEcurrent_timestamp(&ts);
				(void) MTIMEtimestamp_add(&tn, &pnet[idx].seen, &pnet[idx].heartbeat);
				if ( tn.days < ts.days || (tn.days == ts.days && tn.msecs < ts.msecs)) {
#ifdef DEBUG_PETRINET_SCHEDULER
					mnstr_printf(GDKout,"# now %d.%d fire %d.%d,disable\n", ts.days,ts.msecs, tn.days,tn.msecs);
#endif
					force =1;
					break;
				} else {
#ifdef DEBUG_PETRINET_SCHEDULER
				mnstr_printf(GDKout,"# now %d.%d fire %d.%d enable[%d]\n", ts.days,ts.msecs, tn.days,tn.msecs,i);
#endif
				}
			}

			/* consider baskets that are properly filled */
			/* check if all input baskets are available and non-empty, and not controlled by a heartbeat */
			for (j = 0; force == 0 && j < MAXBSKT &&  pnet[i].enabled && pnet[i].inputs[j]; j++) {
				idx = pnet[i].inputs[j];
				/* consider the heart beat trigger, which overrules the filling test */
				if (baskets[idx].heartbeat ) {
					(void) MTIMEcurrent_timestamp(&ts);
					(void) MTIMEtimestamp_add(&tn, &baskets[idx].seen, &baskets[idx].heartbeat);
					if ( !(tn.days < ts.days || (tn.days == ts.days && tn.msecs < ts.msecs)) ){
#ifdef DEBUG_PETRINET_SCHEDULER
						mnstr_printf(GDKout,"#A now %d.%d fire %d.%d,disable\n", ts.days,ts.msecs, tn.days,tn.msecs);
#endif
						pnet[i].enabled = 0;
						break;
					} else {
#ifdef DEBUG_PETRINET_SCHEDULER
					mnstr_printf(GDKout,"#A now %d.%d fire %d.%d enable[%d]\n", ts.days,ts.msecs, tn.days,tn.msecs,i);
#endif
					}
				} else
				/* consider execution only if baskets are properly filled */
				if ( baskets[idx].winsize > 0 && (BUN) baskets[idx].winsize > baskets[idx].count){
					pnet[i].enabled = 0;
					break;
				} else
				if (baskets[idx].winsize != -1 && baskets[idx].count == 0 ){
					pnet[i].enabled = 0;
					break;
				}
			}
			/* check if all output baskets are not controlled by a heartbeat */
			for (j = 0; force == 0 && j < MAXBSKT &&  pnet[i].enabled && pnet[i].outputs[j]; j++) {
				idx = pnet[i].outputs[j];
				if (baskets[idx].heartbeat ) {
					(void) MTIMEcurrent_timestamp(&ts);
					(void) MTIMEtimestamp_add(&tn, &baskets[idx].seen, &baskets[idx].heartbeat);
					if ( !(tn.days < ts.days || (tn.days == ts.days && tn.msecs < ts.msecs)) ){
#ifdef DEBUG_PETRINET_SCHEDULER
						mnstr_printf(GDKout,"#B now %d.%d fire %d.%d,disable\n", ts.days,ts.msecs, tn.days,tn.msecs);
#endif
						pnet[i].enabled = 0;
						break;
					}
					else {
#ifdef DEBUG_PETRINET_SCHEDULER
					mnstr_printf(GDKout,"#B now %d.%d fire %d.%d enable[%d]\n", ts.days,ts.msecs, tn.days,tn.msecs,i);
#endif
					}
				} 
			}

			/* avoid concurrency conflicts */
			if (pnet[i].enabled) {
				/* a basket can be used in at most one continuous query at a time */
				for (j = 0; j < MAXBSKT &&  pnet[i].enabled && pnet[i].inputs[j]; j++) 
					if( claimed[pnet[i].inputs[j]]){
#ifdef DEBUG_PETRINET_SCHEDULER
						mnstr_printf(GDKout, "#petrinet: %s.%s enabled twice,disgarded \n", pnet[i].modname, pnet[i].fcnname);
#endif
						pnet[i].enabled = 0;
						break;
					} 
				for (j = 0; j < MAXBSKT &&  pnet[i].enabled && pnet[i].outputs[j]; j++) 
					if( claimed[pnet[i].outputs[j]]){
#ifdef DEBUG_PETRINET_SCHEDULER
						mnstr_printf(GDKout, "#petrinet: %s.%s enabled twice,disgarded \n", pnet[i].modname, pnet[i].fcnname);
#endif
						pnet[i].enabled = 0;
						break;
					} 

				/* rule out all others */
				if( pnet[i].enabled){
					for (j = 0; j < MAXBSKT &&  pnet[i].enabled && pnet[i].inputs[j]; j++) 
						claimed[pnet[i].inputs[j]]= 1;
					for (j = 0; j < MAXBSKT &&  pnet[i].enabled && pnet[i].outputs[j]; j++) 
						claimed[pnet[i].outputs[j]]= 1;
				}

				/*save the ids of all continuous queries that can be executed */
				enabled[k++] = i;
#ifdef DEBUG_PETRINET_SCHEDULER
				mnstr_printf(GDKout, "#petrinet: %s.%s enabled \n", pnet[i].modname, pnet[i].fcnname);
#endif
			} 
			pntasks += pnet[i].enabled;
		}
		MT_lock_unset(&iotLock); 
		analysis = GDKusec() - now;
#ifdef DEBUG_PETRINET_SCHEDULER
		if(k) mnstr_printf(GDKout, "#Transitions enabled: %d \n", k);
#endif
		PNcycle += k;
		if( pnstatus == PNSTOP)
			continue;

		/* Execute each enabled transformation */
		/* Tricky part is here a single stream used by multiple transitions */
		for (m = 0; m < k; m++) {
			i = enabled[m];
			if (pnet[i].enabled ) {
#ifdef DEBUG_PETRINET
				mnstr_printf(GDKout, "#Run transition %s \n", pnet[i].fcnname);
#endif

				t = GDKusec();
				// Fork MAL execution thread 
				PNexecute(cntxt, i);
/*
				if (MT_create_thread(&pnet[i].tid, PNexecute, (void*) (pnet+i), MT_THR_JOINABLE) < 0){
					msg= createException(MAL,"petrinet.controller","Can not fork the thread");
				} else
*/
					pnet[i].runs++;
				pnet[i].time += GDKusec() - t + analysis;   /* keep around in microseconds */
				if (msg != MAL_SUCCEED ){
					char buf[BUFSIZ];
					if (pnet[i].error == NULL) {
						snprintf(buf, BUFSIZ - 1, "Query %s failed:%s", pnet[i].fcnname, msg);
						pnet[i].error = GDKstrdup(buf);
					} else
						GDKfree(msg);
					/* abort current transaction  */
				} else {
					/* mark the time the query is started */
					(void) MTIMEcurrent_timestamp(&pnet[i].seen);
					for (j = 0; j < MAXBSKT && pnet[i].inputs[j]; j++) {
						idx = pnet[i].inputs[j];
						(void) MTIMEcurrent_timestamp(&baskets[idx].seen);
					}
				}
			}
		}
		/* after one sweep all threads should be released */
/*
		for (m = 0; m < k; m++)
		if(pnet[enabled[m]].tid){
#ifdef DEBUG_PETRINET
			mnstr_printf(GDKout, "#Terminate query thread %s limit %d \n", pnet[enabled[m]].fcnname, pnet[enabled[m]].limit);
#endif
			MT_join_thread(pnet[enabled[m]].tid);
			if( pnet[enabled[m]].limit > 0) pnet[enabled[m]].limit--;
			if(  pnet[enabled[m]].limit  ==0)
				 pnet[enabled[m]].status = PNPAUSED; 
			//if( pnet[enabled[m]].limit == 0)
				//PNderegisterInternal(enabled[m]);
		}

#ifdef DEBUG_PETRINET
		if (pnstatus == PNRUNNING && cycleDelay) MT_sleep_ms(cycleDelay);  
#endif
		MT_sleep_ms(PNDELAY);  
*/
		while (pnstatus == PNPAUSED)	{ /* scheduler is paused */
			MT_sleep_ms(cycleDelay);  
#ifdef DEBUG_PETRINET
			mnstr_printf(GDKout, "#petrinet.controller paused\n");
#endif
		}
	}
#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#petrinet.scheduler stopped\n");
#endif
	//MCcloseClient(cntxt);
	pnstatus = PNINIT;
	(void) dummy;
}

void
PNstartScheduler(void)
{
	MT_Id pid;
	int s;

#ifdef DEBUG_PETRINET
	mnstr_printf(GDKout, "#Start PNscheduler\n");
#endif
	if (pnstatus== PNINIT && MT_create_thread(&pid, PNscheduler, &s, MT_THR_JOINABLE) != 0){
#ifdef DEBUG_PETRINET
		mnstr_printf(GDKout, "#Start PNscheduler failed\n");
#endif
		GDKerror( "petrinet creation failed");
	}
	(void) pid;
}

/* inspection  routines */
str
PNtable(bat *modnameId, bat *fcnnameId, bat *statusId, bat *seenId, bat *runsId, bat *timeId, bat * errorId)
{
	BAT *modname = NULL, *fcnname = NULL, *status = NULL, *seen = NULL, *runs = NULL, *time = NULL, *error = NULL;
	int i;

	modname = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (modname == 0)
		goto wrapup;
	fcnname = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (fcnname == 0)
		goto wrapup;
	status = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (status == 0)
		goto wrapup;
	seen = COLnew(0, TYPE_timestamp, BATTINY, TRANSIENT);
	if (seen == 0)
		goto wrapup;
	runs = COLnew(0, TYPE_int, BATTINY, TRANSIENT);
	if (runs == 0)
		goto wrapup;
	time = COLnew(0, TYPE_lng, BATTINY, TRANSIENT);
	if (time == 0)
		goto wrapup;
	error = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (error == 0)
		goto wrapup;

	for (i = 0; i < pnettop; i++) {
		lng avg= pnet[i].runs? pnet[i].time / pnet[i].runs:0;
		BUNappend(modname, pnet[i].modname, FALSE);
		BUNappend(fcnname, pnet[i].fcnname, FALSE);
		BUNappend(status, statusname[pnet[i].status], FALSE);
		BUNappend(seen, &pnet[i].seen, FALSE);
		BUNappend(runs, &pnet[i].runs, FALSE);
		BUNappend(time, &avg, FALSE);
		BUNappend(error, (pnet[i].error ? pnet[i].error : ""), FALSE);
	}
	BBPkeepref(*modnameId = modname->batCacheid);
	BBPkeepref(*fcnnameId = fcnname->batCacheid);
	BBPkeepref(*statusId = status->batCacheid);
	BBPkeepref(*seenId = seen->batCacheid);
	BBPkeepref(*runsId = runs->batCacheid);
	BBPkeepref(*timeId = time->batCacheid);
	BBPkeepref(*errorId = error->batCacheid);
	return MAL_SUCCEED;
wrapup:
	if (modname)
		BBPunfix(modname->batCacheid);
	if (fcnname)
		BBPunfix(fcnname->batCacheid);
	if (status)
		BBPunfix(status->batCacheid);
	if (seen)
		BBPunfix(seen->batCacheid);
	if (runs)
		BBPunfix(runs->batCacheid);
	if (time)
		BBPunfix(time->batCacheid);
	if (error)
		BBPunfix(error->batCacheid);
	throw(MAL, "iot.queries", MAL_MALLOC_FAIL);
}

str PNinputplaces(bat *schemaId, bat *tableId, bat *modnameId, bat *fcnnameId)
{
	BAT *schema, *table = NULL, *modname = NULL, *fcnname = NULL;
	int i,j;

	schema = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (schema == 0)
		goto wrapup;

	table = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (table == 0)
		goto wrapup;

	modname = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (modname == 0)
		goto wrapup;

	fcnname = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (fcnname == 0)
		goto wrapup;

	for (i = 0; i < pnettop; i++) {
#ifdef DEBUG_PETRINET
		mnstr_printf(GDKout, "#collect input places %s.%s\n", pnet[i].modname, pnet[i].fcnname);
#endif
		for( j =0; j < MAXBSKT && pnet[i].inputs[j]; j++){
			BUNappend(schema, baskets[pnet[i].inputs[j]].schema_name, FALSE);
			BUNappend(table, baskets[pnet[i].inputs[j]].table_name, FALSE);
			BUNappend(modname, pnet[i].modname, FALSE);
			BUNappend(fcnname, pnet[i].fcnname, FALSE);
		}
	}
	BBPkeepref(*schemaId = schema->batCacheid);
	BBPkeepref(*tableId = table->batCacheid);
	BBPkeepref(*modnameId = modname->batCacheid);
	BBPkeepref(*fcnnameId = fcnname->batCacheid);
	return MAL_SUCCEED;
wrapup:
	if (schema)
		BBPunfix(schema->batCacheid);
	if (table)
		BBPunfix(table->batCacheid);
	if (modname)
		BBPunfix(modname->batCacheid);
	if (fcnname)
		BBPunfix(fcnname->batCacheid);
	throw(MAL, "iot.inputs", MAL_MALLOC_FAIL);
}

str PNoutputplaces(bat *schemaId, bat *tableId, bat *modnameId, bat *fcnnameId)
{
	BAT *schema, *table = NULL, *modname = NULL, *fcnname = NULL;
	int i,j;

	schema = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (schema == 0)
		goto wrapup;

	table = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (table == 0)
		goto wrapup;

	modname = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (modname == 0)
		goto wrapup;

	fcnname = COLnew(0, TYPE_str, BATTINY, TRANSIENT);
	if (fcnname == 0)
		goto wrapup;

	for (i = 0; i < pnettop; i++) 
	for( j =0; j < MAXBSKT && pnet[i].outputs[j]; j++){
		BUNappend(schema, baskets[pnet[i].outputs[j]].schema_name, FALSE);
		BUNappend(table, baskets[pnet[i].outputs[j]].table_name, FALSE);
		BUNappend(modname, pnet[i].modname, FALSE);
		BUNappend(fcnname, pnet[i].fcnname, FALSE);
	}
	BBPkeepref(*schemaId = schema->batCacheid);
	BBPkeepref(*tableId = table->batCacheid);
	BBPkeepref(*modnameId = modname->batCacheid);
	BBPkeepref(*fcnnameId = fcnname->batCacheid);
	return MAL_SUCCEED;
wrapup:
	if (schema)
		BBPunfix(schema->batCacheid);
	if (table)
		BBPunfix(table->batCacheid);
	if (modname)
		BBPunfix(modname->batCacheid);
	if (fcnname)
		BBPunfix(fcnname->batCacheid);
	throw(MAL, "iot.outputs", MAL_MALLOC_FAIL);
}
