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
#include "mal_builder.h"
#include "opt_prelude.h"

#define MAXPN 200           /* it is the minimum, if we need more space GDKrealloc */
#define PNcontrolInfinit 1  /* infinit loop of PNController  */
#define PNcontrolEnd 2      /* when all factories are disable PNController exits */

#define _DEBUG_PETRINET_ 

static str statusname[6] = { "<unknown>", "init", "paused", "running", "stop", "error" };

/*static int controlRounds = PNcontrolInfinit;*/
static void
PNstartScheduler(void);

typedef struct {
	str modname;	/* the MAL query block */
	str fcnname;
	MalBlkPtr mb;       /* Query block */
	MalStkPtr stk;    	/* might be handy */

	int status;     /* query status waiting/running/ready */
	int enabled;	/* all baskets are available */
	int places[MAXBSKT], targets[MAXBSKT];

	MT_Id	tid;
	int delay;      /* maximum delay between calls */
	timestamp seen; /* last executed */

	int cycles;     /* number of invocations of the factory */
	int events;     /* number of events consumed */
	str error;      /* last error seen */
	lng time;       /* total time spent for all invocations */
} PNnode;

PNnode pnet[MAXPN];
int pnettop = 0;

int enabled[MAXPN];     /*array that contains the id's of all queries that are enable to fire*/

static int status = BSKTINIT;
static int cycleDelay = 1; /* be careful, it affects response/throughput timings */

str PNanalyseWrapper(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	Module scope;
	Symbol s = 0;
	str modnme = *getArgReference_str(stk, pci, 1);
	str fcnnme = *getArgReference_str(stk, pci, 2);

	(void) mb;
	scope = findModule(cntxt->nspace, putName(modnme, (int) strlen(modnme)));
	if (scope)
		s = findSymbolInModule(scope, putName(fcnnme, (int) strlen(fcnnme)));
	if (s == NULL)
		throw(MAL, "petrinet.analysis", "Could not find function");

	return PNanalysis(cntxt, s->def);
}


str PNregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	Module scope;
	Symbol s = 0;
	str modnme = *getArgReference_str(stk, pci, 1);
	str fcnnme = *getArgReference_str(stk, pci, 2);

	(void) mb;
	scope = findModule(cntxt->nspace, putName(modnme, (int) strlen(modnme)));
	if (scope)
		s = findSymbolInModule(scope, putName(fcnnme, (int) strlen(fcnnme)));

	if (s == NULL)
		throw(MAL, "petrinet.register", "Could not find function");

	return PNregisterInternal(cntxt,s->def);
}

static int
PNlocate(str modname, str fcnname)
{
	int i;
	for (i = 0; i < pnettop; i++)
		if (strcmp(pnet[i].modname, modname) == 0 && strcmp(pnet[i].fcnname, fcnname) == 0)
			break;
	return i;
}
/* A transition is only allowed when all inputs are privately used */
str
PNregisterInternal(Client cntxt, MalBlkPtr mb)
{
	int i, init= pnet == 0;
	InstrPtr sig;
	str msg = MAL_SUCCEED;

	if (pnettop == MAXPN) 
		GDKerror("petrinet.register:Too many transitions");

	sig= getInstrPtr(mb,0);
	i = PNlocate(getModuleId(sig), getFunctionId(sig));
	if (i != pnettop)
		throw(MAL, "petrinet.register", "Duplicate definition of transition");

	pnet[pnettop].modname = GDKstrdup(getModuleId(sig));
	pnet[pnettop].fcnname = GDKstrdup(getFunctionId(sig));

	pnet[pnettop].status = BSKTPAUSE;
	pnet[pnettop].cycles = 0;
	pnet[pnettop].seen = *timestamp_nil;
	/* all the rest is zero */

	pnettop++;
	msg = PNanalysis(cntxt, mb);
	/* start the scheduler if analysis does not show errors */
	if( msg == MAL_SUCCEED && init)
		PNstartScheduler();
	return msg;
}

static str
PNstatus( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, int newstatus){
	str modname= NULL;
	str fcnname= NULL;
	int i;

	(void) cntxt;
	(void) mb;
	MT_lock_set(&iotLock);
	if ( pci->argc == 3){
		modname= *getArgReference_str(stk,pci,1);
		fcnname= *getArgReference_str(stk,pci,2);
		i = PNlocate(modname,fcnname);
		if ( i == pnettop)
			throw(SQL,"iot.pause","Continuous query not found");
		pnet[i].status = newstatus;
#ifdef _DEBUG_PETRINET_
	mnstr_printf(PNout, "#scheduler status %s.%s %s\n", modname,fcnname, statusname[newstatus]);
#endif
		return MAL_SUCCEED;
	}
	for ( i = 0; i < pnettop; i++){
		pnet[i].status = newstatus;
#ifdef _DEBUG_PETRINET_
		mnstr_printf(PNout, "#scheduler status %s\n", statusname[newstatus]);
#endif
	}
	MT_lock_unset(&iotLock);
	return MAL_SUCCEED;
}
str
PNpause(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	return PNstatus(cntxt, mb, stk, pci, BSKTPAUSE);
}

str
PNresume(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	return PNstatus(cntxt, mb, stk, pci, BSKTRUNNING);
}

str
PNstop(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	return PNstatus(cntxt, mb, stk, pci, BSKTSTOP);
}

str
PNcycles(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
#ifdef _DEBUG_PETRINET_
		mnstr_printf(PNout, "#scheduler cycles set \n");
#endif
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
}

// remove a transition

str PNdump(void *ret)
{
	int i, k, idx;
	mnstr_printf(PNout, "#scheduler status %s\n", statusname[status]);
	for (i = 0; i < pnettop; i++) {
		mnstr_printf(PNout, "#[%d]\t%s.%s %s delay %d cycles %d events %d time " LLFMT " ms\n",
				i, pnet[i].modname, pnet[i].fcnname, statusname[pnet[i].status], pnet[i].delay, pnet[i].cycles, pnet[i].events, pnet[i].time / 1000);
		if (pnet[i].error)
			mnstr_printf(PNout, "#%s\n", pnet[i].error);
		for (k = 0; k < MAXBSKT && pnet[i].places[k]; k++){
			idx = pnet[i].places[k];
			mnstr_printf(PNout, "#<--\t%s basket[%d] %d %d\n",
					baskets[idx].table,
					idx,
					baskets[idx].count,
					baskets[idx].events);
		}
		for (k = 0; k <MAXBSKT &&  pnet[i].targets[k]; k++){
			idx = pnet[i].targets[k];
			mnstr_printf(PNout, "#-->\t%s basket[%d] %d %d\n",
					baskets[idx].table,
					idx,
					baskets[idx].count,
					baskets[idx].events);
		}
	}
	(void) ret;
	return MAL_SUCCEED;
}

/* check the routine for input/output relationships */
/* Make sure we do not re-use the same source more than once */
str
PNanalysis(Client cntxt, MalBlkPtr mb)
{
	int i, j, k;
	InstrPtr p;
	str tbl;

	(void) cntxt;
	/* first check for errors */
	for (i = 0; i < mb->stop; i++) {
		p = getInstrPtr(mb, i);
		if (getModuleId(p) == basketRef && getFunctionId(p) == registerRef) {
			tbl = getVarConstant(mb, getArg(p, p->argc - 1)).val.sval;
			for (j = 0; j < pnettop; j++)
				for (k = 0; k < MAXBSKT && pnet[j].places[k]; k++)
					if (strcmp(tbl, baskets[pnet[j].places[k]].table) == 0)
						throw(MAL, "iot.register", "Duplicate use of continuous query input");
		}
	}
	for (i = 0; i < mb->stop; i++) {
		p = getInstrPtr(mb, i);
		if (getModuleId(p) == basketRef && getFunctionId(p) == registerRef) {
			tbl = getVarConstant(mb, getArg(p, p->argc - 1)).val.sval;
		}
		if (getModuleId(p) == basketRef && getFunctionId(p) == putName("pass", 4)) {
			tbl = getVarConstant(mb, getArg(p, p->retc)).val.sval;
			mnstr_printf(cntxt->fdout, "#output basket %s \n", tbl);
		}
	}
	return MAL_SUCCEED;
}
/*
 * The PetriNet controller lives in an separate thread.
 * It cycles through all transition nodes, hunting for paused queries that can fire.
 * The current policy is a simple round-robin. Later we will
 * experiment with more advanced schemes, e.g., priority queues.
 *
 * During each step cycle we first enable the transformations.
 */
static void
PNexecute( void *n)
{
	PNnode *node= (PNnode *) n;
	node->status = BSKTPAUSE;
#ifdef _DEBUG_PETRINET_
		mnstr_printf(PNout, "#petrinet.executed %s.%s\n",node->modname, node->fcnname);
#endif
}
static void
PNcontroller(void *dummy)
{
	int idx = -1, i, j;
	Client cntxt;
	int k = -1;
	int m = 0;
	str msg = MAL_SUCCEED;
	lng t, analysis, now;

	cntxt = mal_clients; /* run as admin in SQL mode*/
	 if( strcmp(cntxt->scenario, "sql") )
		 SQLinitEnvironment(cntxt, NULL, NULL, NULL);

	status = BSKTRUNNING;

	while( status != BSKTSTOP){
		if (cycleDelay)
			MT_sleep_ms(cycleDelay);  /* delay to make it more tractable */
		while (status == BSKTPAUSE)
			MT_sleep_ms(cycleDelay);  /* delay to make it more tractable */

		/* collect latest statistics, note that we don't need a lock here,
		   because the count need not be accurate to the usec. It will simply
		   come back. We also only have to check the places that are marked
		   empty. */
		now = GDKusec();
		for (k = i = 0; status == BSKTRUNNING && i < pnettop; i++) 
		if ( pnet[i].status != BSKTPAUSE ){
			pnet[i].enabled = 1;
			for (j = 0; j < MAXBSKT &&  pnet[i].enabled && pnet[i].places[j]; j++) {
				idx = pnet[i].places[j];
				if (baskets[idx].count == 0  || baskets[idx].count < baskets[idx].threshold ) { /* nothing to take care of */
					pnet[i].enabled = 0;
					break;
				}
				if (pnet[i].enabled) {
					timestamp ts, tn;
					/* only look at large enough baskets */
					/* check heart beat delays */
					if (baskets[idx].beat) {
						(void) MTIMEunix_epoch(&ts);
						(void) MTIMEtimestamp_add(&tn, &baskets[idx].seen, &baskets[idx].beat);
						if (tn.days < ts.days || (tn.days == ts.days && tn.msecs < ts.msecs)) {
							pnet[i].enabled = 0;
							break;
						}
					}
#ifdef _debug_petrinet_
					mnstr_printf(cntxt->fdout, "#petrinet:%d tuples for %s, source %d\n", baskets[idx].count, pnet[i].fcnname, j);
#endif
				} 
			}
			if (pnet[i].enabled )
				/*save the ids of all continuous queries that can be executed */
				enabled[k++] = i;
		}
		analysis = GDKusec() - now;

		/* execute each enabled transformation */
		/* We don't need to access again all the factories and check again which are available to execute them
		 * we have already kept the enable ones in the enabled list (created in the previous loop)
		 * and now it is enough to access that list*/
		for (m = 0; m < k; m++) {
			i = enabled[m];
			if (pnet[i].enabled ) {
#ifdef _DEBUG_PETRINET_
				mnstr_printf(cntxt->fdout, "#Run transition %s \n", pnet[i].fcnname);
#endif

				(void) MTIMEcurrent_timestamp(&baskets[idx].seen);
				t = GDKusec();
				pnet[i].cycles++;
				// Fork MAL execution thread 
				if (MT_create_thread(&pnet[i].tid, PNexecute, (void*) &pnet[i], MT_THR_JOINABLE) < 0){
					msg= createException(MAL,"petrinet.controller","Can not fork the thread");
				}
				pnet[i].time += GDKusec() - t + analysis;   /* keep around in microseconds */
				if (msg != MAL_SUCCEED && !strstr(msg, "too early")) {
					char buf[BUFSIZ];
					if (pnet[i].error == NULL) {
						snprintf(buf, BUFSIZ - 1, "Query %s failed:%s", pnet[i].fcnname, msg);
						pnet[i].error = GDKstrdup(buf);
					} else
						GDKfree(msg);
					pnet[i].enabled = -1;
					/* abort current transaction  */
				} else {
					(void) MTIMEcurrent_timestamp(&pnet[i].seen);
					for (j = 0; j < MAXBSKT && pnet[i].places[j]; j++) {
						idx = pnet[i].places[j];
						(void) MTIMEcurrent_timestamp(&baskets[idx].seen);
					}
				}
			}
		}
	}
	MT_lock_set(&iotLock);
	status = BSKTINIT;
	MT_lock_unset(&iotLock);
	(void) dummy;
}

void
PNstartScheduler(void)
{
	MT_Id pid;
	int s;
#ifdef _DEBUG_PETRINET_
	PNdump(&s);
#else
	(void) s;
#endif

	if (status== BSKTINIT && MT_create_thread(&pid, PNcontroller, &s, MT_THR_JOINABLE) != 0){
		GDKerror( "petrinet creation failed");
	}
	(void) pid;
}

/* inspection  routines */
str
PNtable(bat *modnameId, bat *fcnnameId, bat *statusId, bat *seenId, bat *cyclesId, bat *eventsId, bat *timeId, bat * errorId)
{
	BAT *modname = NULL, *fcnname = NULL, *status = NULL, *seen = NULL, *cycles = NULL, *events = NULL, *time = NULL, *error = NULL;
	int i;

	modname = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (modname == 0)
		goto wrapup;
	BATseqbase(modname, 0);
	fcnname = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (fcnname == 0)
		goto wrapup;
	BATseqbase(fcnname, 0);
	status = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (status == 0)
		goto wrapup;
	BATseqbase(status, 0);
	seen = BATnew(TYPE_void, TYPE_timestamp, BATTINY, TRANSIENT);
	if (seen == 0)
		goto wrapup;
	BATseqbase(seen, 0);
	cycles = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (cycles == 0)
		goto wrapup;
	BATseqbase(cycles, 0);
	events = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (events == 0)
		goto wrapup;
	BATseqbase(events, 0);
	time = BATnew(TYPE_void, TYPE_lng, BATTINY, TRANSIENT);
	if (time == 0)
		goto wrapup;
	BATseqbase(time, 0);
	error = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (error == 0)
		goto wrapup;
	BATseqbase(error, 0);

	for (i = 0; i < pnettop; i++) {
		BUNappend(modname, pnet[i].modname, FALSE);
		BUNappend(fcnname, pnet[i].fcnname, FALSE);
		BUNappend(status, statusname[pnet[i].status], FALSE);
		BUNappend(seen, &pnet[i].seen, FALSE);
		BUNappend(cycles, &pnet[i].cycles, FALSE);
		BUNappend(events, &pnet[i].events, FALSE);
		BUNappend(time, &pnet[i].time, FALSE);
		BUNappend(error, (pnet[i].error ? pnet[i].error : ""), FALSE);
	}
	BBPkeepref(*modnameId = modname->batCacheid);
	BBPkeepref(*fcnnameId = fcnname->batCacheid);
	BBPkeepref(*statusId = status->batCacheid);
	BBPkeepref(*seenId = seen->batCacheid);
	BBPkeepref(*cyclesId = cycles->batCacheid);
	BBPkeepref(*eventsId = events->batCacheid);
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
	if (cycles)
		BBPunfix(cycles->batCacheid);
	if (events)
		BBPunfix(events->batCacheid);
	if (time)
		BBPunfix(time->batCacheid);
	if (error)
		BBPunfix(error->batCacheid);
	throw(MAL, "iot.queries", MAL_MALLOC_FAIL);
}
