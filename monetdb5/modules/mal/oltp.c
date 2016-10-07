/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/*
 * (c) Martin Kersten
 * This module provides an advisary lock manager for SQL transactions
 * that prefer waiting over transaction failures due to OCC
 * The table may only grow with lockable items
 * It could be extended with a semaphore for queue management
 */
#include "monetdb_config.h"
#include "oltp.h"

#define MAXOLTPLOCK 256
#define LOCKTIMEOUT 20 * 1000
#define DELAY 20

typedef struct{
	Client cntxt;
	char lckname[2 * IDLENGTH];
} OLTPlockRecord;

static OLTPlockRecord locks[MAXOLTPLOCK];
static int top;

static void
OLTPdump_(Client cntxt, str msg)
{
	int i;

	mnstr_printf(cntxt->fdout,"%s",msg);
	for(i=0; i< top; i++)
		mnstr_printf(cntxt->fdout,"#[%i] %3d %s\n",i, (locks[i].cntxt ? locks[i].cntxt->idx: -1), locks[i].lckname);
}

str
OLTPreset(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i;
	// release all locks held in reverse order
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	
	MT_lock_set(&mal_oltpLock);
	for( i=0; i<top; i--){
		locks[i].lckname[0] = 0;
		locks[i].cntxt = 0;
	}
	MT_lock_unset(&mal_oltpLock);
	return MAL_SUCCEED;
}

str
OLTPinit(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	// nothing to be done right now
	return OLTPreset(cntxt,mb,stk,pci);
}

// a naive locking scheme without queueing
// only return when you can access all locks

str
OLTPlock(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i,j,cnt;
	lng clk = GDKms();
	int index[MAXOLTPLOCK];

	(void) stk;
	// prepare probing the lock table
	MT_lock_set(&mal_oltpLock);
	for( i=1; i< pci->argc; i++){
		for(j=0; j< top; j++)
			if( strcmp(locks[j].lckname, getVarConstant(mb, getArg(pci,i)).val.sval) == 0){
				index[i] = j;
				goto next;
			}
		if( j == MAXOLTPLOCK){
			MT_lock_unset(&mal_oltpLock);
			return MAL_SUCCEED;
		}
		if( j == top && top < MAXOLTPLOCK){
			strcpy(locks[top].lckname, getVarConstant(mb, getArg(pci,i)).val.sval);
			index[i] = j;
			top++;
		}
		next: /*nothing*/;
	}
	MT_lock_unset(&mal_oltpLock);

	do{
		MT_lock_set(&mal_oltpLock);
		// check if all the locks are available first
		cnt = 0;
		for( i=1; i< pci->argc; i++)
			cnt +=(locks[index[i]].cntxt == cntxt || locks[index[i]].cntxt == 0);

		if( cnt == pci->argc -1){
			for( i=1; i< pci->argc; i++)
				locks[index[i]].cntxt = cntxt;
			//OLTPdump_(cntxt,"#grabbed the locks\n");
			MT_lock_unset(&mal_oltpLock);
			return MAL_SUCCEED;
		} else {
			MT_lock_unset(&mal_oltpLock);
			MT_sleep_ms(DELAY);
		}
	} while( GDKms() - clk < LOCKTIMEOUT);
	return MAL_SUCCEED;
}

str
OLTPrelease(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i,j;

	(void) cntxt;
	(void) stk;
	MT_lock_set(&mal_oltpLock);
	for( i=1; i< pci->argc; i++){
		for(j=0; j< top; j++)
			if( strcmp(locks[j].lckname, getVarConstant(mb, getArg(pci,i)).val.sval) == 0){
				locks[j].cntxt = 0;
				continue;
			}
	}
	//OLTPdump_(cntxt, "#released the locks\n");
	MT_lock_unset(&mal_oltpLock);
	return MAL_SUCCEED;
}

str
OLTPdump(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) mb;
	(void) stk;
	(void) pci;
	OLTPdump_(cntxt,"#lock table\n");
	return MAL_SUCCEED;
}
