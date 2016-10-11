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
#include "mtime.h"

#define LOCKTIMEOUT 20 * 1000
#define LOCKDELAY 20

typedef struct{
	Client cntxt;	// user holding the write lock
	lng start;		// time when it started
	str query;
	int used;		// how often it used, for balancing
	int locked;		// writelock set or not
} OLTPlockRecord;

static OLTPlockRecord oltp_locks[MAXOLTPLOCKS];
static int oltp_delay;

/*
static void
OLTPdump_(Client cntxt, str msg)
{
	int i;

	mnstr_printf(cntxt->fdout,"%s",msg);
	for(i=0; i< MAXOLTPLOCKS; i++)
	if( oltp_locks[i].locked)
		mnstr_printf(cntxt->fdout,"#[%i] %3d\n",i, (oltp_locks[i].cntxt ? oltp_locks[i].cntxt->idx: -1));
}
*/

str
OLTPreset(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	int i;
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	
	MT_lock_set(&mal_oltpLock);
	for( i=0; i<MAXOLTPLOCKS; i++){
		oltp_locks[i].locked = 0;
		oltp_locks[i].cntxt = 0;
		oltp_locks[i].start = 0;
		oltp_locks[i].query = 0;
		oltp_locks[i].used = 0;
	}
	MT_lock_unset(&mal_oltpLock);
	return MAL_SUCCEED;
}

str
OLTPenable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	oltp_delay = TRUE;
	return MAL_SUCCEED;
}

str
OLTPdisable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	OLTPreset(cntxt, mb, stk,pci);
	oltp_delay = FALSE;
	return MAL_SUCCEED;
}

str
OLTPinit(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	// nothing to be done right now
	return OLTPreset(cntxt,mb,stk,pci);
}

// The locking is based in the hash-table.
// It contains all write locks outstanding
// A transaction may proceed if no element in its read set is locked

str
OLTPlock(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i,cnt,lck;
	lng clk;
	lng ms = GDKms();
	str sql,cpy;

	(void) stk;
	if ( oltp_delay == FALSE )
		return MAL_SUCCEED;

	clk = (lng) time(0);

	do{
		// checking the collision of read locks with write locks
		// does not require a lock on the table.
		// we can always try it again
		for( i=1; i< pci->argc; i++){
			lck= getVarConstant(mb, getArg(pci,i)).val.ival;
			if( lck < 0 && oltp_locks[-lck].locked )
				goto lockdelay;
		}

		MT_lock_set(&mal_oltpLock);
		// check if all the locks are available 
		cnt = 0;
		for( i=1; i< pci->argc; i++){
			lck= getVarConstant(mb, getArg(pci,i)).val.ival;
			if ( lck > 0)
				cnt += oltp_locks[lck].locked == 0;
			else cnt++;
		}

		if( cnt == pci->argc -1){
			for( i=1; i< pci->argc; i++){
				lck= getVarConstant(mb, getArg(pci,i)).val.ival;
				if( lck > 0){
					oltp_locks[i].cntxt = cntxt;
					oltp_locks[i].start = clk;
					oltp_locks[i].used++;
					oltp_locks[i].locked = 1;
				}
			}
			//OLTPdump_(cntxt,"#grabbed the locks\n");
			MT_lock_unset(&mal_oltpLock);
			return MAL_SUCCEED;
		} else {
			MT_lock_unset(&mal_oltpLock);
lockdelay:
			MT_sleep_ms(LOCKDELAY);
		}
	} while( GDKms() - ms < LOCKTIMEOUT);

	// if the time out is related to a copy_from query, we should not start it either.
	sql = getName("sql");
	cpy = getName("copy_from");

	for( i = 0; i < mb->stop; i++)
		if( getModuleId(getInstrPtr(mb,i)) == sql && getFunctionId(getInstrPtr(mb,i)) == cpy)
			throw(SQL,"oltp.lock","Conflicts with other write operations\n");
	return MAL_SUCCEED;
}

str
OLTPrelease(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i,lck;

	(void) cntxt;
	(void) stk;
	if ( oltp_delay == FALSE )
		return MAL_SUCCEED;

	MT_lock_set(&mal_oltpLock);
	for( i=1; i< pci->argc; i++){
		lck= getVarConstant(mb, getArg(pci,i)).val.ival;
		if( lck > 0){
				oltp_locks[i].cntxt = 0;
				oltp_locks[i].start = 0;
				oltp_locks[i].query = 0;
				oltp_locks[i].locked = 0;
			}
	}
	//OLTPdump_(cntxt, "#released the locks\n");
	MT_lock_unset(&mal_oltpLock);
	return MAL_SUCCEED;
}

str
OLTPtable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	BAT *bs= NULL, *bu= NULL, *bl= NULL, *bq= NULL, *bc = NULL;
	bat *started = getArgReference_bat(stk,pci,0);
	bat *userid = getArgReference_bat(stk,pci,1);
	bat *lockid = getArgReference_bat(stk,pci,2);
	bat *used = getArgReference_bat(stk,pci,3);
	bat *query = getArgReference_bat(stk,pci,4);
	int i;
	lng now;
	str msg = MAL_SUCCEED; 
	timestamp ts, tsn;

	(void) cntxt;
	(void) mb;

	bs = COLnew(0, TYPE_timestamp, 0, TRANSIENT);
	bu = COLnew(0, TYPE_str, 0, TRANSIENT);
	bl = COLnew(0, TYPE_int, 0, TRANSIENT);
	bc = COLnew(0, TYPE_int, 0, TRANSIENT);
	bq = COLnew(0, TYPE_str, 0, TRANSIENT);

	if( bs == NULL || bu == NULL || bl == NULL  || bq == NULL || bc == NULL){
		if( bs) BBPunfix(bs->batCacheid);
		if( bl) BBPunfix(bl->batCacheid);
		if( bu) BBPunfix(bu->batCacheid);
		if( bc) BBPunfix(bc->batCacheid);
		if( bq) BBPunfix(bq->batCacheid);
	}
	for( i = 0; msg ==  MAL_SUCCEED && i < MAXOLTPLOCKS; i++)
	if (oltp_locks[i].locked ){
		now = oltp_locks[i].start * 1000; // convert to timestamp microsecond
		msg= MTIMEunix_epoch(&ts);
		if ( msg == MAL_SUCCEED)
			msg = MTIMEtimestamp_add(&tsn, &ts, &now);

		if( msg== MAL_SUCCEED && oltp_locks[i].start)
			BUNappend(bs, &tsn, FALSE);
		else
			BUNappend(bs, timestamp_nil, FALSE);

		if( oltp_locks[i].cntxt)
			BUNappend(bu, &oltp_locks[i].cntxt->username, FALSE);
		else 
			BUNappend(bu, str_nil, FALSE);
		BUNappend(bl, &oltp_locks[i].locked, FALSE);
		BUNappend(bc, &oltp_locks[i].used, FALSE);
		if( oltp_locks[i].query)
			BUNappend(bq, &oltp_locks[i].query, FALSE);
		else
			BUNappend(bq, str_nil, FALSE);
	}
	//OLTPdump_(cntxt,"#lock table\n");
	BBPkeepref(*started = bs->batCacheid);
	BBPkeepref(*userid = bl->batCacheid);
	BBPkeepref(*lockid = bu->batCacheid);
	BBPkeepref(*used = bc->batCacheid);
	BBPkeepref(*query = bq->batCacheid);
	return msg;
}
