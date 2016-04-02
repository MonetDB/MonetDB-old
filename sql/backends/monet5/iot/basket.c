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

/* author: M. Kersten
 * Continuous query processing relies on event baskets
 * passed through a processing pipeline. The baskets
 * are derived from ordinary SQL tables where the delta
 * processing is ignored.
 *
 */

#include "monetdb_config.h"
#include <gdk.h>
#include "iot.h"
#include "basket.h"
#include "mal_exception.h"
#include "mal_builder.h"
#include "opt_prelude.h"

//#define _DEBUG_BASKET_ if(0)
#define _DEBUG_BASKET_ 

str statusname[6] = { "<unknown>", "init", "paused", "running", "stop", "error" };

BasketRec *baskets;   /* the global iot catalog */
static int bsktTop = 0, bsktLimit = 0;

// Find an empty slot in the basket catalog
static int BSKTnewEntry(void)
{
	int i;
	if (bsktLimit == 0) {
		bsktLimit = MAXBSKT;
		baskets = (BasketRec *) GDKzalloc(bsktLimit * sizeof(BasketRec));
		bsktTop = 1; /* entry 0 is used as non-initialized */
	} else if (bsktTop +1 == bsktLimit) {
		bsktLimit += MAXBSKT;
		baskets = (BasketRec *) GDKrealloc(baskets, bsktLimit * sizeof(BasketRec));
	}
	for (i = 1; i < bsktLimit; i++)
		if (baskets[i].table_name == NULL)
			break;
	bsktTop++;
	return i;
}


// free all malloced space
static void
BSKTclean(int idx)
{
	GDKfree(baskets[idx].schema_name);
	GDKfree(baskets[idx].table_name);
	baskets[idx].schema_name = NULL;
	baskets[idx].table_name = NULL;

	BBPreclaim(baskets[idx].errors);
	baskets[idx].errors = NULL;
	baskets[idx].count = 0;
	MT_lock_destroy(&baskets[idx].lock);
}

// locate the basket in the catalog
int
BSKTlocate(str sch, str tbl)
{
	int i;

	if( sch == 0 || tbl == 0)
		return 0;
	for (i = 1; i < bsktTop; i++)
		if (baskets[i].schema_name && strcmp(sch, baskets[i].schema_name) == 0 &&
			baskets[i].table_name && strcmp(tbl, baskets[i].table_name) == 0)
			return i;
	return 0;
}

// Instantiate a basket description for a particular table
static str
BSKTnewbasket(sql_schema *s, sql_table *t)
{
	int idx;
	node *o;

	// Don't introduce the same basket twice
	if( BSKTlocate(s->base.name, t->base.name) > 0)
		return MAL_SUCCEED;
	//MT_lock_set(&iotLock);
	idx = BSKTnewEntry();
	MT_lock_init(&baskets[idx].lock,"newbasket");

	baskets[idx].schema_name = GDKstrdup(s->base.name);
	baskets[idx].table_name = GDKstrdup(t->base.name);
	baskets[idx].seen = * timestamp_nil;

	baskets[idx].count = 0;
	for (o = t->columns.set->h; o; o = o->next){
        sql_column *col = o->data;
        int tpe = col->type.type->localtype;

        if ( !(tpe < TYPE_str || tpe == TYPE_date || tpe == TYPE_daytime || tpe == TYPE_timestamp) )
			throw(MAL,"baskets.register","Unsupported type %d",tpe);
		baskets[idx].count++;
	}
	baskets[idx].errors = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (baskets[idx].table_name == NULL ||
	    baskets[idx].errors == NULL) {
		BSKTclean(idx);
		MT_lock_unset(&iotLock);
		throw(MAL,"baskets.register",MAL_MALLOC_FAIL);
	}

	baskets[idx].schema = s;
	baskets[idx].table = t;
	//MT_lock_unset(&iotLock);
	return MAL_SUCCEED;
}

// MAL/SQL interface for registration of a single table
str
BSKTregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	sql_schema  *s;
	sql_table   *t;
	mvc *m = NULL;
	str msg = getSQLContext(cntxt, mb, &m, NULL);
	str sch, tbl;

	if ( msg != MAL_SUCCEED)
		return msg;
	if( stk == 0){
		sch = getVarConstant(mb, getArg(pci,1)).val.sval;
		tbl = getVarConstant(mb, getArg(pci,2)).val.sval;
	} else{
		sch = *getArgReference_str(stk, pci, 1);
		tbl = *getArgReference_str(stk, pci, 2);
	}

	/* check double registration */
	if( BSKTlocate(sch, tbl) > 0)
		return msg;
	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
		return msg;

	s = mvc_bind_schema(m, sch);
	if (s == NULL)
		throw(SQL, "iot.register", "Schema missing");

	t = mvc_bind_table(m, s, tbl);
	if (t == NULL)
		throw(SQL, "iot.register", "Table missing '%s'", tbl);

	msg=  BSKTnewbasket(s, t);
	return msg;
}

str
BSKTbind(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat *ret = getArgReference_bat(stk,pci,0);
	str sch = *getArgReference_str(stk,pci,1);
	str tbl = *getArgReference_str(stk,pci,2);
	str col = *getArgReference_str(stk,pci,3);
	int idx;
	mvc *m = NULL;
	sql_schema *s = NULL;
	sql_table *t = NULL;
	sql_column *c = NULL;
	BAT *b;
	str msg;

	(void) mb;
	*ret = 0;

	msg= getSQLContext(cntxt,NULL, &m, NULL);
	if( msg != MAL_SUCCEED)
		return msg;
	s= mvc_bind_schema(m, sch);
	if ( s)
		t= mvc_bind_table(m, s, tbl);
	if ( t)
		c= mvc_bind_column(m, t, col);

	idx= BSKTlocate(sch,tbl);
	if (idx <= 0){
		msg=  BSKTnewbasket(s, t);
		if ( msg != MAL_SUCCEED)
			return msg;
	}

	if( c){
		b = store_funcs.bind_col(m->session->tr,c,RD_UPD_VAL);
		if( b)
			BBPkeepref(*ret =  b->batCacheid);
		return MAL_SUCCEED;
	}
	throw(SQL,"iot.bind","Stream table column '%s.%s.%s' not found",sch,tbl,col);
}

/*
 * The locks are designated towards the baskets.
 * If you can not grab the lock then we have to wait.
 */
str BSKTlock(void *ret, str *sch, str *tbl, int *delay)
{
	int bskt;

	bskt = BSKTlocate(*sch, *tbl);
	if (bskt <= 0)
		throw(SQL, "basket.lock", "Could not find the basket %s.%s",*sch,*tbl);
	_DEBUG_BASKET_ mnstr_printf(BSKTout, "lock group %s.%s\n", *sch, *tbl);
	MT_lock_set(&baskets[bskt].lock);
	_DEBUG_BASKET_ mnstr_printf(BSKTout, "got  group locked %s.%s\n", *sch, *tbl);
	(void) delay;  /* control spinlock */
	(void) ret;
	return MAL_SUCCEED;
}


str BSKTlock2(void *ret, str *sch, str *tbl)
{
	int delay = 0;
	return BSKTlock(ret, sch, tbl, &delay);
}

str BSKTunlock(void *ret, str *sch,str *tbl)
{
	int bskt;

	(void) ret;
	bskt = BSKTlocate(*sch,*tbl);
	if (bskt == 0)
		throw(SQL, "basket.lock", "Could not find the basket %s.%s",*sch,*tbl);
	MT_lock_unset(&baskets[bskt].lock);
	return MAL_SUCCEED;
}


str
BSKTdrop(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int bskt;
	str sch= *getArgReference_str(stk,pci,1);
	str tbl= *getArgReference_str(stk,pci,2);

	(void) cntxt;
	(void) mb;
	bskt = BSKTlocate(sch,tbl);
	if (bskt == 0)
		throw(SQL, "basket.drop", "Could not find the basket %s.%s",sch,tbl);
	BSKTclean(bskt);
	return MAL_SUCCEED;
}

str
BSKTreset(void *ret)
{
	int i;
	(void) ret;
	for (i = 1; i < bsktLimit; i++)
		if (baskets[i].table_name)
			BSKTclean(i);
	return MAL_SUCCEED;
}

/* collect the binary files and append them to what we have */
str 
BSKTpush(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    str sch = *getArgReference_str(stk, pci, 1);
    str tbl = *getArgReference_str(stk, pci, 2);
    str dir = *getArgReference_str(stk, pci, 3);
    int bskt;
	char buf[BUFSIZ];
	node *n;
	mvc *m = NULL;
	BAT *b;
	int first=1;
	BUN cnt =0;
	str msg;

	msg= getSQLContext(cntxt,NULL, &m, NULL);
	if( msg != MAL_SUCCEED)
		return msg;
    bskt = BSKTlocate(sch,tbl);
	if (bskt == 0)
		throw(SQL, "iot.push", "Could not find the basket %s.%s",sch,tbl);

	// check access permission to directory first
	if( access (dir , F_OK | R_OK)){
		throw(SQL, "iot.push", "Could not access the basket directory %s. error %d",dir,errno);
	}
	
	// types are already checked during stream initialization
	MT_lock_set(&baskets[bskt].lock);
	for( n = baskets[bskt].table->columns.set->h; n; n= n->next){
		sql_column *c = n->data;
		snprintf(buf,BUFSIZ, "%s%c%s",dir,DIR_SEP, c->base.name);
		_DEBUG_BASKET_ mnstr_printf(BSKTout,"Attach the file %s\n",buf);
		BATattach(c->type.type->localtype,buf,PERSISTENT);
		b = store_funcs.bind_col(m->session->tr,c,RD_UPD_VAL);
		if( b){ 
			baskets[bskt].count = BATcount(b);
			BBPunfix(b->batCacheid);
			if( first){
				cnt = BATcount(b);
				first = 0;
			} else
				if( cnt != BATcount(b)){
					MT_lock_unset(&baskets[bskt].lock);
					throw(MAL,"iot.push","Non-aligned binary input files");
				}
		}
	}
	MT_lock_unset(&baskets[bskt].lock);
    (void) mb;
    return MAL_SUCCEED;
}
str
BSKTdump(void *ret)
{
	int bskt;
	BUN cnt;
	BAT *b;
	node *n;
	sql_column *c;
	mvc *m = NULL;
	str msg = MAL_SUCCEED;

	mnstr_printf(GDKout, "#baskets table\n");
	for (bskt = 1; bskt < bsktLimit; bskt++)
		if (baskets[bskt].table_name) {
			msg = getSQLContext(mal_clients, 0, &m, NULL);
			if ( msg != MAL_SUCCEED)
				break;
			cnt = 0;
			n = baskets[bskt].table->columns.set->h;
			c = n->data;
			b = store_funcs.bind_col(m->session->tr,c,RD_UPD_VAL);
			if( b){
				cnt = BATcount(b);
				BBPunfix(b->batCacheid);
			}

			mnstr_printf(GDKout, "#baskets[%2d] %s.%s columns %d threshold %d window=[%d,%d] time window=[" LLFMT "," LLFMT "] beat " LLFMT " milliseconds" BUNFMT"\n",
					bskt,
					baskets[bskt].schema_name,
					baskets[bskt].table_name,
					baskets[bskt].count,
					baskets[bskt].threshold,
					baskets[bskt].winsize,
					baskets[bskt].winstride,
					baskets[bskt].timeslice,
					baskets[bskt].timestride,
					baskets[bskt].beat,
					cnt);
		}

	(void) ret;
	return msg;
}

str
BSKTappend(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    int *res = getArgReference_int(stk, pci, 0);
    mvc *m = NULL;
    str msg;
    str sname = *getArgReference_str(stk, pci, 2);
    str tname = *getArgReference_str(stk, pci, 3);
    str cname = *getArgReference_str(stk, pci, 4);
    ptr ins = getArgReference(stk, pci, 5);
    int tpe = getArgType(mb, pci, 5);
    sql_schema *s;
    sql_table *t;
    sql_column *c;
    BAT *bn=0, *b = 0;

    *res = 0;
    if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != NULL)
        return msg;
    if ((msg = checkSQLContext(cntxt)) != NULL)
        return msg;
    if (tpe > GDKatomcnt)
        tpe = TYPE_bat;
    if (tpe == TYPE_bat && (ins = BATdescriptor(*(int *) ins)) == NULL)
        throw(SQL, "basket.append", "Cannot access descriptor");
    if (ATOMextern(tpe))
        ins = *(ptr *) ins;
    if ( tpe == TYPE_bat)
        b =  (BAT*) ins;

    s = mvc_bind_schema(m, sname);
    if (s == NULL)
        throw(SQL, "basket.append", "Schema missing");
    t = mvc_bind_table(m, s, tname);
	if ( t)
		c= mvc_bind_column(m, t, cname);
	else throw(SQL,"basket.append","Stream table %s.%s not accessible\n",sname,tname);
	if( c) {
		bn = store_funcs.bind_col(m->session->tr,c,RD_UPD_VAL);
		if( bn){
			if( tpe == TYPE_bat)
				BATappend(bn, b, TRUE);
			else BUNappend(bn, ins, TRUE);
			BBPunfix(bn->batCacheid);
		}
	} else throw(SQL,"basket.append","Stream column %s.%s.%s not accessible\n",sname,tname,cname);
	if (tpe == TYPE_bat) {
		BBPunfix(((BAT *) ins)->batCacheid);
	}
	return MAL_SUCCEED;
}

InstrPtr
BSKTupdateInstruction(MalBlkPtr mb, str sch, str tbl)
{
	(void) mb;
	(void) sch;
	(void) tbl;
	return NULL;
}

/* provide a tabular view for inspection */
str
BSKTtable(bat *schemaId, bat *nameId, bat *thresholdId, bat * winsizeId, bat *winstrideId, bat *timesliceId, bat *timestrideId, bat *beatId, bat *seenId, bat *eventsId)
{
	BAT *schema= NULL, *name = NULL, *seen = NULL, *events = NULL;
	BAT *threshold = NULL, *winsize = NULL, *winstride = NULL, *beat = NULL;
	BAT *timeslice = NULL, *timestride = NULL;
	int i;

	schema = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (schema == 0)
		goto wrapup;
	BATseqbase(schema, 0);
	name = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (name == 0)
		goto wrapup;
	BATseqbase(name, 0);
	threshold = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (threshold == 0)
		goto wrapup;
	BATseqbase(threshold, 0);
	winsize = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (winsize == 0)
		goto wrapup;
	BATseqbase(winsize, 0);
	winstride = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (winstride == 0)
		goto wrapup;
	BATseqbase(winstride, 0);
	beat = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (beat == 0)
		goto wrapup;
	BATseqbase(beat, 0);
	seen = BATnew(TYPE_void, TYPE_timestamp, BATTINY, TRANSIENT);
	if (seen == 0)
		goto wrapup;
	BATseqbase(seen, 0);
	events = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (events == 0)
		goto wrapup;
	BATseqbase(events, 0);

	timeslice = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (timeslice == 0)
		goto wrapup;
	BATseqbase(timeslice, 0);
	timestride = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (timestride == 0)
		goto wrapup;
	BATseqbase(timestride, 0);

	for (i = 1; i < bsktTop; i++)
		if (baskets[i].table_name) {
			BUNappend(schema, baskets[i].schema_name, FALSE);
			BUNappend(name, baskets[i].table_name, FALSE);
			BUNappend(threshold, &baskets[i].threshold, FALSE);
			BUNappend(winsize, &baskets[i].winsize, FALSE);
			BUNappend(winstride, &baskets[i].winstride, FALSE);
			BUNappend(beat, &baskets[i].beat, FALSE);
			BUNappend(seen, &baskets[i].seen, FALSE);
			baskets[i].events = 0; //(int) BATcount( baskets[i].bats[0]);
			BUNappend(events, &baskets[i].events, FALSE);
			BUNappend(timeslice, &baskets[i].timeslice, FALSE);
			BUNappend(timestride, &baskets[i].timestride, FALSE);
		}

	BBPkeepref(*schemaId = schema->batCacheid);
	BBPkeepref(*nameId = name->batCacheid);
	BBPkeepref(*thresholdId = threshold->batCacheid);
	BBPkeepref(*winsizeId = winsize->batCacheid);
	BBPkeepref(*winstrideId = winstride->batCacheid);
	BBPkeepref(*timesliceId = timeslice->batCacheid);
	BBPkeepref(*timestrideId = timestride->batCacheid);
	BBPkeepref(*beatId = beat->batCacheid);
	BBPkeepref(*seenId = seen->batCacheid);
	BBPkeepref(*eventsId = events->batCacheid);
	return MAL_SUCCEED;
wrapup:
	if (name)
		BBPunfix(name->batCacheid);
	if (threshold)
		BBPunfix(threshold->batCacheid);
	if (winsize)
		BBPunfix(winsize->batCacheid);
	if (winstride)
		BBPunfix(winstride->batCacheid);
	if (timeslice)
		BBPunfix(timeslice->batCacheid);
	if (timestride)
		BBPunfix(timestride->batCacheid);
	if (beat)
		BBPunfix(beat->batCacheid);
	if (seen)
		BBPunfix(seen->batCacheid);
	if (events)
		BBPunfix(events->batCacheid);
	throw(SQL, "iot.baskets", MAL_MALLOC_FAIL);
}

str
BSKTtableerrors(bat *nameId, bat *errorId)
{
	BAT  *name, *error;
	BATiter bi;
	BUN p, q;
	int i;
	name = BATnew(TYPE_void, TYPE_str, BATTINY, PERSISTENT);
	if (name == 0)
		throw(SQL, "baskets.errors", MAL_MALLOC_FAIL);
	error = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (error == 0) {
		BBPunfix(name->batCacheid);
		throw(SQL, "baskets.errors", MAL_MALLOC_FAIL);
	}

	for (i = 1; i < bsktTop; i++)
		if (BATcount(baskets[i].errors) > 0) {
			bi = bat_iterator(baskets[i].errors);
			BATloop(baskets[i].errors, p, q)
			{
				str err = BUNtail(bi, p);
				BUNappend(name, &baskets[i].table_name, FALSE);
				BUNappend(error, err, FALSE);
			}
		}


	BBPkeepref(*nameId = name->batCacheid);
	BBPkeepref(*errorId = error->batCacheid);
	return MAL_SUCCEED;
}
