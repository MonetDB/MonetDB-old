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
#include <unistd.h>
#include "gdk.h"
#include "iot.h"
#include "basket.h"
#include "mal_exception.h"
#include "mal_builder.h"
#include "opt_prelude.h"

#define _DEBUG_BASKET_ if(0)

str statusname[3] = { "<unknown>", "waiting", "filled" };

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
	MT_lock_init(&baskets[i].lock,"bsktlock");

	bsktTop++;
	return i;
}


// free all malloced space
void
BSKTclean(int idx)
{
	if( idx){
		GDKfree(baskets[idx].schema_name);
		GDKfree(baskets[idx].table_name);
		baskets[idx].schema_name = NULL;
		baskets[idx].table_name = NULL;

		BBPreclaim(baskets[idx].errors);
		baskets[idx].errors = NULL;
		baskets[idx].winstride = -1;
		baskets[idx].count = 0;
	}
	for(idx = 1; idx < bsktTop; idx++){
		GDKfree(baskets[idx].schema_name);
		GDKfree(baskets[idx].table_name);
		baskets[idx].schema_name = NULL;
		baskets[idx].table_name = NULL;

		BBPreclaim(baskets[idx].errors);
		baskets[idx].errors = NULL;
		baskets[idx].winstride = -1;
		baskets[idx].count = 0;
	}
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
BSKTnewbasket(mvc *m, sql_schema *s, sql_table *t)
{
	int i, idx, colcnt=0;
	BAT *b;
	node *o;

	// Don't introduce the same basket twice
	if( BSKTlocate(s->base.name, t->base.name) > 0)
		return MAL_SUCCEED;
	MT_lock_set(&iotLock);
	idx = BSKTnewEntry();

	baskets[idx].schema_name = GDKstrdup(s->base.name);
	baskets[idx].table_name = GDKstrdup(t->base.name);
	(void) MTIMEcurrent_timestamp(&baskets[idx].seen);

	baskets[idx].status = BSKTWAIT;
	baskets[idx].count = 0;

	baskets[idx].winstride = -1; /* all tuples are removed */
	
	// Check the column types first
	for (o = t->columns.set->h; o && colcnt <MAXCOLS; o = o->next){
        sql_column *col = o->data;
        int tpe = col->type.type->localtype;

        if ( !(tpe <= TYPE_str || tpe == TYPE_date || tpe == TYPE_daytime || tpe == TYPE_timestamp) ){
			MT_lock_unset(&iotLock);
			throw(MAL,"baskets.register","Unsupported type %d",tpe);
		}
		colcnt++;
	}
	if( colcnt == MAXCOLS){
		BSKTclean(idx);
		throw(MAL,"baskets.register","too many columns");
	}

	// collect the column names and the storage
	i=0;
	for ( i=0, o = t->columns.set->h; i <colcnt && o; o = o->next){
        sql_column *col = o->data;
		b = store_funcs.bind_col(m->session->tr,col,RD_INS);
		assert(b);
		BBPfix(b->batCacheid);
		baskets[idx].bats[i]= b;
		baskets[idx].cols[i++]=  GDKstrdup(col->base.name);
	}
	
	// Create the error table
	baskets[idx].errors = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (baskets[idx].table_name == NULL || baskets[idx].errors == NULL) {
		BSKTclean(idx);
		MT_lock_unset(&iotLock);
		throw(MAL,"baskets.register",MAL_MALLOC_FAIL);
	}
	MT_lock_unset(&iotLock);
	return MAL_SUCCEED;
}

// MAL/SQL interface for registration of a single table
str
BSKTregisterInternal(Client cntxt, MalBlkPtr mb, str sch, str tbl)
{
	sql_schema  *s;
	sql_table   *t;
	mvc *m = NULL;
	str msg = getSQLContext(cntxt, mb, &m, NULL);

	if ( msg != MAL_SUCCEED)
		return msg;

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

	msg=  BSKTnewbasket(m, s, t);
	return msg;
}

str
BSKTregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch, tbl;
	str msg= MAL_SUCCEED;

	(void) stk;
	(void) pci;
	sch = getVarConstant(mb, getArg(pci,2)).val.sval;
	tbl = getVarConstant(mb, getArg(pci,3)).val.sval;
	msg = BSKTregisterInternal(cntxt,mb,sch,tbl);
	// also lock the basket
	if( msg == MAL_SUCCEED){
	}
	return msg;
}

str
BSKTheartbeat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch = *getArgReference_str(stk,pci,1);
	str tbl = *getArgReference_str(stk,pci,2);
	int ticks = *getArgReference_int(stk,pci,3);
	int idx;

	(void) cntxt;
	(void) mb;

	if( ticks < 0)
		throw(SQL,"basket.heartbeat","Positive heartbeat expected]n");
	idx = BSKTlocate(sch, tbl);
	if( idx == 0){
		BSKTregisterInternal(cntxt, mb, sch, tbl);
		idx = BSKTlocate(sch, tbl);
		if( idx ==0)
			throw(SQL,"basket.heartbeat","Stream table %s.%s not accessible to deactivate\n",sch,tbl);
	}
	baskets[idx].heartbeat = ticks;
	return MAL_SUCCEED;
}

str
BSKTgetheartbeat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int *ret = getArgReference_int(stk,pci,0);
	str sch = *getArgReference_str(stk,pci,1);
	str tbl = *getArgReference_str(stk,pci,2);
	int idx;

	(void) cntxt;
	(void) mb;

	idx = BSKTlocate(sch, tbl);
	if( idx == 0)
		throw(SQL,"basket.heartbeat","Stream table %s.%s not accessible to deactivate\n",sch,tbl);
	*ret = baskets[idx].heartbeat;
	return MAL_SUCCEED;
}

str
BSKTwindow(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch = *getArgReference_str(stk,pci,1);
	str tbl = *getArgReference_str(stk,pci,2);
	int elm = *getArgReference_int(stk,pci,3);
	int idx;

	(void) cntxt;
	(void) mb;
	if( elm <= 0)
		throw(SQL,"basket.window","Positive slice expected]n");
	idx = BSKTlocate(sch, tbl);
	if( idx == 0){
		BSKTregisterInternal(cntxt, mb, sch, tbl);
		idx = BSKTlocate(sch, tbl);
		if( idx ==0)
			throw(SQL,"basket.window","Stream table %s.%s not accessible to deactivate\n",sch,tbl);
	}
	baskets[idx].winsize = elm;
	baskets[idx].winstride = elm;
	return MAL_SUCCEED;
}

str
BSKTgetwindow(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int *ret = getArgReference_int(stk,pci,0);
	str sch = *getArgReference_str(stk,pci,1);
	str tbl = *getArgReference_str(stk,pci,2);
	int idx;

	(void) cntxt;
	(void) mb;
	idx = BSKTlocate(sch, tbl);
	if( idx == 0)
		throw(SQL,"basket.window","Stream table %s.%s not accessible to deactivate\n",sch,tbl);
	*ret = baskets[idx].winsize;
	return MAL_SUCCEED;
}

static BAT *
BSKTbindColumn(str sch, str tbl, str col)
{
	int idx =0,i;

	if( (idx = BSKTlocate(sch,tbl)) < 0)
		return NULL;

	for( i=0; i < MAXCOLS && baskets[idx].cols[i]; i++)
		if( strcmp(baskets[idx].cols[i], col)== 0)
			break;
	if(  i < MAXCOLS)
		return baskets[idx].bats[i];
	return NULL;
}

str
BSKTtid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat *ret = getArgReference_bat(stk,pci,0);
	str sch = *getArgReference_str(stk,pci,2);
	str tbl = *getArgReference_str(stk,pci,3);
	BAT *tids, *b;
	int bskt;
	
	(void) cntxt;
	(void) mb;

	bskt = BSKTlocate(sch,tbl);
	if( bskt == 0)	
		throw(SQL,"basket.bind","Stream table column '%s.%s' not found",sch,tbl);
	b = baskets[bskt].bats[0];

    tids = BATnew(TYPE_void, TYPE_void, 0, TRANSIENT);
    if (tids == NULL)
        throw(SQL, "basket.tid", MAL_MALLOC_FAIL);
    BATsetcount(tids, BATcount(b));
    BATseqbase(tids, 0);
    BATseqbase(BATmirror(tids), 0);

	BBPkeepref( *ret = tids->batCacheid);
	return MAL_SUCCEED;
}

str
BSKTbind(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat *ret = getArgReference_bat(stk,pci,0);
	str sch = *getArgReference_str(stk,pci,2);
	str tbl = *getArgReference_str(stk,pci,3);
	str col = *getArgReference_str(stk,pci,4);
	BAT *bn, *b = BSKTbindColumn(sch,tbl,col);
	int bskt;

	(void) cntxt;
	(void) mb;

	*ret = 0;
	if( b){
		bskt = BSKTlocate(sch,tbl);
		if( bskt > 0){
			if( baskets[bskt].winsize >0){
				bn = VIEWcreate(0,b);
				if( bn){
					VIEWbounds(b,bn, 0, baskets[bskt].winsize);
					BBPkeepref(*ret =  bn->batCacheid);
				} else
					throw(SQL,"iot.bind","Can not create view %s.%s.%s["BUNFMT"]",sch,tbl,col,baskets[bskt].winsize );
			} else{
				BBPkeepref( *ret = b->batCacheid);
				BBPfix(b->batCacheid); // don't loose it
			}
		}
		return MAL_SUCCEED;
	}
	throw(SQL,"iot.bind","Stream table column '%s.%s.%s' not found",sch,tbl,col);
}

str
BSKTdrop(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int bskt;
	str sch= *getArgReference_str(stk,pci,2);
	str tbl= *getArgReference_str(stk,pci,3);

	(void) cntxt;
	(void) mb;
	bskt = BSKTlocate(sch,tbl);
	if (bskt == 0)
		throw(SQL, "basket.drop", "Could not find the basket %s.%s",sch,tbl);
	MT_lock_set(&iotLock);
	BSKTclean(bskt);
	MT_lock_unset(&iotLock);
	return MAL_SUCCEED;
}

/* collect the binary files and append them to what we have */
#define MAXLINE 4096
str
BSKTimportInternal(Client cntxt, int bskt)
{
	char buf[PATHLENGTH];
	BAT *b;
	int first=1,i,j;
	BUN cnt =0, bcnt=0;
	str msg= MAL_SUCCEED;
	FILE *f;
	long fsize;
	char line[MAXLINE];
	str dir = baskets[bskt].source;
	str cname= NULL;

	(void)cntxt;
	// check access permission to directory first
	if( access (dir , F_OK | R_OK)){
		throw(SQL, "iot.basket", "Could not access the basket directory %s. error %d",dir,errno);
	}
	
	/* check for missing files */
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		cname = baskets[bskt].cols[i];
		snprintf(buf,PATHLENGTH, "%s%c%s",dir,DIR_SEP, cname);
		_DEBUG_BASKET_ mnstr_printf(BSKTout,"Attach the file %s\n",buf);
		if( access (buf,R_OK))
			throw(MAL,"iot.basket","Could not access the column %s file %s\n",cname, buf);
		b = baskets[bskt].bats[i];
		if( b == 0)
			throw(MAL,"iot.basket","Could not access the column %s\n",cname);
	}

	// types are already checked during stream initialization
	MT_lock_set(&iotLock);
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		cname = baskets[bskt].cols[i];
		snprintf(buf,PATHLENGTH, "%s%c%s",dir,DIR_SEP, cname);
		_DEBUG_BASKET_ mnstr_printf(BSKTout,"Attach the file %s\n",buf);
		f=  fopen(buf,"r");
		if( f == NULL){
			msg= createException(MAL,"iot.basket","Could not access the column %s file %s\n",cname, buf);
			break;
		}
		(void) fseek(f,0, SEEK_END);
		fsize = ftell(f);
		rewind(f);
		b = baskets[bskt].bats[i];
		assert( b);
		bcnt = BATcount(b);

		switch(ATOMstorage(b->ttype)){
		case TYPE_bit:
		case TYPE_bte:
		case TYPE_sht:
		case TYPE_int:
		case TYPE_void:
		case TYPE_oid:
		case TYPE_flt:
		case TYPE_dbl:
		case TYPE_lng:
#ifdef HAVE_HGE
		case TYPE_hge:
#endif
			if( BATextend(b, bcnt + fsize / ATOMsize(b->ttype)) != GDK_SUCCEED){
				(void) fclose(f);
				msg= createException(MAL,"iot.basket","Could not extend basket %s\n",baskets[bskt].cols[i]);
				goto recover;
			}
			/* append the binary partition */
			if( fread(Tloc(b, BUNlast(b)),1,fsize, f) != (size_t) fsize){
				(void) fclose(f);
				msg= createException(MAL,"iot.basket","Could not read complete basket file %s\n",baskets[bskt].cols[i]);
				goto recover;
			}
			BATsetcount(b, bcnt + fsize/ ATOMsize(b->ttype));
		break;
		case TYPE_str:
			while (fgets(line, MAXLINE, f) != 0){ //Use getline? http://man7.org/linux/man-pages/man3/getline.3.html
				if ( line[j= (int) strlen(line)-1] != '\n')
					msg= createException(MAL,"iot.basket","string too long\n");
				else{
					line[j] = 0;
					BUNappend(b, line, TRUE);
					bcnt++;
				}
			}
			BATsetcount(b, bcnt );
			break;
		default:
			msg= createException(MAL,"iot.basket","Import type not yet supported\n");
		}
		(void) fclose(f);
	}

	/* check for mis-aligned columns and derive properties */
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		b = baskets[bskt].bats[i];
		assert( b );
		BATderiveProps(b, FALSE);
		if( first){
			first = 0;
			cnt = BATcount(b);
		} else
		if( cnt != BATcount(b))
			msg= createException(MAL,"iot.basket","Columns mis-aligned %s\n",baskets[bskt].cols[i]);
	}
	/* remove the basket files */
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		snprintf(buf,PATHLENGTH, "%s%c%s",dir,DIR_SEP, baskets[bskt].cols[i]);
		assert( access (buf,R_OK) == 0);
		//unlink(buf);
	}
	baskets[bskt].count = cnt;
	baskets[bskt].status = BSKTFILLED;

recover:
	/* reset all BATs when they are misaligned or error occurred */
	if( msg != MAL_SUCCEED)
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		b = baskets[bskt].bats[i];
		assert( b );
		BATsetcount(b,0);
	}

	MT_lock_unset(&iotLock);
    return msg;
}

str 
BSKTimport(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    str sch = *getArgReference_str(stk, pci, 1);
    str tbl = *getArgReference_str(stk, pci, 2);
    str dir = *getArgReference_str(stk, pci, 3);
    int bskt;

	BSKTregisterInternal(cntxt, mb, sch, tbl);
    bskt = BSKTlocate(sch,tbl);
	if (bskt == 0)
		throw(SQL, "iot.basket", "Could not find the basket %s.%s",sch,tbl);
	baskets[bskt].source = GDKstrdup(dir);
	return BSKTimportInternal(cntxt,bskt);
}

static str
BSKTexportInternal(Client cntxt, int bskt)
{
	char buf[PATHLENGTH];
	BAT *b;
	int i;
	str msg= MAL_SUCCEED;
	FILE *f;
	long fsize;
	str dir = baskets[bskt].source;
	str cname= NULL;

	(void)cntxt;
	// check access permission to directory first
	(void) mkdir(dir,0755);
	if( access (dir , F_OK | R_OK)){
		throw(SQL, "iot.basket", "Could not access the target basket directory %s. error %d",dir,errno);
	}
	
	/* check for leftover files */
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		cname = baskets[bskt].cols[i];
		snprintf(buf,PATHLENGTH, "%s%c%s",dir,DIR_SEP, cname);
		_DEBUG_BASKET_ mnstr_printf(BSKTout,"Check for the file %s\n",buf);
		if( !access (buf,R_OK))
			throw(MAL,"iot.export","Left over %s file %s\n",cname, buf);
		b = baskets[bskt].bats[i];
		if( b == 0)
			throw(MAL,"iot.export","Could not access the column %s\n",cname);
	}

	// types are already checked during stream initialization
	MT_lock_set(&iotLock);
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		cname = baskets[bskt].cols[i];
		snprintf(buf,PATHLENGTH, "%s%c%s",dir,DIR_SEP, cname);
		_DEBUG_BASKET_ mnstr_printf(BSKTout,"Attach the file %s\n",buf);
		f=  fopen(buf,"w");
		if( f == NULL){
			msg= createException(MAL,"iot.export","Could not access the column %s file %s\n",cname, buf);
			break;
		}
		b = baskets[bskt].bats[i];
		assert( b);

		switch(ATOMstorage(b->ttype)){
		case TYPE_bit:
		case TYPE_bte:
		case TYPE_sht:
		case TYPE_int:
		case TYPE_void:
		case TYPE_oid:
		case TYPE_flt:
		case TYPE_dbl:
		case TYPE_lng:
#ifdef HAVE_HGE
		case TYPE_hge:
#endif
			/* append the binary partition */
			fsize = BATcount(b) * ATOMsize(b->ttype);
			if( fwrite(Tloc(b, BUNlast(b)),1,fsize, f) != (size_t) fsize){
				(void) fclose(f);
				msg= createException(MAL,"iot.export","Could not write complete basket file %s\n",baskets[bskt].cols[i]);
				goto recover;
			}
		break;
		case TYPE_str:
			msg= createException(MAL,"iot.export","Export type string not yet supported\n");
			break;
		default:
			msg= createException(MAL,"iot.export","export type not yet supported\n");
		}
		(void) fclose(f);
	}

	/* reset all BATs when they are exported */
	for( i=0; i < MAXCOLS && baskets[bskt].cols[i]; i++){
		b = baskets[bskt].bats[i];
		assert( b );
		BATsetcount(b,0);
	}

recover:
	MT_lock_unset(&iotLock);
    return msg;
}

str 
BSKTexport(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    str sch = *getArgReference_str(stk, pci, 1);
    str tbl = *getArgReference_str(stk, pci, 2);
    str dir = *getArgReference_str(stk, pci, 3);
    int bskt;

	BSKTregisterInternal(cntxt, mb, sch, tbl);
    bskt = BSKTlocate(sch,tbl);
	if (bskt == 0)
		throw(SQL, "iot.basket", "Could not find the basket %s.%s",sch,tbl);
	baskets[bskt].source = GDKstrdup(dir);
	return BSKTexportInternal(cntxt,bskt);
}

/* remove tuples from a basket according to the sliding policy */
#define ColumnShift(B,TPE, STRIDE) { \
	TPE *first= (TPE*) Tloc(B, BUNfirst(B));\
	TPE *n = first+STRIDE;\
	TPE *last=  (TPE*) Tloc(B, BUNlast(B));\
	for( ; n < last; n++, first++)\
		*first=*n;\
}

static str
BSKTtumbleInternal(Client cntxt, str sch, str tbl, int stride)
{
	BAT *b;
	BUN cnt;
	int i, bskt;
	(void) cntxt;

    bskt = BSKTlocate(sch,tbl);
	if (bskt == 0)
		throw(SQL, "iot.tumble", "Could not find the basket %s.%s",sch,tbl);

	_DEBUG_BASKET_ mnstr_printf(BSKTout,"Tumble %s.%s %d elements\n",sch,tbl,stride);
	for(i=0; i< MAXCOLS && baskets[bskt].cols[i]; i++){
		b = baskets[bskt].bats[i];
		assert( b );
		if( stride != -1)
			cnt = (BUN) stride;
		else
			cnt= BATcount(b);

		switch(ATOMstorage(b->ttype)){
		case TYPE_bit:ColumnShift(b,bit,cnt); break;
		case TYPE_bte:ColumnShift(b,bte,cnt); break;
		case TYPE_sht:ColumnShift(b,sht,cnt); break;
		case TYPE_int:ColumnShift(b,int,cnt); break;
		case TYPE_oid:ColumnShift(b,oid,cnt); break;
		case TYPE_flt:ColumnShift(b,flt,cnt); break;
		case TYPE_dbl:ColumnShift(b,dbl,cnt); break;
		case TYPE_lng:ColumnShift(b,lng,cnt); break;
#ifdef HAVE_HGE
		case TYPE_hge:ColumnShift(b,hge,cnt); break;
#endif
		case TYPE_str:
			switch(b->T->width){
			case 1: ColumnShift(b,bte,cnt); break;
			case 2: ColumnShift(b,sht,cnt); break;
			case 4: ColumnShift(b,int,cnt); break;
			case 8: ColumnShift(b,lng,cnt); break;
			}
				break;
		default: break;
		}
		if( stride == -1)
			BATsetcount(b, 0);
		else BATsetcount(b, BATcount(b)-cnt);
		if( BATcount(b) == 0){
			baskets[bskt].status = BSKTWAIT;
		}
		BATderiveProps(b, FALSE);
	}
	return MAL_SUCCEED;
}

/* set the tumbling properties */
str
BSKTtumble(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch;
	str tbl;
	int elm = -1;
	int idx;

	(void) cntxt;
	(void) mb;

	sch = *getArgReference_str(stk,pci,2);
	tbl = *getArgReference_str(stk,pci,3);

	idx = BSKTlocate(sch, tbl);
	if( idx == 0){
		BSKTregisterInternal(cntxt, mb, sch, tbl);
		idx = BSKTlocate(sch, tbl);
		if( idx ==0)
			throw(SQL,"basket.tumble","Stream table %s.%s not accessible \n",sch,tbl);
	}
	/* also take care of time-based tumbling */
	elm =(int) baskets[idx].winstride;
	return BSKTtumbleInternal(cntxt, sch, tbl, elm);
}

str
BSKTgettumble(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int *ret;
	str sch;
	str tbl;
	int idx;

	(void) cntxt;
	(void) mb;

	ret = getArgReference_int(stk,pci,0);
	sch = *getArgReference_str(stk,pci,1);
	tbl = *getArgReference_str(stk,pci,2);

	idx = BSKTlocate(sch, tbl);
	if( idx == 0)
		throw(SQL,"basket.tumble","Stream table %s.%s not accessible \n",sch,tbl);
	*ret  =(int) baskets[idx].winstride;
	return MAL_SUCCEED;
}

str
BSKTsettumble(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch = *getArgReference_str(stk,pci,1);
	str tbl = *getArgReference_str(stk,pci,2);
	int idx;

	(void) cntxt;
	(void) mb;
	idx = BSKTlocate(sch, tbl);
	if( idx ==0)
		throw(SQL,"basket.tumble","Stream table %s.%s not accessible\n",sch,tbl);
	baskets[idx].winstride = *(int*)getArgReference_int(stk,pci,3);
	return MAL_SUCCEED;
}

str
BSKTcommit(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str sch = *getArgReference_str(stk,pci,2);
	str tbl = *getArgReference_str(stk,pci,3);
	int idx;

	(void) cntxt;
	(void) mb;

	idx = BSKTlocate(sch, tbl);
	if( idx ==0)
		throw(SQL,"basket.commit","Stream table %s.%s not accessible\n",sch,tbl);
	/* release the basket lock */
	return MAL_SUCCEED;
}

str
BSKTdump(void *ret)
{
	int bskt;
	BUN cnt;
	BAT *b;
	str msg = MAL_SUCCEED;

	mnstr_printf(GDKout, "#baskets table\n");
	for (bskt = 1; bskt < bsktLimit; bskt++)
		if (baskets[bskt].table_name) {
			cnt = 0;
			b = baskets[bskt].bats[0];
			if( b)
				cnt = BATcount(b);

			mnstr_printf(GDKout, "#baskets[%2d] %s.%s columns "BUNFMT" threshold %d window=["BUNFMT","BUNFMT"] time window=[" LLFMT "," LLFMT "] beat " LLFMT " milliseconds" BUNFMT"\n",
					bskt,
					baskets[bskt].schema_name,
					baskets[bskt].table_name,
					baskets[bskt].count,
					baskets[bskt].threshold,
					baskets[bskt].winsize,
					baskets[bskt].winstride,
					baskets[bskt].timeslice,
					baskets[bskt].timestride,
					baskets[bskt].heartbeat,
					cnt);
		}

	(void) ret;
	return msg;
}

str
BSKTappend(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    int *res = getArgReference_int(stk, pci, 0);
    str sname = *getArgReference_str(stk, pci, 2);
    str tname = *getArgReference_str(stk, pci, 3);
    str cname = *getArgReference_str(stk, pci, 4);
    ptr value = getArgReference(stk, pci, 5);
    int tpe = getArgType(mb, pci, 5);
    BAT *bn=0, *binsert = 0;
	int bskt;
	BUN cnt =0;

	(void) cntxt;
    *res = 0;

    if ( isaBatType(tpe) && (binsert = BATdescriptor(*(int *) value)) == NULL)
        throw(SQL, "basket.append", "Cannot access source descriptor");
	if ( !isaBatType(tpe) && ATOMextern(getColumnType(tpe)))
		value = *(ptr*) value;

	bskt = BSKTlocate(sname,tname);
	if( bskt == 0)
		throw(SQL, "basket.append", "Cannot access basket descriptor %s.%s",sname,tname);
	bn = BSKTbindColumn(sname,tname,cname);

	if( bn){
		if (binsert)
			BATappend(bn, binsert, TRUE);
		else
			BUNappend(bn, value, TRUE);
		cnt = BATcount(bn);
		BATderiveProps(bn, FALSE);
	} else throw(SQL, "basket.append", "Cannot access target column %s.%s.%s",sname,tname,cname);
	
	if(cnt){
		baskets[bskt].count = cnt;
		baskets[bskt].status = BSKTFILLED;
	}
	if (binsert )
		BBPunfix(((BAT *) binsert)->batCacheid);
	return MAL_SUCCEED;
}

str
BSKTupdate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    int *res = getArgReference_int(stk, pci, 0);
    str sname = *getArgReference_str(stk, pci, 2);
    str tname = *getArgReference_str(stk, pci, 3);
    str cname = *getArgReference_str(stk, pci, 4);
    bat rows = *getArgReference_bat(stk, pci, 5);
    bat val = *getArgReference_bat(stk, pci, 6);
    BAT *bn=0, *rid=0, *bval = 0;
	int bskt;

	(void) cntxt;
	(void) mb;
    *res = 0;

    rid = BATdescriptor(rows);
	if( rid == NULL)
        throw(SQL, "basket.append", "Cannot access source oid descriptor");
    bval = BATdescriptor(val);
	if( bval == NULL){
		BBPunfix(rid->batCacheid);
        throw(SQL, "basket.append", "Cannot access source descriptor");
	}

	bskt = BSKTlocate(sname,tname);
	if( bskt == 0)
		throw(SQL, "basket.append", "Cannot access basket descriptor %s.%s",sname,tname);
	bn = BSKTbindColumn(sname,tname,cname);

	if( bn){
		void_replace_bat(bn, rid, bval, TRUE);
		BATderiveProps(bn, FALSE);
	} else throw(SQL, "basket.append", "Cannot access target column %s.%s.%s",sname,tname,cname);
	
	baskets[bskt].status = BSKTFILLED;
	BBPunfix(rid->batCacheid);
	BBPunfix(bval->batCacheid);
	return MAL_SUCCEED;
}

str
BSKTreset(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    lng *res = getArgReference_lng(stk, pci, 0);
    str sname = *getArgReference_str(stk, pci, 2);
    str tname = *getArgReference_str(stk, pci, 3);
	int i, idx;
	BAT *b;
	(void) cntxt;
	(void) mb;

    *res = 0;
	idx = BSKTlocate(sname,tname);
	if( idx <= 0)
		throw(SQL,"basket.clear","Stream table %s.%s not registered \n",sname,tname);
	// do actual work
	MT_lock_set(&baskets[idx].lock);
	for( i=0; baskets[idx].cols[i]; i++){
		b = baskets[idx].bats[i];
		if(b)
			BATsetcount(b,0);
	}
	baskets[idx].status = BSKTWAIT;
	MT_lock_unset(&baskets[idx].lock);
	return MAL_SUCCEED;
}

static str
BSKTerrorInternal(bat *ret, str sname, str tname, str err)
{
	int idx;
	idx = BSKTlocate(sname,tname);
	if( idx == 0)
		throw(SQL,"basket.error","Stream table %s.%s not accessible for commit\n",sname,tname);

	if( baskets[idx].errors == NULL)
		baskets[idx].errors = BATnew(TYPE_void, TYPE_str, 0, TRANSIENT);
		
	if( baskets[idx].errors == NULL)
		throw(SQL,"basket.error",MAL_MALLOC_FAIL);

	BUNappend(baskets[idx].errors, err, FALSE);
	
	BBPkeepref(*ret = baskets[idx].errors->batCacheid);
	return MAL_SUCCEED;
}

str
BSKTerror(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat *ret  = getArgReference_bat(stk,pci,0);
    str sname = *getArgReference_str(stk, pci, 1);
    str tname = *getArgReference_str(stk, pci, 2);
    str err = *getArgReference_str(stk, pci, 3);
	int idx;
	str msg = MAL_SUCCEED;
	(void) cntxt;
	(void) mb;

	idx = BSKTlocate(sname,tname);
	if( idx == 0)
		throw(SQL,"basket.error","Stream table %s.%s not accessible for commit\n",sname,tname);

	MT_lock_set(&iotLock);
	msg = BSKTerrorInternal(ret,sname,tname,err);
	MT_lock_unset(&iotLock);
	return msg;
}

/* provide a tabular view for inspection */
str
BSKTtable (Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat *schemaId = getArgReference_bat(stk,pci,0);
	bat *nameId = getArgReference_bat(stk,pci,1);
	bat *statusId = getArgReference_bat(stk,pci,2);
	bat *thresholdId = getArgReference_bat(stk,pci,3);
	bat *winsizeId = getArgReference_bat(stk,pci,4);
	bat *winstrideId = getArgReference_bat(stk,pci,5);
	bat *timesliceId = getArgReference_bat(stk,pci,6);
	bat *timestrideId = getArgReference_bat(stk,pci,7);
	bat *beatId = getArgReference_bat(stk,pci,8);
	bat *seenId = getArgReference_bat(stk,pci,9);
	bat *eventsId = getArgReference_bat(stk,pci,10);
	BAT *schema = NULL, *name = NULL, *status = NULL,  *seen = NULL, *events = NULL;
	BAT *threshold = NULL, *winsize = NULL, *winstride = NULL, *beat = NULL;
	BAT *timeslice = NULL, *timestride = NULL;
	int i;
	BAT *bn = NULL;

	(void) mb;
	(void) cntxt;

	schema = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (schema == 0)
		goto wrapup;
	BATseqbase(schema, 0);
	name = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (name == 0)
		goto wrapup;
	BATseqbase(status, 0);
	status = BATnew(TYPE_void, TYPE_str, BATTINY, TRANSIENT);
	if (status == 0)
		goto wrapup;
	BATseqbase(status, 0);
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
	timeslice = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (timeslice == 0)
		goto wrapup;
	BATseqbase(timeslice, 0);
	timestride = BATnew(TYPE_void, TYPE_int, BATTINY, TRANSIENT);
	if (timestride == 0)
		goto wrapup;
	BATseqbase(timestride, 0);
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


	for (i = 1; i < bsktTop; i++)
		if (baskets[i].table_name) {
			BUNappend(schema, baskets[i].schema_name, FALSE);
			BUNappend(name, baskets[i].table_name, FALSE);
			BUNappend(status, statusname[baskets[i].status], FALSE);
			BUNappend(threshold, &baskets[i].threshold, FALSE);
			BUNappend(winsize, &baskets[i].winsize, FALSE);
			BUNappend(winstride, &baskets[i].winstride, FALSE);
			BUNappend(timeslice, &baskets[i].timeslice, FALSE);
			BUNappend(timestride, &baskets[i].timestride, FALSE);
			BUNappend(beat, &baskets[i].heartbeat, FALSE);
			BUNappend(seen, &baskets[i].seen, FALSE);
			bn = BSKTbindColumn(baskets[i].schema_name, baskets[i].table_name, baskets[i].cols[0]);
			baskets[i].events = bn ? (int) BATcount( bn): 0;
			BUNappend(events, &baskets[i].events, FALSE);
		}

	BBPkeepref(*schemaId = schema->batCacheid);
	BBPkeepref(*nameId = name->batCacheid);
	BBPkeepref(*statusId = status->batCacheid);
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
	if (schema)
		BBPunfix(schema->batCacheid);
	if (name)
		BBPunfix(name->batCacheid);
	if (status)
		BBPunfix(status->batCacheid);
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
		if (baskets[i].errors && BATcount(baskets[i].errors) > 0) {
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
