/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "bitcandidates.h"
#include "mal.h"
#include "mal_instruction.h"
#include "mal_interpreter.h"
#include "sys/param.h"

str
BCLcompress(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    bat *ret = getArgReference_bat(stk,pci,0);
    bat *val = getArgReference_bat(stk,pci,1);
	BAT *b, *bn;
	oid *p,*q, first, last, comp;
	char *o;

    (void) cntxt;
    (void) mb;
	b = BATdescriptor(*val);
	if( b == NULL)
		throw(MAL,"compress",INTERNAL_BAT_ACCESS);
	if ( b->ttype == TYPE_void || isVIEW(b) || BATcount(b) == 0){
		BBPkeepref(*ret = *val);
		return MAL_SUCCEED;
	}
	p = (oid *) Tloc(b,0);
	q = (oid *) Tloc(b,BUNlast(b));

	first = *(p);
	last = *(q-1);
	comp = (last-first)/sizeof(msk) + 2 * sizeof(oid); // at least 2 oids to avoid trivial properties set by BBPkeepref
#ifdef _DEBUG_BITCANDIDATES_
	fprintf(stderr,"# BLCcompress base "BUNFMT" first "BUNFMT" range " BUNFMT" count "BUNFMT" vector " BUNFMT"\n", b->tseqbase, first, last, BATcount(b), comp);
#endif

	bn = COLnew(0, TYPE_msk, comp, TRANSIENT);
	if( bn == NULL)
		throw(MAL,"compress",MAL_MALLOC_FAIL);
	/* zap the bitvector */
	o = (char *) Tloc(bn,0);
	memset(o, 0, comp);
	for( ; p < q; p++){
		setbit(o, (*p - first));
#ifdef _DEBUG_BITCANDIDATES_
		fprintf(stderr,"# set value " BUNFMT" bit "BUNFMT"\n", *p,  (*p -first));
#endif
	}
	BATsetcount(bn, comp);
	bn->hseqbase = b->hseqbase;
	bn->tseqbase = first;
	bn->tsorted = b->tsorted;
	bn->trevsorted = b->trevsorted;
	bn->tkey = b->tkey;
	bn->tnil = b->tnil;
	bn->tnonil = b->tnonil;
#ifdef _DEBUG_BITCANDIDATES_
	fprintf(stderr,"#compress %d base "OIDFMT","OIDFMT"\n", bn->batCacheid, bn->hseqbase,bn->tseqbase);
#endif
	BBPkeepref(*ret = bn->batCacheid);
	BBPunfix(*val);
	return MAL_SUCCEED;
}

str
BCLdecompress(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    bat *ret = getArgReference_bat(stk,pci,0);
    bat *val = getArgReference_bat(stk,pci,1);
	BAT *b, *bn;
	oid o = 0, *p;
	BUN i, limit;
	char *vect;

    (void) cntxt;
    (void) mb;
	b = BATdescriptor(*val);
	if( b == NULL)
		throw(MAL,"decompress",INTERNAL_BAT_ACCESS);
	if ( b->ttype == TYPE_void || isVIEW(b) || BATcount(b) == 0){
		BBPkeepref(*ret = *val);
		return MAL_SUCCEED;
	}
	vect = (char*) Tloc(b,0);
#ifdef _DEBUG_BITCANDIDATES_
	fprintf(stderr,"#decompress %d base "OIDFMT","OIDFMT"\n", b->batCacheid, b->hseqbase,b->tseqbase);
#endif

	limit= BATcount(b) * sizeof(msk);
	bn = COLnew(0, TYPE_oid, limit, TRANSIENT);
	if ( bn == 0)
		throw(MAL,"decompress",MAL_MALLOC_FAIL);
	p = (oid*) Tloc(bn,0);
	o = b->tseqbase;
	for ( i = 0; i < limit; i++, o++)
	if( isset( vect, i) ){
#ifdef _DEBUG_BITCANDIDATES_
		fprintf(stderr,"#test "BUNFMT" oid "BUNFMT"\n", i,o);
#endif
		*p++ = o;
	}
	BATsetcount(bn,p - (oid*) Tloc(bn,0));
#ifdef _DEBUG_BITCANDIDATES_
	fprintf(stderr,"#decompress %d base "OIDFMT","OIDFMT" cnt "BUNFMT"\n", bn->batCacheid, b->hseqbase,b->tseqbase, BATcount(bn));
#endif
	bn->hseqbase = b->hseqbase;
	bn->tseqbase = 0;
	bn->tsorted = b->tsorted;
	bn->trevsorted = b->trevsorted;
	bn->tkey = b->tkey;
	bn->tnil = b->tnil;
	bn->tnonil = b->tnonil;

	BBPkeepref(*ret = bn->batCacheid);
	BBPunfix(b->batCacheid);
	return MAL_SUCCEED;
}
