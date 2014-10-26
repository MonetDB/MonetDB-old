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
 * Copyright August 2008-2014 MonetDB B.V.
 * All Rights Reserved.
 */

/* This contains algebra functions used for RDF store only */

#include "monetdb_config.h"
#include "rdf.h"
#include "algebra.h"
#include <gdk.h>
#include "tokenizer.h"

str
RDFleftfetchjoin_sorted(bat *result, const bat *lid, const bat *rid)
{
	BAT *left, *right, *bn = NULL;

	if ((left = BATdescriptor(*lid)) == NULL) {
		throw(MAL, "rdf.leftfetchjoin_sorted", RUNTIME_OBJECT_MISSING);
	}
	if ((right = BATdescriptor(*rid)) == NULL) {
		BBPreleaseref(left->batCacheid);
		throw(MAL, "rdf.leftfetchjoin_sorted", RUNTIME_OBJECT_MISSING);
	}
	bn = BATleftfetchjoin(left, right, BUN_NONE);
	BBPreleaseref(left->batCacheid);
	BBPreleaseref(right->batCacheid);
	if (bn == NULL)
		throw(MAL, "rdf.leftfetchjoin_sorted", GDK_EXCEPTION);

	bn->tsorted = TRUE;

	if (!(bn->batDirty & 2))
		bn = BATsetaccess(bn, BAT_READ);
	*result = bn->batCacheid;
	BBPkeepref(*result);
	return MAL_SUCCEED;
}

/*
str
RDFpartialjoin(bat *retid, bat *lid, bat *rid, bat *inputid){
	BAT *left, *right, *result, *map, *input;  
	BATiter resulti,inputi;
	BUN	p,q; 
	oid	*rbt; 
	oid	*ibt; 
	

	if ((left = BATdescriptor(*lid)) == NULL) {
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}
	if ((right = BATdescriptor(*rid)) == NULL) {
		BBPreleaseref(left->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	if ((input = BATdescriptor(*inputid)) == NULL) {
		BBPreleaseref(left->batCacheid);
		BBPreleaseref(right->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	map = VIEWcreate(BATmirror(left), right);
	//map = BATleftfetchjoin(BATmirror(left), right, BUN_NONE);

	BBPreleaseref(left->batCacheid);
	BBPreleaseref(right->batCacheid);

	//BATprint(map); 

	result = BATouterjoin(input, map, BUN_NONE); 

	resulti = bat_iterator(result);
	inputi = bat_iterator(input);

	BATloop(result, p, q){
		rbt = (oid *) BUNtloc(resulti, p); 
		if (*rbt == oid_nil){
			ibt = (oid *) BUNtloc(inputi, p); 
			*rbt = *ibt; 
		}
	}

	BBPreleaseref(input->batCacheid);

	//BATprint(result); 
	if (result == NULL)
		throw(MAL, "rdf.RDFpartialjoin", GDK_EXCEPTION);

	*retid = result->batCacheid; 
	BBPkeepref(*retid); 

	return MAL_SUCCEED; 
}
*/

str
RDFpartialjoin(bat *retid, bat *lid, bat *rid, bat *inputid){
	BAT *left, *right, *result1, *result2, *result, *input;  
	BATiter resulti,inputi;
	BUN	p,q; 
	oid	*rbt; 
	oid	*ibt; 
	

	if ((left = BATdescriptor(*lid)) == NULL) {
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}
	if ((right = BATdescriptor(*rid)) == NULL) {
		BBPreleaseref(left->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	if ((input = BATdescriptor(*inputid)) == NULL) {
		BBPreleaseref(left->batCacheid);
		BBPreleaseref(right->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	BATsubouterjoin(&result1, &result2, input, left, NULL, NULL, 0, BUN_NONE); 
	
	result = BATproject(result2, right); 
	result->T->nil = 0; 
	

	BBPreleaseref(left->batCacheid);
	BBPreleaseref(right->batCacheid);

	resulti = bat_iterator(result);
	inputi = bat_iterator(input);

	BATloop(result, p, q){
		rbt = (oid *) BUNtloc(resulti, p); 
		if (*rbt == oid_nil){
			ibt = (oid *) BUNtail(inputi, p); 
			*rbt = *ibt; 
		}
	}

	BBPreleaseref(input->batCacheid);
	BBPreclaim(result1);
	BBPreclaim(result2);

	//BATprint(result); 
	if (result == NULL)
		throw(MAL, "rdf.RDFpartialjoin", GDK_EXCEPTION);

	*retid = result->batCacheid; 
	BBPkeepref(*retid); 

	return MAL_SUCCEED; 
}

str RDFtriplesubsort(BAT **sbat, BAT **pbat, BAT **obat){

	BAT *o1,*o2,*o3;
	BAT *g1,*g2,*g3;
	BAT *S = NULL, *P = NULL, *O = NULL;

	S = *sbat;
	P = *pbat;
	O = *obat;
	/* order SPO/SOP */
	if (BATsubsort(sbat, &o1, &g1, S, NULL, NULL, 0, 0) == GDK_FAIL){
		if (S != NULL) BBPreclaim(S);
		throw(RDF, "rdf.triplesubsort", "Fail in sorting for S");
	}

	if (BATsubsort(pbat, &o2, &g2, P, o1, g1, 0, 0) == GDK_FAIL){
		BBPreclaim(S);
		if (P != NULL) BBPreclaim(P);
		throw(RDF, "rdf.triplesubsort", "Fail in sub-sorting for P");
	}
	if (BATsubsort(obat, &o3, &g3, O, o2, g2, 0, 0) == GDK_FAIL){
		BBPreclaim(S);
		BBPreclaim(P);
		if (O != NULL) BBPreclaim(O);
		throw(RDF, "rdf.triplesubsort", "Fail in sub-sorting for O");
	}	

	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);

	return MAL_SUCCEED; 
}

/*
 * Sort left bat and re-order right bat according to the lef bat
 * */
str RDFbisubsort(BAT **lbat, BAT **rbat){

	BAT *o1,*o2;
	BAT *g1,*g2;
	BAT *L = NULL, *R = NULL;

	L = *lbat;
	R = *rbat;
	if (BATsubsort(lbat, &o1, &g1, L, NULL, NULL, 0, 0) == GDK_FAIL){
		if (L != NULL) BBPreclaim(L);
		throw(RDF, "rdf.triplesubsort", "Fail in sorting for L");
	}

	if (BATsubsort(rbat, &o2, &g2, R, o1, g1, 0, 0) == GDK_FAIL){
		BBPreclaim(L);
		if (R != NULL) BBPreclaim(R);
		throw(RDF, "rdf.triplesubsort", "Fail in sub-sorting for R");
	}

	BBPunfix(o1->batCacheid);
	BBPunfix(g1->batCacheid);
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);

	return MAL_SUCCEED; 
}

str
TKNZRrdf2str(bat *res, const bat *bid, const bat *map)
{
	BAT *r, *b, *m;
	BATiter bi, mi;
	BUN p, q;
	str s = NULL;

	b = BATdescriptor(*bid);
	if (b == NULL) {
		throw(MAL, "rdf.rdf2str", RUNTIME_OBJECT_MISSING " null bat b");
	}
	m = BATdescriptor(*map);
	if (m == NULL) {
		BBPunfix(*bid);
		throw(MAL, "rdf.rdf2str", RUNTIME_OBJECT_MISSING "null bat m");
	}
	if (!BAThdense(b)) {
		BBPunfix(*bid);
		BBPunfix(*map);
		throw(MAL, "rdf.rdf2str", SEMANTIC_TYPE_ERROR " semantic error");
	}
	r = BATnew(TYPE_void, TYPE_str, BATcount(b), TRANSIENT);
	if (r == NULL) {
		BBPunfix(*bid);
		BBPunfix(*map);
		throw(MAL, "rdf.rdf2str", RUNTIME_OBJECT_MISSING "null bat r");
	}
	*res = r->batCacheid;
	BATseqbase(r, b->hseqbase);
	bi = bat_iterator(b);
	mi = bat_iterator(m);

	BATloop(b, p, q)
	{
		oid id = *(oid *) BUNtloc(bi, p);
		if (id >= RDF_MIN_LITERAL) {
			BUN pos = BUNfirst(m) + (id - RDF_MIN_LITERAL);
			if (pos < BUNfirst(m) || pos >= BUNlast(m)) {
				BBPunfix(*bid);
				BBPunfix(*map);
				BBPunfix(*res);
				throw(MAL, "rdf.rdf2str", OPERATION_FAILED " illegal oid");
			}
			s = (str) BUNtail(mi, pos);
		} else {
			str ret = takeOid(id, &s);
			if (ret != MAL_SUCCEED) {
				BBPunfix(*bid);
				BBPunfix(*map);
				BBPunfix(*res);
				return ret;
			}
		}
		BUNappend(r, s, FALSE);
	}
	BBPunfix(*bid);
	BBPunfix(*map);
	BBPkeepref(*res);
	return MAL_SUCCEED;
}
