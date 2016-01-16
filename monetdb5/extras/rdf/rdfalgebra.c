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
 * Copyright August 2008-2015 MonetDB B.V.
 * All Rights Reserved.
 */

/* This contains algebra functions used for RDF store only */

#include "monetdb_config.h"
#include "rdf.h"
#include "rdfminheap.h"
#include "algebra.h"
#include "tokenizer.h"

str
RDFleftfetchjoin_sorted(bat *result, const bat *lid, const bat *rid)
{
	BAT *left, *right, *bn = NULL;

	if ((left = BATdescriptor(*lid)) == NULL) {
		throw(MAL, "rdf.leftfetchjoin_sorted", RUNTIME_OBJECT_MISSING);
	}
	if ((right = BATdescriptor(*rid)) == NULL) {
		BBPunfix(left->batCacheid);
		throw(MAL, "rdf.leftfetchjoin_sorted", RUNTIME_OBJECT_MISSING);
	}
	bn = BATproject(left, right);
	BBPunfix(left->batCacheid);
	BBPunfix(right->batCacheid);
	if (bn == NULL)
		throw(MAL, "rdf.leftfetchjoin_sorted", GDK_EXCEPTION);

	bn->tsorted = TRUE;

	if (!(bn->batDirty & 2))
		BATsetaccess(bn, BAT_READ);
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
		BBPunfix(left->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	if ((input = BATdescriptor(*inputid)) == NULL) {
		BBPunfix(left->batCacheid);
		BBPunfix(right->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	map = VIEWcreate(BATmirror(left), right);

	BBPunfix(left->batCacheid);
	BBPunfix(right->batCacheid);

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

	BBPunfix(input->batCacheid);

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
		BBPunfix(left->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	if ((input = BATdescriptor(*inputid)) == NULL) {
		BBPunfix(left->batCacheid);
		BBPunfix(right->batCacheid);
		throw(MAL, "rdf.RDFpartialjoin", RUNTIME_OBJECT_MISSING);
	}

	BATouterjoin(&result1, &result2, input, left, NULL, NULL, 0, BUN_NONE); 
	
	result = BATproject(result2, right); 
	result->T->nil = 0; 
	

	BBPunfix(left->batCacheid);
	BBPunfix(right->batCacheid);

	resulti = bat_iterator(result);
	inputi = bat_iterator(input);

	BATloop(result, p, q){
		rbt = (oid *) BUNtloc(resulti, p); 
		if (*rbt == oid_nil){
			ibt = (oid *) BUNtail(inputi, p); 
			*rbt = *ibt; 
		}
	}

	BBPunfix(input->batCacheid);
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
	if (BATsort(sbat, &o1, &g1, S, NULL, NULL, 0, 0) == GDK_FAIL){
		if (S != NULL) BBPreclaim(S);
		throw(RDF, "rdf.triplesubsort", "Fail in sorting for S");
	}

	if (BATsort(pbat, &o2, &g2, P, o1, g1, 0, 0) == GDK_FAIL){
		BBPreclaim(S);
		if (P != NULL) BBPreclaim(P);
		throw(RDF, "rdf.triplesubsort", "Fail in sub-sorting for P");
	}
	if (BATsort(obat, &o3, &g3, O, o2, g2, 0, 0) == GDK_FAIL){
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

static
void append_cand_to_bats(BAT *r_sbat, BAT **r_obats, oid **tmpCand, int *numCand, oid *cand, int cur_p, int np, oid curS){
	int i, j; 
	if (numCand[cur_p] == 0){
		cand[cur_p] = oid_nil;

		if (cur_p < np - 1)
			append_cand_to_bats(r_sbat, r_obats, tmpCand, numCand, cand, cur_p + 1, np, curS); 
		else{
			for (j = 0; j < np; j++){
				BUNappend(r_obats[j], &(cand[j]), TRUE); 
			}
			BUNappend(r_sbat, &curS, TRUE);
		}
	} else {
		for (i = 0; i < numCand[cur_p]; i++){
			cand[cur_p] = tmpCand[cur_p][i]; 
			
			if (cur_p < np - 1)
				append_cand_to_bats(r_sbat, r_obats, tmpCand, numCand, cand, cur_p + 1, np, curS); 
			else{
				for (j = 0; j < np; j++){
					BUNappend(r_obats[j], &(cand[j]), TRUE); 
				}
				BUNappend(r_sbat, &curS, TRUE);
			}
		}		
	}

}
/* This function RDFmultiway_merge_outerjoins()
 * is used to create full outer join from multiple 
 * Input: 
 * - np: Number of properties --> == number of obats, number of sbats
 * - Set of pair of bats corresponding a slice of PSO with certain P value
 * [bat_s1, bat_o1], [bat_s2, bat_o2],....,[bat_sn, bat_on]
 * - All bat_si are sorted 
 * Output:
 * bat_s, bat_o1_new, bat_o2_new, ..., bat_on_new
 * Where bat_s is union of all bat_s1, ..., bat_sn
 * 
 * Use a minheap to merge multiple list
 * */
str RDFmultiway_merge_outerjoins(int np, BAT **sbats, BAT **obats, BAT **r_sbat, BAT **r_obats){
	BUN estimate = 0; 
	int i = 0; 
	MinHeap *hp;
	MinHeapNode *harr;
	oid **sbatCursors, **obatCursors; 
	oid lastS = BUN_NONE; 
	oid tmpO; 

	// This aray is used for storing the candidate outputs in case 
	// that there are multi-valued prop in pso. 
	// In that case, a recursive program need to run to generate
	// all the combinations of the output candidates
	int maxNumExcept = 10000; 	//TODO: This need to be verified
	oid **tmpCand;
	int *numCand; 
	oid *cand; 

	tmpCand = (oid **) malloc(sizeof(oid *) * np); 
	numCand = (int*) malloc(sizeof(int) * np); 
	cand = (oid *) malloc(sizeof(oid) * np); 

	for (i = 0; i < np; i++){
		if (obats[i]) estimate += BATcount(obats[i]); 
		tmpCand[i] = (oid *) malloc(sizeof(oid) * maxNumExcept); 
		numCand[i] = 0; 
		cand[i] = oid_nil; 
	}

	sbatCursors = (oid **) malloc(sizeof(oid*) * np); 
	obatCursors = (oid **) malloc(sizeof(oid*) * np); 

	*r_sbat = BATnew(TYPE_void, TYPE_oid, estimate, TRANSIENT); 
	
	for (i = 0; i < np; i++){
		r_obats[i] = BATnew(TYPE_void, TYPE_oid, estimate, TRANSIENT); 
	
		//assert(BATcount(sbats[i]) > 0); 
		//
		//Keep the cursor to the first element of each input sbats
		if (sbats[i]) sbatCursors[i] = (oid *) Tloc(sbats[i], BUNfirst(sbats[i]));
		else sbatCursors[i] = NULL; 

		if (obats[i]) obatCursors[i] = (oid *) Tloc(obats[i], BUNfirst(obats[i]));	
		else obatCursors[i] = NULL; 
	}

	//Create a min heap with np heap nodes.  Every heap node
	//has first element of an array (pointing to the first element of each sbat)
	harr = (MinHeapNode*)malloc(sizeof(MinHeapNode) * np);
	for (i = 0; i < np; i++){
		if (sbats[i] && BATcount(sbats[i]) != 0){
			harr[i].element =  sbatCursors[i][0]; //Store the first element
			harr[i].i = i; //index of array
			harr[i].j = 1; //Index of next element to be stored from array
		} else {
			harr[i].element =  GDK_oid_max; //Store the INT_MAX in case of empty BATs
			harr[i].i = i; //index of array
			harr[i].j = 1; //Index of next element to be stored from array
		}

	}

	hp = (MinHeap *) malloc(sizeof(MinHeap)); 
	initMinHeap(hp, harr, np);  //Create the heap

	//Now one by one get the minimum element from min
	//heap and replace it with next element of its array
	while (1){
		//Get the minimum element and store it in output
		MinHeapNode root = getMin(hp);
		if (root.element == GDK_oid_max) break; 
		
		if (lastS != root.element){		//New S
			
			//Go through all output o_bat to add Null value
			//if they do not value for the last S

			if (lastS != BUN_NONE) append_cand_to_bats(*r_sbat, r_obats, tmpCand, numCand, cand, 0, np, lastS);

			for (i = 0; i < np; i++){
				numCand[i] = 0;
			}
			//Append the obat corresonding to this root node 
			tmpO = obatCursors[root.i][root.j - 1]; 
			tmpCand[root.i][numCand[root.i]] = tmpO;
			numCand[root.i]++;

			lastS = root.element; 
		}
		else{
			//Get element from the corresponding o
			//Add to the output o
			tmpO = obatCursors[root.i][root.j - 1];
			//BUNappend(r_obats[root.i], &tmpO, TRUE);
			tmpCand[root.i][numCand[root.i]] = tmpO;  
			numCand[root.i]++; 	
		}

		//Find the next elelement that will replace current
		//root of heap. The next element belongs to same
		//array as the current root.
		if (root.j < (int) BATcount(sbats[root.i]))
		{
			root.element = sbatCursors[root.i][root.j];
			root.j += 1;
		}
		//If root was the last element of its array
		else root.element =  GDK_oid_max; 

		//printf("Update root.element = " BUNFMT, root.element); 		
		//Replace root with next element of array
		replaceMin(hp, root);
	}
	
	if (lastS != BUN_NONE) append_cand_to_bats(*r_sbat, r_obats, tmpCand, numCand, cand, 0, np, lastS);

	for (i = 0; i < np; i++){
		free(tmpCand[i]);
	}

	free(tmpCand); 
	free(numCand); 
	free(cand); 
	free(hp); 
	free(harr); 
	free(sbatCursors); 
	free(obatCursors); 
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
	if (BATsort(lbat, &o1, &g1, L, NULL, NULL, 0, 0) == GDK_FAIL){
		if (L != NULL) BBPreclaim(L);
		throw(RDF, "rdf.triplesubsort", "Fail in sorting for L");
	}

	if (BATsort(rbat, &o2, &g2, R, o1, g1, 0, 0) == GDK_FAIL){
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
				throw(MAL, "rdf.rdf2str", OPERATION_FAILED " illegal oid (rdfalgebra.c)");
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
