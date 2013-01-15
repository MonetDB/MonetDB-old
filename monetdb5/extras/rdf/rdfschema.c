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
 * Copyright August 2008-2012 MonetDB B.V.
 * All Rights Reserved.
 */

/* This contains algebra functions used for RDF store only */

#include "monetdb_config.h"
#include "rdf.h"
#include "rdfschema.h"
#include "algebra.h"
#include <gdk.h>
#include <hashmap/hashmap.h>

str
RDFSchemaExplore(int *ret, str *tbname, str *clname)
{
	printf("Explore from table %s with colum %s \n", *tbname, *clname);
	*ret = 1; 
	return MAL_SUCCEED;
}

static void copyIntSet(int* dest, int* orig, int len){
	int i; 
	for (i = 0; i < len; i++){
		dest[i] = orig[i];
	}
}


static 
void addCStoSet(CSset *csSet, CS item)
{
	void *_tmp; 
	if(csSet->numCSadded == csSet->numAllocation) 
	{ 
		csSet->numAllocation += INIT_NUM_CS; 
		
		_tmp = realloc(csSet->items, (csSet->numAllocation * sizeof(CS)));
	
		if (!_tmp){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		}
		csSet->items = (CS*)_tmp;
	}
	csSet->items[csSet->numCSadded] = item;
	csSet->numCSadded++;
}

static 
void freeCSset(CSset *csSet){
	int i;
	for(i = 0; i < csSet->numCSadded; i ++){
		free(csSet->items[i].lstProp);
	}
	free(csSet->items);
	free(csSet);
}

static 
CSset* initCSset(void){
	CSset *csSet = malloc(sizeof(CSset)); 
	csSet->items = malloc(sizeof(CS) * INIT_NUM_CS); 
	csSet->numAllocation = INIT_NUM_CS;
	csSet->numCSadded = 0;

	return csSet;
}

/*
static 
void freeCS(CS *cs){
	free(cs->lstProp);
	free(cs);
}
*/

static 
CS* creatCS(int subId, int numP, int* buff){
	CS *cs = malloc(sizeof(CS)); 
	cs->lstProp =  (int*) malloc(sizeof(int) * numP);
	
	if (cs->lstProp == NULL){
		printf("Malloc failed. at %d", numP);
		exit(-1); 
	}

	copyIntSet(cs->lstProp, buff, numP); 
	cs->subIdx = subId;
	cs->numProp = numP; 
	cs->numAllocation = numP; 
	cs->isSubset = 0; /*By default, this CS is not known to be a subset of any other CS*/
	return cs; 
}

/*
 * Put a CS to the hashmap. 
 * While putting CS to the hashmap, update the support (frequency) value 
 * for an existing CS, and check whether it becomes a frequent CS or not. 
 * If yes, add that frequent CS to the freqCSset. 
 *
 * */
static void putaCStoHash(map_t csmap, int* key, int num, oid *csoid, char isStoreFreqCS, int freqThreshold, CSset **freqCSset){
	oid 	*getCSoid; 
	oid	*putCSoid; 
	int 	err; 
	int* 	csKey; 
	int 	freq = 0; 
	CS	*freqCS; 

	csKey = (int*) malloc(sizeof(int) * num);
	if (csKey==NULL){
		printf("Malloc failed. at %d", num);
		exit(-1); 
	}

	copyIntSet(csKey, key, num); 
	if (hashmap_get(csmap, csKey, num,(void**)(&getCSoid),1, &freq) != MAP_OK){
		putCSoid = malloc(sizeof(oid)); 
		*putCSoid = *csoid; 

		err = hashmap_put(csmap, csKey, num, 1,  putCSoid); 	
		assert(err == MAP_OK); 

		(*csoid)++; 
	}
	else{
		if (isStoreFreqCS == 1){	/* Store the frequent CS to the CSset*/
			//printf("FreqCS: Support = %d, Threshold %d  \n ", freq, freqThreshold);
			if (freq == freqThreshold){
				freqCS = creatCS(*getCSoid, num, key);		
				addCStoSet(*freqCSset, *freqCS);
			}
		}
		free(csKey); 
	}

}

/* Return 1 if sorted arr2[] is a subset of sorted arr1[] 
 * arr1 has m members, arr2 has n members
 * */

static int isSubset(int* arr1, int* arr2, int m, int n)
{
	int i = 0, j = 0;
	 
	if(m < n)
		return 0;
		 
	while( i < n && j < m )
	{
		if( arr1[j] < arr2[i] )
			j++;
		else if( arr1[j] == arr2[i] )
		{
			j++;
			i++;
		}
		else if( arr1[j] > arr2[i] )
			return 0;
	}
		
	if( i < n )
		return 0;
	else
		return 1;
}

/*
static 
void printCS(CS cs){
	int i; 
	printf("CS %d: ", cs.subIdx);
	for (i = 0; i < cs.numProp; i++){
		printf(" %d  ", cs.lstProp[i]);
	}
	printf("\n");
}
*/

/*
 * Get the maximum frequent CSs from a CSset
 * Here maximum frequent CS is a CS that there exist no other CS which contains that CS
 * */
static 
void getMaximumFreqCSs(CSset *freqCSset){

	int numCS = freqCSset->numCSadded; 
	int i, j; 
	int numMaxCSs = 0;

	printf("Retrieving maximum frequent CSs: \n");

	for (i = 0; i < numCS; i++){
		if (freqCSset->items[i].isSubset == 1) continue;
		for (j = (i+1); j < numCS; j++){
			if (isSubset(freqCSset->items[i].lstProp, freqCSset->items[j].lstProp,  
					freqCSset->items[i].numProp,freqCSset->items[j].numProp) == 1) { 
				/* CSj is a subset of CSi */
				freqCSset->items[j].isSubset = 1; 
			}
			else if (isSubset(freqCSset->items[j].lstProp, freqCSset->items[i].lstProp,  
					freqCSset->items[j].numProp,freqCSset->items[i].numProp) == 1) { 
				/* CSj is a subset of CSi */
				freqCSset->items[i].isSubset = 1; 
				break; 
			}
			
		} 
		/* By the end, if this CS is not a subset of any other CS */
		if (freqCSset->items[i].isSubset == 0){
			numMaxCSs++;
			//printCS( freqCSset->items[i]); 
		}
	}
	printf("Number of maximum CSs: %d \n", numMaxCSs);
}




static void putPtoHash(map_t pmap, int key, oid *poid, int support){
	oid 	*getPoid; 
	oid	*putPoid; 
	int 	err; 
	int* 	pkey; 

	pkey = (int*) malloc(sizeof(int));

	*pkey = key; 

	if (hashmap_get_forP(pmap, pkey,(void**)(&getPoid)) != MAP_OK){
		putPoid = malloc(sizeof(oid)); 
		*putPoid = *poid; 

		err = hashmap_put_forP(pmap, pkey, 1, putPoid, support); 	
		assert(err == MAP_OK); 
				
		(*poid)++; 
	}
	else{
		free(pkey); 
	}
}

static void getTopFreqCSs(map_t csmap, int threshold){
	int count;
	hashmap_map* m; 
	count = hashmap_iterate_threshold(csmap, threshold); 
	m = (hashmap_map *) csmap;
	printf("Threshold: %d \n ", threshold);
	printf("Number of frequent CSs %d / Number of CSs %d (Table size: %d) \n" , count, m->size, m->table_size);

	return;

}

/*
static void getStatisticCSsBySize(map_t csmap, int maximumNumP){

	int* statCS; 
	int i; 

	statCS = (int *) malloc(sizeof(int) * (maximumNumP + 1)); 
	
	for (i = 0; i <= maximumNumP; i++) statCS[i] = 0;

	hashmap_statistic_groupcs_by_size(csmap, statCS); 

	// Print the result
	
	printf(" --- Number of CS per size (Max = %d)--- \n", maximumNumP);
	for (i = 1; i <= maximumNumP; i++){
		printf("%d : %d \n", i, statCS[i]); 
	} 

	free(statCS); 
}
*/

static void getStatisticCSsBySupports(map_t csmap, int maxSupport, char isWriteToFile, char isCummulative){

	int* statCS; 
	int i; 
	FILE *fout; 

	statCS = (int *) malloc(sizeof(int) * (maxSupport + 1)); 
	
	for (i = 0; i <= maxSupport; i++) statCS[i] = 0; 
	
	if (isCummulative == 1)
		hashmap_statistic_CSbysupport_cummulative(csmap, statCS, maxSupport); 
	else 
		hashmap_statistic_CSbysupport(csmap, statCS, maxSupport); 

	/* Output the result */
	
	if (isWriteToFile == 0){
		printf(" --- Number of CS per support (Max = %d)--- \n", maxSupport);
		for (i = 1; i <= maxSupport; i++){
			printf("%d : %d \n", i, statCS[i]); 
		} 
	}
	else {
		if (isCummulative == 1)
			fout = fopen("cummulativeNumCSbySupport.txt","wt"); 
		else 
			fout = fopen("numCSbySupport.txt","wt"); 

		fprintf(fout, " --- Number of CS per support (Max = %d)--- \n", maxSupport); 
		
		for (i = 1; i <= maxSupport; i++){
			fprintf(fout, "%d\t:\t%d \n", i, statCS[i]); 
		} 
		fclose(fout); 
		
	}

	free(statCS); 
}

/* Extract CS from SPO triples table */
str
RDFextractCS(int *ret, bat *sbatid, bat *pbatid, int *freqThreshold){
	BUN 	p, q; 
	BAT 	*sbat = NULL, *pbat = NULL; 
	BATiter si, pi; 	/*iterator for BAT of s,p columns in spo table */
	oid 	*bt, *pbt; 
	oid 	curS; 		/* current Subject oid */
	oid 	curP; 		/* current Property oid */
	oid 	CSoid = 0; 	/* Characteristic set oid */
	int 	numP; 		/* Number of properties for current S */
	map_t 	csMap; 		
	int*	buff; 	 
	int 	INIT_PROPERTY_NUM = 5000; 
	int 	maxNumProp = 0; 
	CSset	*freqCSset; 	/* Set of frequent CSs */


	buff = (int *) malloc (sizeof(int) * INIT_PROPERTY_NUM);
	
	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}
	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}
	
	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 

	/* Init a hashmap */
	csMap = hashmap_new(); 
	freqCSset = initCSset();

	numP = 0;
	curP = 0; 

	printf("freqThreshold = %d \n", *freqThreshold);	
	BATloop(sbat, p, q){
		bt = (oid *) BUNtloc(si, p);		
		if (*bt != curS){
			if (p != 0){	/* Not the first S */
				putaCStoHash(csMap, buff, numP, &CSoid, 1, *freqThreshold, &freqCSset); 
				
				if (numP > maxNumProp) 
					maxNumProp = numP; 
			}
			curS = *bt; 
			curP = 0;
			numP = 0;
		}
				
		pbt = (oid *) BUNtloc(pi, p); 

		if (numP > INIT_PROPERTY_NUM){
			throw(MAL, "rdf.RDFextractCS", "# of properties is greater than INIT_PROPERTY_NUM");
			exit(-1);
		}
		
		if (curP != *pbt){	/* Multi values property */		
			buff[numP] = *pbt; 
			numP++; 
			curP = *pbt; 
		}
		//printf("Travel sbat at %d value: %d , for pbat: %d \n", (int) p, (int) *bt, (int) *pbt);
	}
	
	/*put the last CS */
	putaCStoHash(csMap, buff, numP, &CSoid, 1, *freqThreshold, &freqCSset ); 

	if (numP > maxNumProp) 
		maxNumProp = numP; 
		
	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	/*get the statistic */

	getTopFreqCSs(csMap,*freqThreshold);

	getMaximumFreqCSs(freqCSset); 

	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csMap, 5000, 1, 0);

	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	free (buff); 

	freeCSset(freqCSset); 

	hashmap_free(csMap);

	*ret = 1; 
	return MAL_SUCCEED; 
}

/*
 * Get the refer CS 
 * Input: oid of a URI object 
 * Return the id of the CS
 * */
static 
str getReferCS(BAT *sbat, BAT *pbat, oid *obt){

	/* For detecting foreign key relationships */
	BAT	*tmpbat;	/* Get the result of searching objectURI from sbat */
	BATiter	ti; 
	oid	*tbt;
	BUN	pt, qt; 
	oid	*s_t, *p_t; 	
	//int	*tmpbuff;

	//tmpbuff = (int *) malloc (sizeof(int) * INIT_PROPERTY_NUM);

	/* BATsubselect(inputbat, <dont know yet>, lowValue, Highvalue, isIncludeLowValue, isIncludeHigh, <anti> */
	printf("Checking for object " BUNFMT "\n", *obt);
	tmpbat = BATsubselect(sbat, NULL, obt, obt, 1, 1, 0); 
	/* tmpbat tail contain head oid of sbat for matching elements */
	if (tmpbat != NULL){
		printf("Matching: " BUNFMT "\n", BATcount(tmpbat));
		BATprint(tmpbat); 
						
		if (BATcount(tmpbat) > 0){
			ti = bat_iterator(tmpbat);
		        BATloop(tmpbat, pt, qt){
		                tbt = (oid *) BUNtail(ti, pt);
				s_t = (oid *) Tloc(sbat, *tbt);
				p_t = (oid *) Tloc(pbat, *tbt); 
				printf("s_t: " BUNFMT "\n", (*s_t));
				printf("p_t: " BUNFMT "\n", (*p_t));
						/* Check which CS is referred */

			}
		}
	}
	else
		throw(MAL, "rdf.RDFextractCSwithTypes", "Null Bat returned for BATsubselect");			

			
			
	/* temporarily use */
	if (tmpbat)
		BBPunfix(tmpbat->batCacheid);

	return MAL_SUCCEED;
}

/* Extract CS from SPO triples table */
str
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, int *freqThreshold){
	BUN 	p, q; 
	BAT 	*sbat = NULL, *pbat = NULL, *obat = NULL; 
	BATiter si, pi, oi; 	/*iterator for BAT of s,p,o columns in spo table */
	oid 	*sbt, *pbt, *obt; 
	oid 	curS; 		/* current Subject oid */
	oid 	curP; 		/* current Property oid */
	oid 	CSoid = 0; 	/* Characteristic set oid */
	int 	numP; 		/* Number of properties for current S */
	map_t 	csMap; 		
	int*	buff; 	 
	int 	INIT_PROPERTY_NUM = 5000; 
	int 	maxNumProp = 0; 
	CSset	*freqCSset; 	/* Set of frequent CSs */
	oid 	objType;
	

	
	buff = (int *) malloc (sizeof(int) * INIT_PROPERTY_NUM);

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if (!(sbat->tsorted)){
		 throw(MAL, "rdf.RDFextractCSwithTypes", "sbat is not sorted");
	}

	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if ((obat = BATdescriptor(*obatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	
	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat);

	/* Init a hashmap */
	csMap = hashmap_new(); 
	freqCSset = initCSset();

	numP = 0;
	curP = 0; 

	printf("freqThreshold = %d \n", *freqThreshold);	
	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				putaCStoHash(csMap, buff, numP, &CSoid, 1, *freqThreshold, &freqCSset); 
				
				if (numP > maxNumProp) 
					maxNumProp = numP; 
			}
			curS = *sbt; 
			curP = 0;
			numP = 0;
		}
				
		pbt = (oid *) BUNtloc(pi, p); 

		if (numP > INIT_PROPERTY_NUM){
			throw(MAL, "rdf.RDFextractCS", "# of properties is greater than INIT_PROPERTY_NUM");
			exit(-1);
		}
		
		if (curP != *pbt){	/* Multi values property */		
			buff[numP] = *pbt; 
			numP++; 
			curP = *pbt; 
		}
		
		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = ((*obt) >> (sizeof(BUN)*8 - 3))  &  3 ;	/* Get two bits 63th, 62nd from object oid */
		
		//printf("object type: " BUNFMT "\n", objType); 

		/* Look at sbat*/
		if (objType == URI){

			getReferCS(sbat, pbat, obt);		
		}
	}
	
	/*put the last CS */
	putaCStoHash(csMap, buff, numP, &CSoid, 1, *freqThreshold, &freqCSset ); 

	if (numP > maxNumProp) 
		maxNumProp = numP; 
		
	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	/*get the statistic */

	getTopFreqCSs(csMap,*freqThreshold);

	getMaximumFreqCSs(freqCSset); 

	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csMap, 5000, 1, 0);

	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	free (buff); 

	freeCSset(freqCSset); 

	hashmap_free(csMap);

	*ret = 1; 
	return MAL_SUCCEED; 
}

/* Extract Properties and their supports from PSO table */
str
RDFextractPfromPSO(int *ret, bat *pbatid, bat *sbatid){
	BUN 	p, q; 
	BAT 	*sbat = NULL, *pbat = NULL; 
	BATiter si, pi; 	/*iterator for BAT of s,p columns in spo table */
	oid 	*bt, *sbt; 
	oid 	curS; 		/* current Subject oid */
	oid 	curP; 		/* current Property oid */
	map_t 	pMap; 		
	int 	supportP; 	/* Support value for P */
	oid 	Poid = 0; 	/* Characteristic set oid */

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}
	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}
	
	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 

	/* Init a hashmap */
	pMap = hashmap_new(); 
	curP = 0; 
	supportP = 0; 

	BATloop(pbat, p, q){
		bt = (oid *) BUNtloc(pi, p);
		//printf("bt: " BUNFMT "\n", *bt);
		//printf("p: " BUNFMT "\n", p);
		//printf("After: " BUNFMT "\n", *bun);
		if (*bt != curP){
			if (p != 0){	/* Not the first S */
				putPtoHash(pMap, curP, &Poid, supportP); 
				supportP = 0;
			}
			curP = *bt; 
		}

		sbt = (oid *) BUNtloc(si, p); 

		if (curS != *sbt){
			supportP++; 
			curS = *sbt; 
		}
	}
	
	/*put the last P */
	putPtoHash(pMap, *bt, &Poid, supportP); 
	
	//printf("Print all properties \n");
	//hashmap_print(pMap);

	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	hashmap_free(pMap);

	*ret = 1; 
	return MAL_SUCCEED; 
}
