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

static void copyOidSet(oid* dest, oid* orig, int len){
	int i; 
	for (i = 0; i < len; i++){
		dest[i] = orig[i];
	}
}

static void printArray(oid* inputArr, int num){
	int i; 
	printf("Print array \n");
	for (i = 0; i < num; i++){
		printf("%d:  " BUNFMT "\n",i, inputArr[i]);
	}
	printf("End of array \n ");
}


static void initArray(oid* inputArr, int num, oid defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
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
CS* creatCS(oid subId, int numP, oid* buff){
	CS *cs = malloc(sizeof(CS)); 
	cs->lstProp =  (oid*) malloc(sizeof(oid) * numP);
	
	if (cs->lstProp == NULL){
		printf("Malloc failed. at %d", numP);
		exit(-1); 
	}

	copyOidSet(cs->lstProp, buff, numP); 
	cs->subIdx = subId;
	cs->numProp = numP; 
	cs->numAllocation = numP; 
	cs->isSubset = 0; /*By default, this CS is not known to be a subset of any other CS*/
	return cs; 
}


/*
 * Hashing function for a set of values
 * Rely on djb2 http://www.cse.yorku.ca/~oz/hash.html
 *
 */
static oid RDF_hash_oidlist(oid* key, int num){
	//unsigned int hashCode = 5381u; 
	oid  hashCode = 5381u;
	int i; 

	for (i = 0; i < num; i++){
		hashCode = ((hashCode << 5) + hashCode) + key[i];
	}
	
	// return 0x7fffffff & hashCode 
	return hashCode;
}

static 
void appendArrayToBat(BAT *b, BUN* inArray, int num){
	//int i; 
	BUN r = BUNlast(b);
	if (r + num < b->batCapacity){
		BATextend(b, smallbatsz); 
	}
	//for (i = 0; i < num; i++){
	memcpy(Tloc(b, BUNlast(b)), inArray, sizeof(BUN) * num); 
	//}
	BATsetcount(b, (BUN) (b->batCount + num)); 
	
}

static 
void checkCSduplication(BAT* pOffsetBat, BAT* fullPBat, BUN pos, oid* key, int numK){
	oid *offset; 
	oid *offset2; 
	int numP; 
	int i; 
	BUN *existvalue; 

	offset = (oid *) Tloc(pOffsetBat, pos); 
	if ((pos + 1) < pOffsetBat->batCount){
		offset2 = (oid *)Tloc(pOffsetBat, pos + 1);
		numP = *offset2 - *offset;
	}
	else{
		offset2 = malloc(sizeof(oid)); 
		*offset2 = BUNlast(fullPBat); 
		numP = *offset2 - *offset;
		free(offset2); 
	}


	// Check each value
	if (numK != numP) {
		printf("No duplication \n");
		return; 
	}
	else{
		existvalue = (oid *)Tloc(fullPBat, *offset);	
		for (i = 0; i < numP; i++){
			//if (key[i] != (int)*existvalue++) {
			if (key[i] != existvalue[i]) {
				printf("No duplication \n");
				return;
			}	
		}
	}
	
	printf("There is duplication \n");
	return;
} 
/*
 * Put a CS to the hashmap. 
 * While putting CS to the hashmap, update the support (frequency) value 
 * for an existing CS, and check whether it becomes a frequent CS or not. 
 * If yes, add that frequent CS to the freqCSset. 
 *
 * */
static 
oid putaCStoHash(BAT* hsKeyBat, BAT* pOffsetBat, BAT* fullPBat, oid subjId, oid* key, int num, 
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset **freqCSset){
	BUN 	csKey; 
	int 	freq = 0; 
	CS	*freqCS; 
	BUN	bun; 
	BUN	offset; 
	oid	csId; 		/* Id of the characteristic set */

	csKey = RDF_hash_oidlist(key, num);
	bun = BUNfnd(BATmirror(hsKeyBat),(ptr) &csKey);
	if (bun == BUN_NONE) {
		if (hsKeyBat->T->hash && BATcount(hsKeyBat) > 4 * hsKeyBat->T->hash->mask) {
			HASHdestroy(hsKeyBat);
			BAThash(BATmirror(hsKeyBat), 2*BATcount(hsKeyBat));
		}
		hsKeyBat = BUNappend(hsKeyBat, (ptr) &csKey, TRUE);

		
		csId = *csoid;
		(*csoid)++;
		
		offset = BUNlast(fullPBat);
		/* Add list of p to fullPBat and pOffsetBat*/
		BUNappend(pOffsetBat, &offset , TRUE);
		appendArrayToBat(fullPBat, key, num);

	}
	else{
		printf("This CS exists \n");	
		csId = bun; 
		/* Check whether it is really an duplication (same hashvalue but different list of */
		checkCSduplication(pOffsetBat, fullPBat, bun, key, num );

		if (isStoreFreqCS == 1){	/* Store the frequent CS to the CSset*/
			//printf("FreqCS: Support = %d, Threshold %d  \n ", freq, freqThreshold);
			if (freq == freqThreshold){
				freqCS = creatCS(subjId, num, key);		
				addCStoSet(*freqCSset, *freqCS);
			}
		}
	}

	return csId;
}

/* Return 1 if sorted arr2[] is a subset of sorted arr1[] 
 * arr1 has m members, arr2 has n members
 * */

static int isSubset(oid* arr1, oid* arr2, int m, int n)
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

/*
static void getTopFreqCSs(map_t csmap, int threshold){
	int count;
	hashmap_map* m; 
	count = hashmap_iterate_threshold(csmap, threshold); 
	m = (hashmap_map *) csmap;
	printf("Threshold: %d \n ", threshold);
	printf("Number of frequent CSs %d / Number of CSs %d (Table size: %d) \n" , count, m->size, m->table_size);

	return;

}
*/

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

/*
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

	// Output the result 
	
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
*/

/*
 * Get the refer CS 
 * Input: oid of a URI object 
 * Return the id of the CS
 * */

/*
static 
str getReferCS(BAT *sbat, BAT *pbat, oid *obt){

	// For detecting foreign key relationships 
	BAT	*tmpbat;	// Get the result of searching objectURI from sbat 
	BATiter	ti; 
	oid	*tbt;
	BUN	pt, qt; 
	oid	*s_t, *p_t; 	

	// BATsubselect(inputbat, <dont know yet>, lowValue, Highvalue, isIncludeLowValue, isIncludeHigh, <anti> 
	//printf("Checking for object " BUNFMT "\n", *obt);
	tmpbat = BATsubselect(sbat, NULL, obt, obt, 1, 1, 0); 
	// tmpbat tail contain head oid of sbat for matching elements 
	if (tmpbat != NULL){
		//printf("Matching: " BUNFMT "\n", BATcount(tmpbat));
		BATprint(tmpbat); 
						
		if (BATcount(tmpbat) > 0){
			ti = bat_iterator(tmpbat);
		        BATloop(tmpbat, pt, qt){
		                tbt = (oid *) BUNtail(ti, pt);
				s_t = (oid *) Tloc(sbat, *tbt);
				p_t = (oid *) Tloc(pbat, *tbt); 
				// Check which CS is referred 

			}
		}
	}
	else
		throw(MAL, "rdf.RDFextractCSwithTypes", "Null Bat returned for BATsubselect");			

	// temporarily use 
	if (tmpbat)
		BBPunfix(tmpbat->batCacheid);

	return MAL_SUCCEED;
}
*/

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
	oid*	buff; 	 
	int 	INIT_PROPERTY_NUM = 5000; 
	int 	maxNumProp = 0; 
	CSset	*freqCSset; 	/* Set of frequent CSs */
	oid 	objType;

	BAT	*hsKeyBat; 
	//BAT	*hsValueBat;
	BAT	*pOffsetBat; 	/* BAT storing the offset for set of properties, refer to fullPBat */
	BAT	*fullPBat;  	/* Stores all set of properties */

	oid	*subjCSMap; 	/* Store the correspoinding CS Id for each subject */
	BUN	*maxSoid; 	
	oid 	returnCSid; 
	
	buff = (oid *) malloc (sizeof(oid) * INIT_PROPERTY_NUM);

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

	maxSoid = (BUN *) Tloc(sbat, BUNlast(sbat) - 1);

	subjCSMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	initArray(subjCSMap, (*maxSoid), GDK_oid_max);
	
	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat);

	hsKeyBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	//hsValueBat = BATnew(TYPE_void, TYPE_int, smallbatsz);
	pOffsetBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	fullPBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);

	if (hsKeyBat == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", "Error in BAT creation");
	}
	BATseqbase(hsKeyBat, 0);

	freqCSset = initCSset();

	numP = 0;
	curP = 0; 
	curS = 0; 

	printf("freqThreshold = %d \n", *freqThreshold);	
	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				returnCSid = putaCStoHash(hsKeyBat, pOffsetBat, fullPBat, curS, buff, numP, &CSoid, 1, *freqThreshold, &freqCSset); 

				subjCSMap[curS] = returnCSid; 				

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
			//getReferCS(sbat, pbat, obt);		
		}
	}
	
	/*put the last CS */
	returnCSid = putaCStoHash(hsKeyBat, pOffsetBat, fullPBat, curS, buff, numP, &CSoid, 1, *freqThreshold, &freqCSset ); 
	
	subjCSMap[curS] = returnCSid; 				


	if (numP > maxNumProp) 
		maxNumProp = numP; 
		
	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	/*get the statistic */

	//getTopFreqCSs(csMap,*freqThreshold);

	getMaximumFreqCSs(freqCSset); 

	//getStatisticCSsBySize(csMap,maxNumProp); 

	//getStatisticCSsBySupports(csMap, 5000, 1, 0);

	printf("pOffsetBat ------- ");
	BATprint(pOffsetBat);

	printf("fullBat ------- ");
	BATprint(fullPBat);

	printArray(subjCSMap,(int) *maxSoid); 

	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	BBPreclaim(hsKeyBat); 
	BBPreclaim(pOffsetBat); 
	BBPreclaim(fullPBat); 
	
	free (buff); 
	free (subjCSMap); 

	freeCSset(freqCSset); 


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
