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

static void putCStoHash(map_t csmap, int* buff, int num, oid *csoid){
	oid 	*getCSoid; 
	oid	*putCSoid; 
	int 	err; 
	int* 	cs; 
	int 	freq; 

	cs = (int*) malloc(sizeof(int) * num);
	if (cs==NULL){
		printf("Malloc failed. at %d", num);
		exit(-1); 
	}

	copyIntSet(cs, buff, num); 
	if (hashmap_get(csmap, cs, num,(void**)(&getCSoid),1, &freq) != MAP_OK){
		putCSoid = malloc(sizeof(oid)); 
		*putCSoid = *csoid; 

		err = hashmap_put(csmap, cs, num, putCSoid); 	
		assert(err == MAP_OK); 
				
		//printf("Put CS %d into hashmap \n", (int) *putCSoid);

		(*csoid)++; 
	}
	else{
		//printf("The key %d exists in the hashmap with freq %d \n", (int) *getCSoid, freq);
		free(cs); 

	}
}


static void putPtoHash(map_t pmap, int value, oid *poid){
	oid 	*getPoid; 
	oid	*putPoid; 
	int 	err; 
	int* 	pkey; 
	int 	freq; 

	pkey = (int*) malloc(sizeof(int));

	*pkey = value; 

	if (hashmap_get(pmap, pkey, 1,(void**)(&getPoid),1, &freq) != MAP_OK){
		putPoid = malloc(sizeof(oid)); 
		*putPoid = *poid; 

		err = hashmap_put(pmap, pkey, 1, putPoid); 	
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

static void getStatisticCSsBySize(map_t csmap, int maximumNumP){

	int* statCS; 
	int i; 

	statCS = (int *) malloc(sizeof(int) * (maximumNumP + 1)); 
	
	for (i = 0; i <= maximumNumP; i++) statCS[i] = 0;

	hashmap_statistic_groupcs_by_size(csmap, statCS); 

	/* Print the result */
	
	printf(" --- Number of CS per size (Max = %d)--- \n", maximumNumP);
	for (i = 1; i <= maximumNumP; i++){
		printf("%d  :  %d \n", i, statCS[i]); 
	} 

	free(statCS); 
}


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
	
	if (isWriteToFile  == 0){
		printf(" --- Number of CS per support (Max = %d)--- \n", maxSupport);
		for (i = 1; i <= maxSupport; i++){
			printf("%d  :  %d \n", i, statCS[i]); 
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
RDFextractCS(int *ret, bat *sbatid, bat *pbatid){
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
	int 	INIT_PROPERTY_NUM = 50000; 
	int 	maxNumProp = 0; 

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
	numP = 0;
	curP = 0; 

	BATloop(sbat, p, q){
		bt = (oid *) BUNtloc(si, p);		
		if (*bt != curS){
			if (p != 0){	/* Not the first S */
				putCStoHash(csMap, buff, numP, &CSoid); 
				
				if (numP > maxNumProp) 
					maxNumProp = numP; 
					

			}
			curS = *bt; 
			curP = 0;
			numP = 0;
		}
				
		pbt = (oid *) BUNtloc(pi, p); 

		if (numP > INIT_PROPERTY_NUM){
			printf("# of properties %d is greater than INIT_PROPERTY_NUM at CS %d property %d \n", numP, (int)CSoid, (int)*pbt);
			exit(-1);
		}
		
		if (curP != *pbt){
			buff[numP] = *pbt; 
			numP++; 
			curP = *pbt; 
		}
		//printf("Travel sbat at %d  value: %d , for pbat: %d \n", (int) p, (int) *bt, (int) *pbt);
	}
	
	/*put the last CS */
	putCStoHash(csMap, buff, numP, &CSoid); 

	if (numP > maxNumProp) 
		maxNumProp = numP; 
					
	/*get the statistic */
	getTopFreqCSs(csMap,20);

	getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csMap, 5000, 1, 0);

	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	free (buff); 
	hashmap_free(csMap);

	*ret = 1; 
	return MAL_SUCCEED; 
}


/* Extract Properties and their supports from PSO table */
str
RDFextractPfromPSO(int *ret, bat *sbatid, bat *pbatid){
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
		if (*bt != curP){
			if (p != 0){	/* Not the first S */
				putPtoHash(pMap, *bt, &Poid); 
			}
			curP = *bt; 
			curS = 0;
		}

		sbt = (oid *) BUNtloc(si, p); 

		if (curS != *sbt){
			supportP++; 
			curS = *sbt; 
		}
	}
	
	/*put the last P */
	putPtoHash(pMap, *bt, &Poid); 


	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	hashmap_free(pMap);

	*ret = 1; 
	return MAL_SUCCEED; 
}
