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


static void getTopFreqCSs(map_t csmap, int threshold){
	int count;
	hashmap_map* m; 
	count = hashmap_iterate_threshold(csmap, threshold); 
	m = (hashmap_map *) csmap;
	printf("Threshold: %d \n ", threshold);
	printf("Number of frequent CSs %d / Number of CSs %d (Table size: %d) \n" , count, m->size, m->table_size);

	return;

}


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

	/*get the statistic */

	getTopFreqCSs(csMap,20);

	getTopFreqCSs(csMap,10);

	getTopFreqCSs(csMap,5);

	getTopFreqCSs(csMap,2);


	BBPreclaim(sbat); 
	BBPreclaim(pbat); 

	free (buff); 
	hashmap_free(csMap);

	*ret = 1; 
	return MAL_SUCCEED; 
}
