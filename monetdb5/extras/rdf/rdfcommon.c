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
 * Copyright August 2008-2013 MonetDB B.V.
 * All Rights Reserved.
 */

#include "monetdb_config.h"
#include "rdfcommon.h"

void copyOidSet(oid* dest, oid* orig, int len){
	memcpy(dest, orig, len * sizeof(oid));
}

void copyIntSet(int* dest, int* orig, int len){
	memcpy(dest, orig, len * sizeof(int));
}

void copybatSet(bat* dest, bat* orig, int len){
	memcpy(dest, orig, len * sizeof(int));
}

void initCharArray(char* inputArr, int num, char defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}
}

void initArray(oid* inputArr, int num, oid defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}
}


void initIntArray(int* inputArr, int num, oid defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}
}

void getNumCombinedP(oid* arr1, oid* arr2, int m, int n, int *numCombineP){
	
	int i = 0, j = 0;
	int pos = 0;

	while( j < m && i < n )
	{
		if( arr1[j] < arr2[i] ){
			pos++;
			j++;
		}
		else if( arr1[j] == arr2[i] )
		{
			pos++;
			j++;
			i++;
		}
		else if( arr1[j] > arr2[i] ){
			pos++;
			i++;
		}
	}
	if (j == m && i < n){
		while (i < n){
			pos++;
			i++;
		}		
	} 

	if (j < m && i == n){
		while (j < m){
			pos++;
			j++;
		}		
	} 
	
	*numCombineP = pos; 

}

void mergeOidSets(oid* arr1, oid* arr2, oid* mergeArr, int m, int n, int *numCombineP){
	
	int i = 0, j = 0;
	int pos = 0;

	while( j < m && i < n )
	{
		if( arr1[j] < arr2[i] ){
			mergeArr[pos] = arr1[j];
			pos++;
			j++;
		}
		else if( arr1[j] == arr2[i] )
		{
			mergeArr[pos] = arr1[j];	
			pos++;
			j++;
			i++;
		}
		else if( arr1[j] > arr2[i] ){
			mergeArr[pos] = arr2[i];
			pos++;
			i++;
		}
	}
	if (j == m && i < n){
		while (i < n){
			mergeArr[pos] = arr2[i];
			pos++;
			i++;
		}		
	} 

	if (j < m && i == n){
		while (j < m){
			mergeArr[pos] = arr1[j];
			pos++;
			j++;
		}		
	} 
	
	*numCombineP = pos; 
}

/*
 * Find the common items in multiple item lists
 * Assumption: All list are sorted 
 * lists: Lists of items, each list is sorted
 * listCount: Contains number of items in each list
 * interlist: List of intersecting input lists
 * num: number of lists
 * */

void intersect_oidsets(oid** lists, int* listcount, int num, oid** interlist, int *internum){
	int i, j; 	
	int* tmpidx = NULL; 
	int mincount = listcount[0];
	int minidx = 0; 	//index of list contains list number of items
	int numaccept = 0; 

	//Get the max count
	for (i = 0; i < num; i++){
		if (listcount[i] < mincount) {
			mincount = listcount[i];
			minidx = i; 
		}
	}
	
	//Init interesection list using the min count (may be reduntdant)
	*interlist = (oid *)malloc(sizeof(oid) * mincount); 
	
	//Store the current validating position of each list
	tmpidx = (int *) malloc(sizeof(int) * num); 
	for (i = 0; i < num; i++){
		tmpidx[i] = 0; 
	}
	
	//Go through each item in the shorest list
	for (i = 0; i < listcount[minidx]; i++){
		oid tmpitem = lists[minidx][i]; 
		char isaccept = 1; 

		//Go through each list 
		for (j = 0; j < num; j++){
			
			if (isaccept == 0) break; 

			//increase the pointer if the current item of the list is smaller than tmpitem
			while (tmpidx[j] < listcount[j] && lists[j][tmpidx[j]] < tmpitem){
				tmpidx[j]++; 	
			}
			
			//Check whether the current item is the same 
			if (tmpidx[j] == listcount[j] || lists[j][tmpidx[j]] != tmpitem)
				isaccept = 0;
		}

		//Accept an item
		if (isaccept){
			(*interlist)[numaccept] = tmpitem; 
			numaccept++; 
		}
	}
	
	free(tmpidx); 

	*internum = numaccept; 
}

void intersect_intsets(int** lists, int* listcount, int num, int** interlist, int *internum){
	int i, j; 	
	int* tmpidx = NULL; 
	int mincount = listcount[0];
	int minidx = 0; 	//index of list contains list number of items
	int numaccept = 0; 

	//Get the max count
	for (i = 0; i < num; i++){
		if (listcount[i] < mincount) {
			mincount = listcount[i];
			minidx = i; 
		}
	}
	
	//Init interesection list using the min count (may be reduntdant)
	*interlist = (int *)malloc(sizeof(int) * mincount); 
	
	//Store the current validating position of each list
	tmpidx = (int *) malloc(sizeof(int) * num); 
	for (i = 0; i < num; i++){
		tmpidx[i] = 0; 
	}
	
	//Go through each item in the shorest list
	for (i = 0; i < listcount[minidx]; i++){
		int tmpitem = lists[minidx][i]; 
		char isaccept = 1; 

		//Go through each list 
		for (j = 0; j < num; j++){
			
			if (isaccept == 0) break; 

			//increase the pointer if the current item of the list is smaller than tmpitem
			while (tmpidx[j] < listcount[j] && lists[j][tmpidx[j]] < tmpitem){
				tmpidx[j]++; 	
			}
			
			//Check whether the current item is the same 
			if (tmpidx[j] == listcount[j] || lists[j][tmpidx[j]] != tmpitem)
				isaccept = 0;
		}

		//Accept an item
		if (isaccept){
			(*interlist)[numaccept] = tmpitem; 
			numaccept++; 
		}
	}
	
	free(tmpidx); 

	*internum = numaccept; 
}

static
int compareOid (const void * a, const void * b) {
	return (*(oid *)a - *(oid*)b); // sort ascending
}



/*
 * This function should work well with not so long array
 *  
 * */
void get_sorted_distinct_set(oid* src, oid** des, int numsrc, int *numdesc){
	int i; 
	oid lastid; 
	int idx; 
	oid* tmp = (oid *) malloc(sizeof(oid) * numsrc);
	oid* tmp2 = (oid *) malloc(sizeof(oid) * numsrc);
	
	assert(numsrc > 0); 

	copyOidSet(tmp, src, numsrc); 

	qsort(tmp, numsrc, sizeof(oid), compareOid);
	
	lastid = tmp[0];
	tmp2[0] = lastid; 
	idx = 0; 
	for (i = 1; i < numsrc; i++){
		if (tmp[i] != lastid){	//new item
			idx++;
			tmp2[idx] = tmp[i];
			lastid = tmp[i]; 
		}
	}
	
	//Stor number of items of output array
	*numdesc = idx + 1; 

	*des = (oid *) malloc(sizeof(oid) * (*numdesc));
	
	copyOidSet(*des, tmp2, *numdesc); 
	
	free(tmp); 
	free(tmp2); 
}


void appendArrayToBat(BAT *b, BUN* inArray, int num){
	if (num > 0){
		BUN r = BUNlast(b);
		if (r + num > b->batCapacity){
			BATextend(b, b->batCapacity + smallbatsz); 
		}
		memcpy(Tloc(b, BUNlast(b)), inArray, sizeof(BUN) * num); 
		BATsetcount(b, (BUN) (b->batCount + num)); 
	}
}


void appendIntArrayToBat(BAT *b, int* inArray, int num){
	if (num > 0){
		BUN r = BUNlast(b);
		if (r + num > b->batCapacity){
			BATextend(b, b->batCapacity + smallbatsz); 
		}
		memcpy(Tloc(b, BUNlast(b)), inArray, sizeof(int) * num); 
		BATsetcount(b, (BUN) (b->batCount + num)); 
	}
}


void appendbatArrayToBat(BAT *b, bat* inArray, int num){
	if (num > 0){
		BUN r = BUNlast(b);
		if (r + num > b->batCapacity){
			BATextend(b, b->batCapacity + smallbatsz); 
		}
		memcpy(Tloc(b, BUNlast(b)), inArray, sizeof(bat) * num); 
		BATsetcount(b, (BUN) (b->batCount + num)); 
	}
}
