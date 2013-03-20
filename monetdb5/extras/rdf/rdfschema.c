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
#include "tokenizer.h"

#define SHOWPROPERTYNAME 1

str
RDFSchemaExplore(int *ret, str *tbname, str *clname)
{
	printf("Explore from table %s with colum %s \n", *tbname, *clname);
	*ret = 1; 
	return MAL_SUCCEED;
}

static void copyOidSet(oid* dest, oid* orig, int len){
	memcpy(dest, orig, len * sizeof(oid));
}


static void copyTypesSet(char* dest, char* orig, int len){
	memcpy(dest, orig, len * sizeof(char));
}


/*
 * Hashing function for a set of values
 * Rely on djb2 http://www.cse.yorku.ca/~oz/hash.html
 *
 */
static oid RDF_hash_Tyleslist(char* types, int num){
	//unsigned int hashCode = 5381u; 
	oid  hashCode = 5381u;
	int i; 

	for (i = 0; i < num; i++){
		hashCode = ((hashCode << 5) + hashCode) + types[i];
	}
	
	// return 0x7fffffff & hashCode 
	return hashCode;
}

/*
static void printArray(oid* inputArr, int num){
	int i; 
	printf("Print array \n");
	for (i = 0; i < num; i++){
		printf("%d:  " BUNFMT "\n",i, inputArr[i]);
	}
	printf("End of array \n ");
}
*/

static void initArray(oid* inputArr, int num, oid defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}
}


static void initCharArray(char* inputArr, int num, char defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}
}

static void generateFreqCSMap(CSset *freqCSset, char *csFreqMap){
	int i; 
	for (i = 0; i < freqCSset->numCSadded; i++){
		csFreqMap[freqCSset->items[i].csId] = 1;
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
CSrel* creataCSrel(oid csoid){
	CSrel *csrel = (CSrel*) malloc(sizeof(CSrel));
	csrel->origCSoid = csoid; 
	csrel->lstRefCSoid = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);
	csrel->lstPropId = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);
	csrel->lstCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);		
	csrel->numRef = 0;
	csrel->numAllocation = INIT_NUM_CSREL;

	return csrel; 
}


static 
void addReltoCSRel(oid origCSoid, oid refCSoid, oid propId, CSrel *csrel)
{
	void *_tmp; 
	void *_tmp1; 
	void *_tmp2; 

	int i = 0; 

	assert (origCSoid == csrel->origCSoid);

	while (i < csrel->numRef){
		if (refCSoid == csrel->lstRefCSoid[i] && propId == csrel->lstPropId[i]){
			//Existing
			break; 
		}
		i++;
	}
	
	if (i != csrel->numRef){ 
		csrel->lstCnt[i]++; 
		return; 
	}
	else{	// New Ref
	
		if(csrel->numRef == csrel->numAllocation) 
		{ 
			csrel->numAllocation += INIT_NUM_CSREL; 
			
			_tmp = realloc(csrel->lstRefCSoid, (csrel->numAllocation * sizeof(oid)));
			_tmp1 = realloc(csrel->lstPropId, (csrel->numAllocation * sizeof(oid)));
			_tmp2 = realloc(csrel->lstCnt, (csrel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csrel->lstRefCSoid = (oid*)_tmp;
			csrel->lstPropId = (oid*)_tmp1; 
			csrel->lstCnt = (int*)_tmp2; 
		}

		csrel->lstRefCSoid[csrel->numRef] = refCSoid;
		csrel->lstPropId[csrel->numRef] = propId;
		csrel->lstCnt[csrel->numRef] = 1; 
		csrel->numRef++;
	}
}


static 
void addReltoCSRelWithFreq(oid origCSoid, oid refCSoid, oid propId, int freq, CSrel *csrel)
{
	void *_tmp; 
	void *_tmp1; 
	void *_tmp2; 

	int i = 0; 

	assert (origCSoid == csrel->origCSoid);

	while (i < csrel->numRef){
		if (refCSoid == csrel->lstRefCSoid[i] && propId == csrel->lstPropId[i]){
			//Existing
			break; 
		}
		i++;
	}
	
	if (i != csrel->numRef){ 
		csrel->lstCnt[i] = csrel->lstCnt[i] + freq; 
		return; 
	}
	else{	// New Ref
	
		if(csrel->numRef == csrel->numAllocation) 
		{ 
			csrel->numAllocation += INIT_NUM_CSREL; 
			
			_tmp = realloc(csrel->lstRefCSoid, (csrel->numAllocation * sizeof(oid)));
			_tmp1 = realloc(csrel->lstPropId, (csrel->numAllocation * sizeof(oid)));		
			_tmp2 = realloc(csrel->lstCnt, (csrel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csrel->lstRefCSoid = (oid*)_tmp;
			csrel->lstPropId = (oid*)_tmp1; 
			csrel->lstCnt = (int*)_tmp2; 
		}

		csrel->lstRefCSoid[csrel->numRef] = refCSoid;
		csrel->lstPropId[csrel->numRef] = propId;
		csrel->lstCnt[csrel->numRef] = freq; 
		csrel->numRef++;
	}
}


static 
CSrel* initCSrelset(oid numCSrel){
	oid i; 
	CSrel *csrelSet = (CSrel*) malloc(sizeof(CSrel) * numCSrel); 
	CSrel *csrel; 
	for (i = 0; i < numCSrel; i++){
		csrel = creataCSrel(i); 
		csrelSet[i] = (CSrel) *csrel;
	}
	return csrelSet;
}

static 
void freeCSrelSet(CSrel *csrelSet, int numCSrel){
	int i; 

	for (i = 0; i < numCSrel; i++){
		free(csrelSet[i].lstRefCSoid);
		free(csrelSet[i].lstCnt); 
	}
	free(csrelSet);
}

static 
void printCSrelSet(CSrel *csrelSet, char *csFreqMap, BAT* freqBat, int num, char isWriteTofile, int freqThreshold){

	int 	i; 
	int 	j; 
	int 	*freq; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];

	if (isWriteTofile == 0){
		for (i = 0; i < num; i++){
			if (csrelSet[i].numRef != 0){	//Only print CS with FK
				printf("Relationship %d: ", i);
				freq  = (int *) Tloc(freqBat, i);
				printf("CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelSet[i].origCSoid, *freq, csFreqMap[i]);
				for (j = 0; j < csrelSet[i].numRef; j++){
					printf(BUNFMT " (%d) ", csrelSet[i].lstRefCSoid[j],csrelSet[i].lstCnt[j]);	
				}	
				printf("\n");
			}
		}
	}
	else{
	
		strcpy(filename, "csRelationship");
		sprintf(tmpStr, "%d", freqThreshold);
		strcat(filename, tmpStr);
		strcat(filename, ".txt");

		fout = fopen(filename,"wt"); 

		for (i = 0; i < num; i++){
			if (csrelSet[i].numRef != 0){	//Only print CS with FK
				fprintf(fout, "Relationship %d: ", i);
				freq  = (int *) Tloc(freqBat, i);
				fprintf(fout, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelSet[i].origCSoid, *freq, csFreqMap[i]);
				for (j = 0; j < csrelSet[i].numRef; j++){
					fprintf(fout, BUNFMT " (%d) ", csrelSet[i].lstRefCSoid[j],csrelSet[i].lstCnt[j]);	
				}	
				fprintf(fout, "\n");
			}
		}


		fclose(fout);
	}
	
}

/*
 * Show the relationship from each CS to maximumFreqCSs
 * */

static 
str printCSrelWithMaxSet(oid* csSuperCSMap, CSrel *csrelToMaxSet, CSrel *csrelFromMaxSet, CSrel *csrelBetweenMaxSet, CSrel *csrelSet, char *csFreqMap, BAT* freqBat, int num, int freqThreshold){

	int 	i; 
	int 	j; 
	int 	*freq; 
	FILE 	*fout, *fout1, *fout1filter, *fout2,*fout2filter; 
	char 	filename[100], filename1[100], filename2[100];
	char 	tmpStr[50];
	oid 	maxCSoid; 

#if SHOWPROPERTYNAME
	str 	propStr; 
	int	ret; 
	char*   schema = "rdf";

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

#endif	



	// Merge the relationships to create csrelToMaxSet, csrelFromMaxSet
	for (i = 0; i < num; i++){
		maxCSoid = csSuperCSMap[csrelSet[i].origCSoid];
		if (csrelSet[i].numRef != 0){
			for (j = 0; j < csrelSet[i].numRef; j++){		
				if (csSuperCSMap[csrelSet[i].lstRefCSoid[j]] != BUN_NONE){
					addReltoCSRelWithFreq(csrelSet[i].origCSoid, csSuperCSMap[csrelSet[i].lstRefCSoid[j]], csrelSet[i].lstPropId[j], csrelSet[i].lstCnt[j], &csrelToMaxSet[i]);
				}
			}


			// Add to csrelFromMaxSet
			// For a referenced CS that is frequent, use its maxCSoid
			// Else, use its csoid
			if (maxCSoid != BUN_NONE){
				for (j = 0; j < csrelSet[i].numRef; j++){		
					if (csSuperCSMap[csrelSet[i].lstRefCSoid[j]] != BUN_NONE){
						addReltoCSRelWithFreq(maxCSoid, csSuperCSMap[csrelSet[i].lstRefCSoid[j]], csrelSet[i].lstPropId[j], csrelSet[i].lstCnt[j], &csrelFromMaxSet[maxCSoid]);
					}
					else{
						addReltoCSRelWithFreq(maxCSoid, csrelSet[i].lstRefCSoid[j], csrelSet[i].lstPropId[j], csrelSet[i].lstCnt[j], &csrelFromMaxSet[maxCSoid]);
					}
				}
			}
		}
	}

	// Write csrelToMaxSet to File
	
	strcpy(filename, "csRelationshipToMaxFreqCS");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	for (i = 0; i < num; i++){
		if (csrelToMaxSet[i].numRef != 0){	//Only print CS with FK
			fprintf(fout, "Relationship %d: ", i);
			freq  = (int *) Tloc(freqBat, i);
			fprintf(fout, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelToMaxSet[i].origCSoid, *freq, csFreqMap[i]);
			for (j = 0; j < csrelToMaxSet[i].numRef; j++){
				fprintf(fout, BUNFMT " (%d) ", csrelToMaxSet[i].lstRefCSoid[j],csrelToMaxSet[i].lstCnt[j]);	
			}	
			fprintf(fout, "\n");
		}
	}

	fclose(fout);

	// Write csrelFromMaxSet to File
	
	strcpy(filename1, "csRelationshipFromMaxFreqCS");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename1, tmpStr);
	strcat(filename1, ".txt");

	fout1 = fopen(filename1,"wt"); 
	strcat(filename1, ".filter");
	fout1filter = fopen(filename1,"wt");

	for (i = 0; i < num; i++){
		if (csrelFromMaxSet[i].numRef != 0){	//Only print CS with FK
			fprintf(fout1, "Relationship %d: ", i);
			freq  = (int *) Tloc(freqBat, i);
			fprintf(fout1, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelFromMaxSet[i].origCSoid, *freq, csFreqMap[i]);
			fprintf(fout1filter, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelFromMaxSet[i].origCSoid, *freq, csFreqMap[i]);		

			for (j = 0; j < csrelFromMaxSet[i].numRef; j++){
				fprintf(fout1, BUNFMT " (%d) ", csrelFromMaxSet[i].lstRefCSoid[j],csrelFromMaxSet[i].lstCnt[j]);	

				// Only put into the filer output file, the reference with appears in > 1 % of original CS
				if (*freq < csrelFromMaxSet[i].lstCnt[j]*100){
					fprintf(fout1filter, BUNFMT " (%d) ", csrelFromMaxSet[i].lstRefCSoid[j],csrelFromMaxSet[i].lstCnt[j]);
				}
			}	
			fprintf(fout1, "\n");
			fprintf(fout1filter, "\n");

		}
	}

	fclose(fout1);
	fclose(fout1filter);

	/*------------------------*/

	strcpy(filename2, "csRelationshipBetweenMaxFreqCS");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename2, tmpStr);
	strcat(filename2, ".txt");

	fout2 = fopen(filename2,"wt"); 
	strcat(filename2, ".filter");
	fout2filter = fopen(filename2,"wt");

	// Merge the csrelToMaxSet --> csrelBetweenMaxSet
	for (i = 0; i < num; i++){
		maxCSoid = csSuperCSMap[csrelToMaxSet[i].origCSoid];
		if (csrelToMaxSet[i].numRef != 0 && maxCSoid != BUN_NONE){
			for (j = 0; j < csrelToMaxSet[i].numRef; j++){		
				assert(csSuperCSMap[csrelToMaxSet[i].lstRefCSoid[j]] == csrelToMaxSet[i].lstRefCSoid[j]);
				addReltoCSRelWithFreq(maxCSoid, csSuperCSMap[csrelToMaxSet[i].lstRefCSoid[j]], csrelToMaxSet[i].lstPropId[j], csrelToMaxSet[i].lstCnt[j], &csrelBetweenMaxSet[maxCSoid]);
			}
		}
	}
	
	for (i = 0; i < num; i++){
		if (csrelBetweenMaxSet[i].numRef != 0){	//Only print CS with FK
			fprintf(fout2, "Relationship %d: ", i);
			fprintf(fout2filter, "Relationship %d: ", i);
			freq  = (int *) Tloc(freqBat, i);
			fprintf(fout2, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelBetweenMaxSet[i].origCSoid, *freq, csFreqMap[i]);
			fprintf(fout2filter, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", csrelBetweenMaxSet[i].origCSoid, *freq, csFreqMap[i]);

			for (j = 0; j < csrelBetweenMaxSet[i].numRef; j++){
				#if SHOWPROPERTYNAME
				takeOid(csrelBetweenMaxSet[i].lstPropId[j], &propStr);	
				fprintf(fout2, BUNFMT "(P:" BUNFMT " - %s) (%d) ", csrelBetweenMaxSet[i].lstRefCSoid[j],csrelBetweenMaxSet[i].lstPropId[j], propStr, csrelBetweenMaxSet[i].lstCnt[j]);	
				#else
				fprintf(fout2, BUNFMT "(P:" BUNFMT ") (%d) ", csrelBetweenMaxSet[i].lstRefCSoid[j],csrelBetweenMaxSet[i].lstPropId[j], csrelBetweenMaxSet[i].lstCnt[j]);	
				#endif

				if (*freq < csrelBetweenMaxSet[i].lstCnt[j]*100){
					fprintf(fout2filter, BUNFMT "(P:" BUNFMT ") (%d) ", csrelBetweenMaxSet[i].lstRefCSoid[j],csrelBetweenMaxSet[i].lstPropId[j], csrelBetweenMaxSet[i].lstCnt[j]);	
				}
			}	
			fprintf(fout2, "\n");
			fprintf(fout2filter, "\n");
		}
	}

	fclose(fout2);
	fclose(fout2filter);

#if SHOWPROPERTYNAME
	TKNZRclose(&ret);

#endif
	return MAL_SUCCEED; 
}

static 
void printSubCSInformation(SubCSSet *subcsset, int num, char isWriteTofile, int freqThreshold){

	int i; 
	int j; 
	
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];

	if (isWriteTofile == 0){
		for (i = 0; i < num; i++){
			if (subcsset[i].numSubCS != 0){	//Only print CS with FK
				printf("CS " BUNFMT ": ", subcsset[i].csId);
				for (j = 0; j < subcsset[i].numSubCS; j++){
					printf(BUNFMT " (%d) ", subcsset[i].subCSs[j].subCSId, subcsset[i].freq[j]);	
				}	
				printf("\n");
			}
		}
	}
	else{
		
		strcpy(filename, "csSubCSInfo");
		sprintf(tmpStr, "%d", freqThreshold);
		strcat(filename, tmpStr);
		strcat(filename, ".txt");

		fout = fopen(filename,"wt"); 

		for (i = 0; i < num; i++){
			if (subcsset[i].numSubCS != 0){	//Only print CS with FK
				fprintf(fout, "CS " BUNFMT ": ", subcsset[i].csId);
				for (j = 0; j < subcsset[i].numSubCS; j++){
					fprintf(fout, BUNFMT " (%d) ", subcsset[i].subCSs[j].subCSId, subcsset[i].freq[j]);	
				}	
				fprintf(fout, "\n");
			}
		}

		fclose(fout);
	}
}

static 
SubCS* creatSubCS(oid subCSId, int numP, char* buff, oid subCSsign){
	SubCS *subcs = (SubCS*) malloc(sizeof(SubCS)); 
	subcs->subTypes =  (char*) malloc(sizeof(char) * numP);
	
	copyTypesSet(subcs->subTypes, buff, numP); 
	subcs->subCSId = subCSId;
	subcs->numSubTypes = numP; 
	subcs->sign = subCSsign; 
	return subcs; 
}

static 
SubCSSet* createaSubCSSet(oid csId){
	SubCSSet* subCSset = (SubCSSet*) malloc(sizeof(SubCSSet));
	subCSset->csId = csId; 
	subCSset->numAllocation = INIT_NUM_SUBCS;
	subCSset->numSubCS = 0;
	subCSset->subCSs = (SubCS*) malloc(sizeof(SubCS) * INIT_NUM_SUBCS);
	subCSset->freq = (int*) malloc(sizeof(int) * INIT_NUM_SUBCS);

	return subCSset;
}

static 
SubCSSet* initCS_SubCSMap(oid numSubCSSet){
	oid i; 
	SubCSSet *subcssets = (SubCSSet*) malloc(sizeof(SubCSSet) * numSubCSSet); 
	SubCSSet *subcsset;
	for (i = 0; i < numSubCSSet;i++){
		subcsset = createaSubCSSet(i); 
		subcssets[i] = (SubCSSet) *subcsset; 
	}

	return subcssets; 

}

static 
void freeCS_SubCSMapSet(SubCSSet *subcssets, int numSubCSSet){
	int i; 
	int j; 

	for (i = 0; i < numSubCSSet; i++){
		for (j = 0; j < subcssets[i].numSubCS; j++){
			free(subcssets[i].subCSs[j].subTypes);
		}
		free(subcssets[i].subCSs);
		free(subcssets[i].freq); 
	}
	free(subcssets);
}

static 
char checkExistsubCS(oid subCSsign, char* types, int numTypes,  SubCSSet *subcsset, oid *existCSId){
	char isFound = 0; 
	int i; 
	int j; 
	for (i = 0; i < subcsset->numSubCS; i++){
		if ((subcsset->subCSs[i].sign != subCSsign) || (subcsset->subCSs[i].numSubTypes != numTypes))
			continue; 
		else{
			isFound = 1; 
			for (j = 0; j < numTypes; j++){
				if (subcsset->subCSs[i].subTypes[j] != types[j]){
					isFound = 0; 
					break; 
				}
			}

			if (isFound == 1){
				*existCSId = i; 
				return isFound; 
			}
		}
	}

	*existCSId = subcsset->numSubCS; 	//Id of new SubCS

	return isFound; 
}

static 
void addSubCStoSet(SubCSSet *subcsSet, SubCS item)
{
	void *_tmp; 
	void *_tmp2; 

	if(subcsSet->numSubCS == subcsSet->numAllocation) 
	{ 
		subcsSet->numAllocation += INIT_NUM_SUBCS; 
		
		_tmp = realloc(subcsSet->subCSs, (subcsSet->numAllocation * sizeof(SubCS)));
		_tmp2 = realloc(subcsSet->freq, (subcsSet->numAllocation * sizeof(int))); 
	
		if (!_tmp){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		}
		subcsSet->subCSs = (SubCS*)_tmp;
		subcsSet->freq = (int *) _tmp2; 
	}

	subcsSet->subCSs[subcsSet->numSubCS] = item;
	subcsSet->freq[subcsSet->numSubCS] = 1;

	subcsSet->numSubCS++;

}

static 
oid addSubCS(char *buff, int numP, int csId, SubCSSet* csSubCSMap){
	SubCSSet *subcsset;
	oid subCSsign; 
	char isFound; 
	oid  subCSId; 
	SubCS *subCS; 


	subcsset = &(csSubCSMap[csId]);

	// Check the duplication
	subCSsign = RDF_hash_Tyleslist(buff, numP);

	isFound = checkExistsubCS(subCSsign, buff, numP, subcsset, &subCSId);
	
	if (isFound == 0){	// Add new 
		subCS = creatSubCS(subCSId, numP, buff, subCSsign);
		addSubCStoSet(subcsset,*subCS);
	}
	else{			// Exist
		//Update frequency
		subcsset->freq[subCSId]++;
	}

	return subCSId; 

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
	CSset *csSet = (CSset*) malloc(sizeof(CSset)); 
	csSet->items = (CS*) malloc(sizeof(CS) * INIT_NUM_CS); 
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
#if STOREFULLCS
static
CS* creatCS(oid csId, int numP, oid* buff, oid subjectId, oid* lstObject)
#else
static 
CS* creatCS(oid csId, int numP, oid* buff)
#endif	
{
	CS *cs = (CS*)malloc(sizeof(CS)); 
	cs->lstProp =  (oid*) malloc(sizeof(oid) * numP);
	
	if (cs->lstProp == NULL){
		printf("Malloc failed. at %d", numP);
		exit(-1); 
	}

	copyOidSet(cs->lstProp, buff, numP); 
	cs->csId = csId;
	cs->numProp = numP; 
	cs->numAllocation = numP; 
	cs->isSubset = 0; /*By default, this CS is not known to be a subset of any other CS*/
	#if STOREFULLCS
	cs->lstObj =  (oid*) malloc(sizeof(oid) * numP);
	if (cs->lstObj == NULL){
		printf("Malloc failed. at %d", numP);
		exit(-1); 
	}
	copyOidSet(cs->lstObj, lstObject, numP); 
	cs->subject = subjectId; 
	#endif
	return cs; 
}


static 
str printFreqCSSet(CSset *freqCSset, oid* csSuperCSMap, BAT *freqBat, BAT *mapbat, char isWriteTofile, int freqThreshold){

	int 	i; 
	int 	j; 
	int 	*freq; 
	FILE 	*fout, *fout2; 
	char 	filename[100], filename2[100];
	char 	tmpStr[20];

#if SHOWPROPERTYNAME
	str 	propStr; 
	str	subStr; 
	str	objStr; 
	oid 	objOid; 
	char 	objType; 
	BATiter mapi; 
	int	ret; 
	BUN	bun; 
	char*   schema = "rdf";


	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	

	mapi = bat_iterator(mapbat); 
#endif	

	if (isWriteTofile == 0){
		for (i = 0; i < freqCSset->numCSadded; i++){
			CS cs = (CS)freqCSset->items[i];
			freq  = (int *) Tloc(freqBat, cs.csId);

			takeOid(cs.csId, &subStr);	

			printf("CS " BUNFMT " (Freq: %d) | Subject: %s  | Parent " BUNFMT " \n", cs.csId, *freq, subStr, csSuperCSMap[cs.csId]);
			for (j = 0; j < cs.numProp; j++){
				printf("  P:" BUNFMT " --> \n", cs.lstProp[j]);	
			}	
			printf("\n");
		}
	}
	else{
	
		strcpy(filename, "freqCSFullInfo");
		sprintf(tmpStr, "%d", freqThreshold);
		strcat(filename, tmpStr);
		strcpy(filename2, "max");
		strcat(filename2, filename);  
		strcat(filename, ".txt");
		strcat(filename2, ".txt");

		fout = fopen(filename,"wt"); 
		fout2 = fopen(filename2, "wt");

		for (i = 0; i < freqCSset->numCSadded; i++){
			CS cs = (CS)freqCSset->items[i];
			freq  = (int *) Tloc(freqBat, cs.csId);
			

			takeOid(cs.csId, &subStr);	
			
			fprintf(fout,"CS " BUNFMT " (Freq: %d) | Subject: %s  | Parent " BUNFMT " \n", cs.csId, *freq, subStr, csSuperCSMap[cs.csId]);

			// Filter max freq cs set
			if (csSuperCSMap[cs.csId] == cs.csId){
				fprintf(fout2,"CS " BUNFMT " (Freq: %d) | Subject: %s  | Parent " BUNFMT " \n", cs.csId, *freq, subStr, csSuperCSMap[cs.csId]);
			}

			for (j = 0; j < cs.numProp; j++){
				takeOid(cs.lstProp[j], &propStr);
				//fprintf(fout, "  P:" BUNFMT " --> ", cs.lstProp[j]);	
				fprintf(fout, "  P:%s --> ", propStr);	
				if (csSuperCSMap[cs.csId] == cs.csId){
					fprintf(fout2, "  P:%s --> ", propStr);
				}

				// Get object value
				objOid = cs.lstObj[j]; 

				objType = (char) (objOid >> (sizeof(BUN)*8 - 3))  &  3 ; 

				if (objType == URI){
					takeOid(objOid, &objStr); 
				}
				else{
					objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */ 
					bun = BUNfirst(mapbat);
					objStr = (str) BUNtail(mapi, bun + objOid); 
				}

				fprintf(fout, "  O: %s \n", objStr);
				if (csSuperCSMap[cs.csId] == cs.csId){
					fprintf(fout2, "  O: %s \n", objStr);
				}


			}	
			fprintf(fout, "\n");
			if (csSuperCSMap[cs.csId] == cs.csId){
				fprintf(fout2, "\n");
			}
		}

		fclose(fout);
		fclose(fout2);
	}
	
#if SHOWPROPERTYNAME
	TKNZRclose(&ret);
#endif
	return MAL_SUCCEED;
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
	if (r + num > b->batCapacity){
		BATextend(b, b->batCapacity + smallbatsz); 
	}
	//for (i = 0; i < num; i++){
	memcpy(Tloc(b, BUNlast(b)), inArray, sizeof(BUN) * num); 
	//}
	BATsetcount(b, (BUN) (b->batCount + num)); 
	
}

static 
char checkCSduplication(BAT* hsKeyBat, BAT* pOffsetBat, BAT* fullPBat, BUN cskey, oid* key, int numK, oid *csId){
	oid *offset; 
	oid *offset2; 
	int numP; 
	int i; 
	BUN *existvalue; 
	BUN pos; 
	char isDuplication = 0; 

	BATiter bi = bat_iterator(BATmirror(hsKeyBat));
			
	HASHloop(bi, hsKeyBat->T->hash, pos, (ptr) &cskey){
		//printf("  pos: " BUNFMT, pos);

		offset = (oid *) Tloc(pOffsetBat, pos); 
		if ((pos + 1) < pOffsetBat->batCount){
			offset2 = (oid *)Tloc(pOffsetBat, pos + 1);
			numP = *offset2 - *offset;
		}
		else{
			numP = BUNlast(fullPBat) - *offset;
		}


		// Check each value
		if (numK != numP) {
			continue; 
		}
		else{
			isDuplication = 1; 
			existvalue = (oid *)Tloc(fullPBat, *offset);	
			for (i = 0; i < numP; i++){
				//if (key[i] != (int)*existvalue++) {
				if (key[i] != existvalue[i]) {
					isDuplication = 0;
					break; 
				}	
			}


			//Everything match
			if (isDuplication == 1){
				//printf("Everything match!!!!!");
				*csId = pos; 
				return 1; 
			}
		}
		
		
	}
	

	*csId = pos;  // = BUN_NONE

	return 0;
}

/*
static 
void testBatHash(void){

	BUN	bun; 
	BAT* 	testBat; 
	int 	i; 
	oid	key[7] = {3,5,6,3,5,7,5};
	oid 	csKey; 

	testBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
		
	for (i = 0; i < 7; i++){
		csKey = key[i]; 
		bun = BUNfnd(BATmirror(testBat),(ptr) &key[i]);
		if (bun == BUN_NONE) {
			if (testBat->T->hash && BATcount(testBat) > 4 * testBat->T->hash->mask) {
				HASHdestroy(testBat);
				BAThash(BATmirror(testBat), 2*BATcount(testBat));
			}

			testBat = BUNappend(testBat, (ptr) &csKey, TRUE);
		
		}
		else{

			printf("Input: " BUNFMT, csKey);
			printf(" --> bun: " BUNFMT "\n", bun);



			testBat = BUNappend(testBat, (ptr) &csKey, TRUE);

		}
	}
	BATprint(testBat);

	BBPreclaim(testBat); 
}
*/

static 
void addNewCS(CSBats *csBats, BUN* csKey, oid* key, oid *csoid, int num){
	int freq = 1; 
	BUN	offset; 

	if (csBats->hsKeyBat->T->hash && BATcount(csBats->hsKeyBat) > 4 * csBats->hsKeyBat->T->hash->mask) {
		HASHdestroy(csBats->hsKeyBat);
		BAThash(BATmirror(csBats->hsKeyBat), 2*BATcount(csBats->hsKeyBat));
	}

	csBats->hsKeyBat = BUNappend(csBats->hsKeyBat, csKey, TRUE);
		
	(*csoid)++;
		
	offset = BUNlast(csBats->fullPBat);
	/* Add list of p to fullPBat and pOffsetBat*/
	BUNappend(csBats->pOffsetBat, &offset , TRUE);
	appendArrayToBat(csBats->fullPBat, key, num);

	BUNappend(csBats->freqBat, &freq, TRUE); 
}
/*
 * Put a CS to the hashmap. 
 * While putting CS to the hashmap, update the support (frequency) value 
 * for an existing CS, and check whether it becomes a frequent CS or not. 
 * If yes, add that frequent CS to the freqCSset. 
 *
 * */
#if STOREFULLCS
static 
oid putaCStoHash(CSBats *csBats, oid* key, int num, 
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset, oid subjectId, oid* buffObjs)
#else
static 
oid putaCStoHash(CSBats *csBats, oid* key, int num, 
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset)
#endif	
{
	BUN 	csKey; 
	int 	*freq; 
	CS	*freqCS; 
	BUN	bun; 
	oid	csId; 		/* Id of the characteristic set */
	char	isDuplicate = 0; 

	csKey = RDF_hash_oidlist(key, num);
	bun = BUNfnd(BATmirror(csBats->hsKeyBat),(ptr) &csKey);
	if (bun == BUN_NONE) {
		csId = *csoid; 
		addNewCS(csBats, &csKey, key, csoid, num);
		
		//Handle the case when freqThreshold == 1 
		if (isStoreFreqCS ==1 && freqThreshold == 1){
			#if STOREFULLCS
			freqCS = creatCS(csId, num, key, subjectId, buffObjs);		
			#else
			freqCS = creatCS(csId, num, key);			
			#endif
			addCStoSet(freqCSset, *freqCS);
		}
	}
	else{
		//printf("Same HashKey: ");	
		/* Check whether it is really an duplication (same hashvalue but different list of */
		isDuplicate = checkCSduplication(csBats->hsKeyBat, csBats->pOffsetBat, csBats->fullPBat, csKey, key, num, &csId);

		if (isDuplicate == 0) {
			//printf(" No duplication (new CS) \n");	
			// New CS
			csId = *csoid;
			addNewCS(csBats, &csKey, key, csoid, num);
			
			//Handle the case when freqThreshold == 1 
			if (isStoreFreqCS ==1 && freqThreshold == 1){
				
				#if STOREFULLCS
				freqCS = creatCS(csId, num, key, subjectId, buffObjs);		
				#else
				freqCS = creatCS(csId, num, key);			
				#endif
				addCStoSet(freqCSset, *freqCS);
			}

		}
		else{
			//printf(" Duplication (existed CS) at csId = " BUNFMT "\n", csId);	

			// Update freqCS value
			freq = (int *)Tloc(csBats->freqBat, csId);
			(*freq)++; 

			if (isStoreFreqCS == 1){	/* Store the frequent CS to the CSset*/
				//printf("FreqCS: Support = %d, Threshold %d  \n ", freq, freqThreshold);
				if (*freq == freqThreshold){
					#if STOREFULLCS
					freqCS = creatCS(csId, num, key, subjectId, buffObjs);		
					#else
					freqCS = creatCS(csId, num, key);			
					#endif
					addCStoSet(freqCSset, *freqCS);
				}
			}
		}
	}

	if (csId == BUN_NONE){
		printf("Not acceptable cdId " BUNFMT " \n", csId);
	}

	//assert(csId != BUN_NONE);

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
void getMaximumFreqCSs(CSset *freqCSset, oid* csSuperCSMap, int numCS){

	int 	numFreqCS = freqCSset->numCSadded; 
	int 	i, j; 
	int 	numMaxCSs = 0;

	oid 	tmpCSId; 

	printf("Retrieving maximum frequent CSs: \n");

	for (i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].isSubset == 1) continue;
		for (j = (i+1); j < numFreqCS; j++){
			if (isSubset(freqCSset->items[i].lstProp, freqCSset->items[j].lstProp,  
					freqCSset->items[i].numProp,freqCSset->items[j].numProp) == 1) { 
				/* CSj is a subset of CSi */
				freqCSset->items[j].isSubset = 1; 
				csSuperCSMap[freqCSset->items[j].csId] = freqCSset->items[i].csId;
			}
			else if (isSubset(freqCSset->items[j].lstProp, freqCSset->items[i].lstProp,  
					freqCSset->items[j].numProp,freqCSset->items[i].numProp) == 1) { 
				/* CSj is a subset of CSi */
				freqCSset->items[i].isSubset = 1; 
				csSuperCSMap[freqCSset->items[i].csId] = freqCSset->items[j].csId;
				break; 
			}
			
		} 
		/* By the end, if this CS is not a subset of any other CS */
		if (freqCSset->items[i].isSubset == 0){
			numMaxCSs++;
			csSuperCSMap[freqCSset->items[i].csId] = freqCSset->items[i].csId;
			//printCS( freqCSset->items[i]); 
		}
	}
	printf("Number of maximum CSs: %d / %d CSs \n", numMaxCSs, numCS);

	/*
	printf("CS - SuperCS before tunning ");
	for (i = 0; i < numCS; i++){
		if (csSuperCSMap[i] != BUN_NONE)
			printf("SuperCS[%d]=" BUNFMT " \n", i, csSuperCSMap[i]);
	}
	*/

	//Tunning
	for (i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].isSubset == 1){
			tmpCSId = freqCSset->items[i].csId; 
			while (csSuperCSMap[tmpCSId] != tmpCSId){
				tmpCSId = csSuperCSMap[tmpCSId];	// tracing to the maximum CS
			}

			//End. Update maximum CS for csSuperCSMap
			csSuperCSMap[freqCSset->items[i].csId] = tmpCSId; 
		}
	}

	/*
	printf("CS - SuperCS after tunning ");
	for (i = 0; i < numCS; i++){
		if (csSuperCSMap[i] != BUN_NONE)
			printf("SuperCS[%d]=" BUNFMT " \n", i, csSuperCSMap[i]);
	}
	*/
}




static void putPtoHash(map_t pmap, int key, oid *poid, int support){
	oid 	*getPoid; 
	oid	*putPoid; 
	int 	err; 
	int* 	pkey; 

	pkey = (int*) malloc(sizeof(int));

	*pkey = key; 

	if (hashmap_get_forP(pmap, pkey,(void**)(&getPoid)) != MAP_OK){
		putPoid = (oid*) malloc(sizeof(oid)); 
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


static void getStatisticCSsBySupports(BAT *pOffsetBat, BAT *freqBat, BAT *fullPBat, oid* csSuperCSMap, char isWriteToFile, int freqThreshold){

	//int 	*csPropNum; 
	//int	*csFreq; 
	FILE 	*fout; 
	oid 	*offset, *offset2; 
	int	numP; 
	BUN 	p, q; 
	BATiter	pi, freqi; 
	int	*freq; 
	char 	filename[100];
	char 	tmpStr[20];

	strcpy(filename, "csStatistic");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 
	fprintf(fout, " csId  #Prop   #frequency maxCSid\n"); 

	pi = bat_iterator(pOffsetBat);
	freqi = bat_iterator(freqBat);

	BATloop(pOffsetBat, p, q){
		offset = (oid *) BUNtloc(pi, p);		

		if ((p+1) != BUNlast(pOffsetBat)){
			offset2 = (oid *)BUNtloc(pi, p + 1);
			numP = *offset2 - *offset;
		}
		else	//Last element
			numP = BUNlast(fullPBat) - *offset;

		freq = (int *) BUNtloc(freqi, p); 


		// Output the result 
		if (isWriteToFile == 0)
			printf(BUNFMT "  %d  %d " BUNFMT "\n", p, numP, *freq, csSuperCSMap[p]); 
		else 
			fprintf(fout, BUNFMT " %d  %d " BUNFMT "\n", p, numP, *freq, csSuperCSMap[p]); 
	}

	fclose(fout); 
	//free(csPropNum); 
}


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

static 
CSBats* initCSBats(void){

	CSBats *csBats = (CSBats *) malloc(sizeof(CSBats));
	csBats->hsKeyBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);

	BATseqbase(csBats->hsKeyBat, 0);
	
	if (csBats->hsKeyBat == NULL) {
		return NULL; 
	}

	(void)BATprepareHash(BATmirror(csBats->hsKeyBat));
	if (!(csBats->hsKeyBat->T->hash)){
		return NULL;
	}

	csBats->hsValueBat = BATnew(TYPE_void, TYPE_int, smallbatsz);

	if (csBats->hsValueBat == NULL) {
		return NULL; 
	}
	csBats->pOffsetBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	
	if (csBats->pOffsetBat == NULL) {
		return NULL; 
	}
	csBats->fullPBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	
	if (csBats->fullPBat == NULL) {
		return NULL; 
	}
	csBats->freqBat = BATnew(TYPE_void, TYPE_int, smallbatsz);
	
	if (csBats->freqBat == NULL) {
		return NULL; 
	}

	return csBats; 
}



static 
void freeCSBats(CSBats *csBats){
	BBPreclaim(csBats->hsKeyBat); 
	BBPreclaim(csBats->hsValueBat); 
	BBPreclaim(csBats->freqBat); 
	BBPreclaim(csBats->pOffsetBat); 
	BBPreclaim(csBats->fullPBat); 

	free(csBats);

}

#if STOREFULLCS
static 
str RDFassignCSId(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi, CSset *freqCSset, int *freqThreshold, CSBats* csBats, oid *subjCSMap, oid *maxCSoid, int *maxNumProp, int *maxNumPwithDup){
#else
static 
str RDFassignCSId(int *ret, BAT *sbat, BATiter si, BATiter pi, CSset *freqCSset, int *freqThreshold, CSBats* csBats, oid *subjCSMap, oid *maxCSoid, int *maxNumProp, int *maxNumPwithDup){
#endif

	BUN 	p, q; 
	oid 	*sbt, *pbt; 
	oid 	curS; 		/* current Subject oid */
	oid 	curP; 		/* current Property oid */
	oid 	CSoid = 0; 	/* Characteristic set oid */
	int 	numP; 		/* Number of properties for current S */
	int 	numPwithDup = 0; 
	oid*	buff; 	 
	oid*	_tmp;
	int 	INIT_PROPERTY_NUM = 100; 
	oid 	returnCSid; 
	
	#if STOREFULLCS
	oid	*obt; 
	oid* 	buffObjs;
	oid* 	_tmpObjs; 
	#endif
	
	buff = (oid *) malloc (sizeof(oid) * INIT_PROPERTY_NUM);
	#if STOREFULLCS
	buffObjs =  (oid *) malloc (sizeof(oid) * INIT_PROPERTY_NUM);
	#endif	

	numP = 0;
	curP = 0; 
	curS = 0; 

	printf("freqThreshold = %d \n", *freqThreshold);	
	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				#if STOREFULLCS
				returnCSid = putaCStoHash(csBats, buff, numP, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs); 
				#else
				returnCSid = putaCStoHash(csBats, buff, numP, &CSoid, 1, *freqThreshold, freqCSset); 
				#endif

				subjCSMap[curS] = returnCSid; 			
				//printf("subjCSMap[" BUNFMT "]=" BUNFMT " (CSoid = " BUNFMT ") \n", curS, returnCSid, CSoid);

				if (numP > *maxNumProp) 
					*maxNumProp = numP; 
				if (numPwithDup > *maxNumPwithDup)
					*maxNumPwithDup = numPwithDup; 
				if (returnCSid > *maxCSoid)
					*maxCSoid = returnCSid; 
				 
			}
			curS = *sbt; 
			curP = 0;
			numP = 0;
			numPwithDup = 0; 
		}
				
		pbt = (oid *) BUNtloc(pi, p); 

		if (INIT_PROPERTY_NUM <= numP){
			//throw(MAL, "rdf.RDFextractCS", "# of properties is greater than INIT_PROPERTY_NUM");
			//exit(-1);
			INIT_PROPERTY_NUM = INIT_PROPERTY_NUM * 2; 
                	_tmp = realloc(buff, (INIT_PROPERTY_NUM * sizeof(oid)));
	                if (!_tmp){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			buff = (oid*)_tmp;

			#if STOREFULLCS	
                	_tmpObjs = realloc(buffObjs, (INIT_PROPERTY_NUM * sizeof(oid)));
	                if (!_tmpObjs){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			buffObjs = (oid*)_tmpObjs; 

			#endif
		
		}

		
		if (curP != *pbt){	/* Multi values property */		
			buff[numP] = *pbt; 
			#if STOREFULLCS
			obt = (oid *) BUNtloc(oi, p); 
			buffObjs[numP] = *obt; 
			#endif
			numP++; 
			curP = *pbt; 

		}

		numPwithDup++;
		
	}
	
	/*put the last CS */
	#if STOREFULLCS
	returnCSid = putaCStoHash(csBats, buff, numP, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs); 
	#else
	returnCSid = putaCStoHash(csBats, buff, numP, &CSoid, 1, *freqThreshold, freqCSset ); 
	#endif
	
	subjCSMap[curS] = returnCSid; 			
	//printf("subjCSMap[" BUNFMT "]=" BUNFMT " (CSoid = " BUNFMT ") \n", curS, returnCSid, CSoid);

	if (numP > *maxNumProp) 
		*maxNumProp = numP; 

	if (numPwithDup > *maxNumPwithDup)
		*maxNumPwithDup = numPwithDup; 

	if (returnCSid > *maxCSoid)
		*maxCSoid = returnCSid; 

	free (buff); 
	#if STOREFULLCS
	free (buffObjs); 
	#endif
		
	*ret = 1; 

	return MAL_SUCCEED; 
}

static 
str RDFrelationships(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi,  
		oid *subjCSMap, oid *subjSubCSMap, SubCSSet *csSubCSMap, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup){

	BUN	 	p, q; 
	oid 		*sbt, *obt, *pbt; 
	oid 		curS; 		/* current Subject oid */
	//oid 		CSoid = 0; 	/* Characteristic set oid */
	int 		numPwithDup;	/* Number of properties for current S */
	char 		objType;
	oid 		returnSubCSid; 
	char* 		buffTypes; 

	buffTypes = (char *) malloc(sizeof(char) * (maxNumPwithDup + 1)); 

	numPwithDup = 0;
	curS = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[*sbt], csSubCSMap);
				//Get the subCSId
				subjSubCSMap[*sbt] = returnSubCSid; 

			}
			curS = *sbt; 
			numPwithDup = 0;
		}
				
		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = (char) ((*obt) >> (sizeof(BUN)*8 - 3))  &  3 ;	/* Get two bits 63th, 62nd from object oid */

		buffTypes[numPwithDup] = objType; 
		numPwithDup++; 
		
		/* Look at sbat*/
		if (objType == URI){
			pbt = (oid *) BUNtloc(pi, p); 
			if (*obt <= maxSoid && subjCSMap[*obt] != BUN_NONE){
				////printf(" Subject " BUNFMT " refer to CS " BUNFMT " \n",*sbt, subjCSMap[*obt]);
				addReltoCSRel(subjCSMap[*sbt], subjCSMap[*obt], *pbt, &csrelSet[subjCSMap[*sbt]]);
			}
		}
	}
	
	/* Check for the last CS */
	returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[*sbt], csSubCSMap);
	subjSubCSMap[*sbt] = returnSubCSid; 

	free (buffTypes); 



	*ret = 1; 

	return MAL_SUCCEED; 
}

/* Extract CS from SPO triples table */
str
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold){

	BAT 		*sbat = NULL, *pbat = NULL, *obat = NULL, *mbat = NULL; 
	BATiter 	si, pi, oi; 	/*iterator for BAT of s,p,o columns in spo table */
	CSset		*freqCSset; 	/* Set of frequent CSs */

	CSBats		*csBats; 
	oid		*subjCSMap; 	/* Store the corresponding CS Id for each subject */
	oid		*subjSubCSMap;  /* Store the corresponding CS sub Id for each subject */
	BUN		*maxSoid; 	
	oid 		maxCSoid = 0; 
	int		maxNumProp = 0;
	int		maxNumPwithDup = 0; 
	char		*csFreqMap; 
	CSrel   	*csrelSet;
	CSrel		*csrelToMaxFreqSet, *csrelFromMaxFreqSet;
	CSrel		*csrelBetweenMaxFreqSet; 
	SubCSSet 	*csSubCSMap; 
	oid		*csSuperCSMap;  

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
	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}

	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat);

	csBats = initCSBats();

	freqCSset = initCSset();

	maxSoid = (BUN *) Tloc(sbat, BUNlast(sbat) - 1);
	printf("Max S oid: " BUNFMT "\n", *maxSoid);

	assert(*maxSoid != BUN_NONE); 

	subjCSMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	subjSubCSMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	
	initArray(subjCSMap, (*maxSoid) + 1, BUN_NONE);


	//Phase 1: Assign an ID for each CS
	#if STOREFULLCS
	RDFassignCSId(ret, sbat, si, pi, oi, freqCSset, freqThreshold, csBats, subjCSMap, &maxCSoid, &maxNumProp, &maxNumPwithDup);
	#else
	RDFassignCSId(ret, sbat, si, pi, freqCSset, freqThreshold, csBats, subjCSMap, &maxCSoid, &maxNumProp, &maxNumPwithDup);
	#endif



	//Phase 2: Check the relationship	

	printf("Max CS oid: " BUNFMT "\n", maxCSoid);

	printf("Max Number of P (considering duplicated P): %d \n", maxNumPwithDup);

	csFreqMap = (char*) malloc(sizeof(char) * (maxCSoid +1)); 
	initCharArray(csFreqMap, maxCSoid +1, 0); 

	csSuperCSMap = (oid*) malloc(sizeof(oid) * (maxCSoid + 1));
	initArray(csSuperCSMap, maxCSoid + 1, BUN_NONE);


	generateFreqCSMap(freqCSset,csFreqMap); 


	csrelSet = initCSrelset(maxCSoid + 1);


	csSubCSMap = initCS_SubCSMap(maxCSoid +1); 

	RDFrelationships(ret, sbat, si, pi, oi, subjCSMap, subjSubCSMap, csSubCSMap, csrelSet, *maxSoid, maxNumPwithDup);


	printCSrelSet(csrelSet,csFreqMap, csBats->freqBat, maxCSoid + 1, 1, *freqThreshold);  

	printSubCSInformation(csSubCSMap, maxCSoid + 1, 1, *freqThreshold); 

	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	/*get the statistic */

	//getTopFreqCSs(csMap,*freqThreshold);

	getMaximumFreqCSs(freqCSset, csSuperCSMap, maxCSoid + 1); 

	printFreqCSSet(freqCSset, csSuperCSMap, csBats->freqBat, mbat, 1, *freqThreshold); 

	csrelToMaxFreqSet = initCSrelset(maxCSoid + 1);	// CS --> Reference MaxCSs
	csrelFromMaxFreqSet = initCSrelset(maxCSoid + 1);	// CS --> Reference MaxCSs
	csrelBetweenMaxFreqSet = initCSrelset(maxCSoid + 1);	// MaxCS --> Reference MaxCSs

	printCSrelWithMaxSet(csSuperCSMap, csrelToMaxFreqSet, csrelFromMaxFreqSet, csrelBetweenMaxFreqSet, csrelSet,csFreqMap, csBats->freqBat, maxCSoid + 1, *freqThreshold);  


	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csBats->pOffsetBat, csBats->freqBat, csBats->fullPBat, csSuperCSMap, 1, *freqThreshold);

	BBPreclaim(sbat); 
	BBPreclaim(pbat); 
	BBPreclaim(obat);

	free (subjCSMap); 
	free (subjSubCSMap);
	free (csFreqMap);
	free (csSuperCSMap);

	freeCS_SubCSMapSet(csSubCSMap, maxCSoid + 1); 

	freeCSrelSet(csrelSet, maxCSoid + 1); 
	freeCSrelSet(csrelToMaxFreqSet, maxCSoid + 1); 
	freeCSrelSet(csrelBetweenMaxFreqSet, maxCSoid + 1);  

	freeCSBats(csBats);

	freeCSset(freqCSset); 

	//testBatHash(); 

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
