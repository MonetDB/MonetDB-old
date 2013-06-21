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
#include "rdflabels.h"
#include "algebra.h"
#include <gdk.h>
#include <hashmap/hashmap.h>
#include "tokenizer.h"
#include <math.h>

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



static void initcsIdFreqIdxMap(int* inputArr, int num, int defaultValue, CSset *freqCSset){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}

	for (i = 0; i < freqCSset->numCSadded; i++){
		inputArr[freqCSset->items[i].csId] = i; 
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

/*
static 
void addmergeCStoSet(mergeCSset *mergecsSet, mergeCS item)
{
	void *_tmp; 
	if(mergecsSet->nummergeCSadded == mergecsSet->numAllocation) 
	{ 
		mergecsSet->numAllocation += INIT_NUM_CS; 
		
		_tmp = realloc(mergecsSet->items, (mergecsSet->numAllocation * sizeof(CS)));
	
		if (!_tmp){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		}
		mergecsSet->items = (mergeCS*)_tmp;
	}
	mergecsSet->items[mergecsSet->nummergeCSadded] = item;
	mergecsSet->nummergeCSadded++;
}
*/

static 
CSrel* creataCSrel(oid csoid){
	CSrel *csrel = (CSrel*) malloc(sizeof(CSrel));
	csrel->origCSoid = csoid; 
	csrel->lstRefCSoid = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);
	csrel->lstPropId = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);
	csrel->lstCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);		
	csrel->lstBlankCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);		
	csrel->numRef = 0;
	csrel->numAllocation = INIT_NUM_CSREL;

	return csrel; 
}


static 
void addReltoCSRel(oid origCSoid, oid refCSoid, oid propId, CSrel *csrel, char isBlankNode)
{
	void *_tmp; 
	void *_tmp1; 
	void *_tmp2;
	void *_tmp3; 

	int i = 0; 

	assert (origCSoid == csrel->origCSoid);
#ifdef NDEBUG
	/* parameter origCSoid is not used other in about assertion */
	(void) origCSoid;
#endif

	while (i < csrel->numRef){
		if (refCSoid == csrel->lstRefCSoid[i] && propId == csrel->lstPropId[i]){
			//Existing
			break; 
		}
		i++;
	}
	
	if (i != csrel->numRef){ 
		csrel->lstCnt[i]++; 
		csrel->lstBlankCnt[i] += (int) isBlankNode; 
		return; 
	}
	else{	// New Ref
	
		if(csrel->numRef == csrel->numAllocation) 
		{ 
			csrel->numAllocation += INIT_NUM_CSREL; 
			
			_tmp = realloc(csrel->lstRefCSoid, (csrel->numAllocation * sizeof(oid)));
			_tmp1 = realloc(csrel->lstPropId, (csrel->numAllocation * sizeof(oid)));
			_tmp2 = realloc(csrel->lstCnt, (csrel->numAllocation * sizeof(int)));
			_tmp3 = realloc(csrel->lstBlankCnt, (csrel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2 || !_tmp3){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csrel->lstRefCSoid = (oid*)_tmp;
			csrel->lstPropId = (oid*)_tmp1; 
			csrel->lstCnt = (int*)_tmp2; 
			csrel->lstBlankCnt = (int*)_tmp3; 
		}

		csrel->lstRefCSoid[csrel->numRef] = refCSoid;
		csrel->lstPropId[csrel->numRef] = propId;
		csrel->lstCnt[csrel->numRef] = 1; 
		csrel->lstBlankCnt[csrel->numRef] = (int) isBlankNode; 
		csrel->numRef++;
	}
}


static 
void addReltoCSRelWithFreq(oid origCSoid, oid refCSoid, oid propId, int freq, int numBlank, CSrel *csrel)
{
	void *_tmp; 
	void *_tmp1; 
	void *_tmp2; 
	void *_tmp3; 

	int i = 0; 

	assert (origCSoid == csrel->origCSoid);
#ifdef NDEBUG
	/* parameter origCSoid is not used other in about assertion */
	(void) origCSoid;
#endif

	while (i < csrel->numRef){
		if (refCSoid == csrel->lstRefCSoid[i] && propId == csrel->lstPropId[i]){
			//Existing
			break; 
		}
		i++;
	}
	
	if (i != csrel->numRef){ 
		csrel->lstCnt[i] = csrel->lstCnt[i] + freq; 
		csrel->lstBlankCnt[i] = csrel->lstBlankCnt[i] + numBlank; 
		return; 
	}
	else{	// New Ref
	
		if(csrel->numRef == csrel->numAllocation) 
		{ 
			csrel->numAllocation += INIT_NUM_CSREL; 
			
			_tmp = realloc(csrel->lstRefCSoid, (csrel->numAllocation * sizeof(oid)));
			_tmp1 = realloc(csrel->lstPropId, (csrel->numAllocation * sizeof(oid)));		
			_tmp2 = realloc(csrel->lstCnt, (csrel->numAllocation * sizeof(int)));
			_tmp3 = realloc(csrel->lstBlankCnt, (csrel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2 || !_tmp3){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csrel->lstRefCSoid = (oid*)_tmp;
			csrel->lstPropId = (oid*)_tmp1; 
			csrel->lstCnt = (int*)_tmp2; 
			csrel->lstBlankCnt = (int*)_tmp3; 
		}

		csrel->lstRefCSoid[csrel->numRef] = refCSoid;
		csrel->lstPropId[csrel->numRef] = propId;
		csrel->lstCnt[csrel->numRef] = freq; 
		csrel->lstBlankCnt[csrel->numRef] = numBlank; 
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

static 
oid getMaxCSIdFromCSId(oid csId, int* csIdFreqIdxMap, CSset *freqCSset){
	
	int freqIdx; 
	oid maxCSoid; 

	freqIdx = csIdFreqIdxMap[csId];
	if (freqIdx != -1){ //A freqCS
		if (freqCSset->items[freqIdx].type == MAXCS){
			maxCSoid = freqCSset->items[freqIdx].csId; 
		}
		else 
			maxCSoid = freqCSset->items[freqCSset->items[freqIdx].parentFreqIdx].csId;  
	}
	else{
		maxCSoid = BUN_NONE;  
	}

	return maxCSoid; 
}

/*
 * Show the relationship from each CS to maximumFreqCSs
 * */


static 
str printCSrelWithMaxSet(CSset *freqCSset, int* csIdFreqIdxMap, CSrel *csrelToMaxSet, CSrel *csrelFromMaxSet, CSrel *csrelBetweenMaxSet, CSrel *csrelSet, char *csFreqMap, BAT* freqBat, int num, int freqThreshold){

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
		maxCSoid = getMaxCSIdFromCSId(csrelSet[i].origCSoid, csIdFreqIdxMap,freqCSset); 
		if (csrelSet[i].numRef != 0){
			for (j = 0; j < csrelSet[i].numRef; j++){		
				if (getMaxCSIdFromCSId(csrelSet[i].lstRefCSoid[j],csIdFreqIdxMap,freqCSset) != BUN_NONE){
					addReltoCSRelWithFreq(csrelSet[i].origCSoid, getMaxCSIdFromCSId(csrelSet[i].lstRefCSoid[j], csIdFreqIdxMap,freqCSset), csrelSet[i].lstPropId[j], csrelSet[i].lstCnt[j], csrelSet[i].lstBlankCnt[j], &csrelToMaxSet[i]);
				}
			}


			// Add to csrelFromMaxSet
			// For a referenced CS that is frequent, use its maxCSoid
			// Else, use its csoid
			if (maxCSoid != BUN_NONE){
				for (j = 0; j < csrelSet[i].numRef; j++){		
					if (getMaxCSIdFromCSId(csrelSet[i].lstRefCSoid[j], csIdFreqIdxMap,freqCSset) != BUN_NONE){
						addReltoCSRelWithFreq(maxCSoid, getMaxCSIdFromCSId(csrelSet[i].lstRefCSoid[j], csIdFreqIdxMap,freqCSset), csrelSet[i].lstPropId[j], csrelSet[i].lstCnt[j],csrelSet[i].lstBlankCnt[j], &csrelFromMaxSet[maxCSoid]);
					}
					else{
						addReltoCSRelWithFreq(maxCSoid, csrelSet[i].lstRefCSoid[j], csrelSet[i].lstPropId[j], csrelSet[i].lstCnt[j], csrelSet[i].lstBlankCnt[j], &csrelFromMaxSet[maxCSoid]);
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


	strcpy(filename2, "csRelationshipBetweenMaxFreqCS");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename2, tmpStr);
	strcat(filename2, ".txt");

	fout2 = fopen(filename2,"wt"); 
	strcat(filename2, ".filter");
	fout2filter = fopen(filename2,"wt");

	// Merge the csrelToMaxSet --> csrelBetweenMaxSet
	for (i = 0; i < num; i++){
		maxCSoid = getMaxCSIdFromCSId(csrelToMaxSet[i].origCSoid, csIdFreqIdxMap,freqCSset);
		if (csrelToMaxSet[i].numRef != 0 && maxCSoid != BUN_NONE){
			for (j = 0; j < csrelToMaxSet[i].numRef; j++){		
				assert(getMaxCSIdFromCSId(csrelToMaxSet[i].lstRefCSoid[j], csIdFreqIdxMap,freqCSset) == csrelToMaxSet[i].lstRefCSoid[j]);
				addReltoCSRelWithFreq(maxCSoid, getMaxCSIdFromCSId(csrelToMaxSet[i].lstRefCSoid[j], csIdFreqIdxMap,freqCSset), csrelToMaxSet[i].lstPropId[j], csrelToMaxSet[i].lstCnt[j],csrelToMaxSet[i].lstBlankCnt[j], &csrelBetweenMaxSet[maxCSoid]);
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
				fprintf(fout2, BUNFMT "(P:" BUNFMT " - %s) (%d)(Blank:%d) ", csrelBetweenMaxSet[i].lstRefCSoid[j],csrelBetweenMaxSet[i].lstPropId[j], propStr, csrelBetweenMaxSet[i].lstCnt[j], csrelBetweenMaxSet[i].lstBlankCnt[j]);	
				#else
				fprintf(fout2, BUNFMT "(P:" BUNFMT ") (%d)(Blank:%d) ", csrelBetweenMaxSet[i].lstRefCSoid[j],csrelBetweenMaxSet[i].lstPropId[j], csrelBetweenMaxSet[i].lstCnt[j], csrelBetweenMaxSet[i].lstBlankCnt[j]);	
				#endif

				if (*freq < csrelBetweenMaxSet[i].lstCnt[j]*100){
					fprintf(fout2filter, BUNFMT "(P:" BUNFMT ") (%d)(Blank:%d) ", csrelBetweenMaxSet[i].lstRefCSoid[j],csrelBetweenMaxSet[i].lstPropId[j], csrelBetweenMaxSet[i].lstCnt[j], csrelBetweenMaxSet[i].lstBlankCnt[j]);	
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
void printSubCSInformation(SubCSSet *subcsset, BAT* freqBat, int num, char isWriteTofile, int freqThreshold){

	int i; 
	int j; 
	int *freq; 
	int numSubCSFilter; 

	FILE 	*fout, *foutfreq, *foutfreqfilter; 
	char 	filename[100], filenamefreq[100], filenamefreqfilter[100];
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
		strcpy(filenamefreq, filename); 
		strcpy(filenamefreqfilter, filename); 
		strcat(filename, ".txt");
		strcat(filenamefreq,"_freq.txt");
		strcat(filenamefreqfilter,"_freqfilter.txt");

		fout = fopen(filename,"wt"); 
		foutfreq = fopen(filenamefreq,"wt");
		foutfreqfilter = fopen(filenamefreqfilter,"wt");
		fprintf(foutfreq, "csId	#SubCS \n");
		fprintf(foutfreqfilter, "csId	#FrequentSubCS  \n");

		for (i = 0; i < num; i++){
			if (subcsset[i].numSubCS != 0){	
				freq  = (int *) Tloc(freqBat, i);
				fprintf(fout, "CS " BUNFMT ": ", subcsset[i].csId);
					
				if (*freq > freqThreshold){
					fprintf(foutfreq, BUNFMT "  ", subcsset[i].csId);
					fprintf(foutfreqfilter, BUNFMT "  ", subcsset[i].csId);
				}
				numSubCSFilter = 0;
				for (j = 0; j < subcsset[i].numSubCS; j++){
					fprintf(fout, BUNFMT " (%d) ", subcsset[i].subCSs[j].subCSId, subcsset[i].freq[j]);	
					
					// Check frequent subCS which appears in > 1% 
					if (*freq <  subcsset[i].freq[j]*10){
						numSubCSFilter++;
					}
				}	
				if (*freq > freqThreshold){
					fprintf(foutfreq, "%d \n", subcsset[i].numSubCS);
					fprintf(foutfreqfilter, "%d \n", numSubCSFilter);
				}
				fprintf(fout, "\n");
			}
		}

		fclose(fout);
		fclose(foutfreq);
		fclose(foutfreqfilter);
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

/*
static
void freemergeCSset(mergeCSset *csSet){
	int i;
	for(i = 0; i < csSet->nummergeCSadded; i ++){
		free(csSet->items[i].lstProp);
		free(csSet->items[i].lstConsistsOf);
	}
	free(csSet->items);
	free(csSet);	
}
*/


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
mergeCSset* initmergeCSset(void){
	mergeCSset *mergecsSet = (mergeCSset*) malloc(sizeof(mergeCSset)); 
	mergecsSet->items = (mergeCS*) malloc(sizeof(mergeCS) * INIT_NUM_CS); 
	mergecsSet->numAllocation = INIT_NUM_CS;
	mergecsSet->nummergeCSadded = 0;

	return mergecsSet;
}

*/

/*
static 
void freeCS(CS *cs){
	free(cs->lstProp);
	free(cs);
}
*/
#if STOREFULLCS
static
CS* creatCS(oid csId, int numP, oid* buff, oid subjectId, oid* lstObject, char type, int parentfreqIdx, int support, int coverage)
#else
static 
CS* creatCS(oid csId, int numP, oid* buff, char type,  int parentfreqIdx, int support, int coverage)
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
	/*By default, this CS is not known to be a subset of any other CS*/
	#if STOREFULLCS
	cs->lstObj =  (oid*) malloc(sizeof(oid) * numP);
	if (cs->lstObj == NULL){
		printf("Malloc failed. at %d", numP);
		exit(-1); 
	}
	copyOidSet(cs->lstObj, lstObject, numP); 
	cs->subject = subjectId; 
	//printf("Create a CS with subjectId: " BUNFMT "\n", subjectId);
	#endif

	cs->type = type; 

	// This value is set for the 
	cs->parentFreqIdx = parentfreqIdx; 
	cs->support = support;
	cs->coverage = coverage; 

	return cs; 
}

static 
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
	/*
	printf("pos = %d, numCombineP = %d\n", pos, numCombineP);

	for (i = 0; i < m; i++){
		printf(BUNFMT " ", arr1[i]);
	}
	
	printf("\n");
	for (i = 0; i < n; i++){
		printf(BUNFMT " ", arr2[i]);
	}

	
	printf("\n");
	for (i = 0; i < pos; i++){
		printf(BUNFMT " ", mergeArr[i]);
	}
	
	printf("\n");
	*/

		
}

static 
CS* mergeTwoCSs(CS cs1, CS cs2, int freqIdx1, int freqIdx2, oid mergeCSId){
	
	int numCombineP; 
	CS *mergecs = (CS*) malloc (sizeof (CS)); 
	mergecs->type = MERGECS; 
	mergecs->numConsistsOf = 2; 
	mergecs->lstConsistsOf = (int*) malloc(sizeof(int) * 2);

	//mergecs->lstConsistsOf[0] = cs1->csId;  
	//mergecs->lstConsistsOf[1] = cs2->csId; 

	mergecs->lstConsistsOf[0] = freqIdx1;  
	mergecs->lstConsistsOf[1] = freqIdx2; 
	
	mergecs->lstProp = (oid*) malloc(sizeof(oid) * (cs1.numProp + cs2.numProp));  // will be redundant

	if (mergecs->lstProp == NULL){
		printf("Malloc failed. at %d", numCombineP);
		exit(-1);
	}

	mergeOidSets(cs1.lstProp, cs2.lstProp, mergecs->lstProp, cs1.numProp, cs2.numProp, &numCombineP); 

	mergecs->numProp = numCombineP;
	mergecs->support = cs1.support + cs2.support;
	mergecs->coverage = cs1.coverage + cs2.coverage;
	mergecs->parentFreqIdx = -1; 
	mergecs->csId = mergeCSId; 
	
	return mergecs; 

}


static 
void mergeACStoExistingmergeCS(CS cs, int freqIdx , CS *mergecs){
	
	int numCombineP; 
	oid* _tmp1; 
	oid* _tmp2; 
	oid* oldlstProp; 

        _tmp1 = realloc(mergecs->lstConsistsOf, ((mergecs->numConsistsOf + 1) * sizeof(int)));

	if (!_tmp1){
		fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
	}
	mergecs->lstConsistsOf = (int*)_tmp1;
	//mergecs->lstConsistsOf[mergecs->numConsistsOf] = cs.csId; 
	mergecs->lstConsistsOf[mergecs->numConsistsOf] = freqIdx; 
	mergecs->numConsistsOf++;

	oldlstProp = malloc (sizeof(oid) * (mergecs->numProp)); 
	memcpy(oldlstProp, mergecs->lstProp, (mergecs->numProp) * sizeof(oid));
	
        _tmp2 = realloc(mergecs->lstProp, ((mergecs->numProp + cs.numProp) * sizeof(oid)));

	if (!_tmp2){
		fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
	}
	mergecs->lstProp = (oid*)_tmp2;

	mergeOidSets(cs.lstProp, oldlstProp, mergecs->lstProp, cs.numProp, mergecs->numProp, &numCombineP); 

	mergecs->numProp = numCombineP;
	mergecs->support += cs.support;
	mergecs->coverage += cs.coverage;

	free(oldlstProp);
}


/*Merge two mergeCSs with the condition that no parent belongs to both of them */
static 
void mergeTwomergeCS(CS *mergecs1, CS *mergecs2, int parentFreqIdx){
	
	int numCombineP; 
	int* _tmp1; 
	oid* _tmp2; 
	oid* oldlstProp1; 
	oid* oldlstProp2; 
	int i; 

        _tmp1 = realloc(mergecs1->lstConsistsOf, ((mergecs1->numConsistsOf + mergecs2->numConsistsOf) * sizeof(int)));

	if (!_tmp1){
		fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
	}
	mergecs1->lstConsistsOf = (int*)_tmp1;
	for (i = 0; i < mergecs2->numConsistsOf; i++){
		mergecs1->lstConsistsOf[mergecs1->numConsistsOf] = mergecs2->lstConsistsOf[i]; 
		mergecs1->numConsistsOf++;
	}


	oldlstProp1 = malloc (sizeof(oid) * mergecs1->numProp); 
	memcpy(oldlstProp1, mergecs1->lstProp, (mergecs1->numProp) * sizeof(oid));
	
	oldlstProp2 = malloc (sizeof(oid) * mergecs2->numProp); 
	memcpy(oldlstProp2, mergecs2->lstProp, (mergecs2->numProp) * sizeof(oid));

        _tmp2 = realloc(mergecs1->lstProp, ((mergecs1->numProp + mergecs2->numProp) * sizeof(oid)));

	if (!_tmp2){
		fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
	}
	mergecs1->lstProp = (oid*)_tmp2;

	mergeOidSets(oldlstProp1, oldlstProp2, mergecs1->lstProp, mergecs1->numProp, mergecs2->numProp, &numCombineP); 

	mergecs1->numProp = numCombineP;
	mergecs1->support += mergecs2->support;
	mergecs1->coverage += mergecs2->coverage;

	// Remove mergecs2
	mergecs2->parentFreqIdx = parentFreqIdx; 

	free(oldlstProp1);
	free(oldlstProp2); 
}

static 
str printFreqCSSet(CSset *freqCSset, BAT *freqBat, BAT *mapbat, char isWriteTofile, int freqThreshold){

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

			printf("CS " BUNFMT " (Freq: %d) | Subject: %s  |  Parent " BUNFMT " \n", cs.csId, *freq, subStr, freqCSset->items[cs.parentFreqIdx].csId);
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
			

			takeOid(cs.subject, &subStr);	
			
			fprintf(fout,"CS " BUNFMT " (Freq: %d) | Subject: %s  | FreqParentIdx %d \n", cs.csId, *freq, subStr, cs.parentFreqIdx);

			// Filter max freq cs set
			if (cs.type == MAXCS){
				fprintf(fout2,"CS " BUNFMT " (Freq: %d) | Subject: %s  | Parent " BUNFMT " \n", cs.csId, *freq, subStr, cs.csId);
			}

			for (j = 0; j < cs.numProp; j++){
				takeOid(cs.lstProp[j], &propStr);
				//fprintf(fout, "  P:" BUNFMT " --> ", cs.lstProp[j]);	
				fprintf(fout, "  P:%s --> ", propStr);	
				if (cs.type == MAXCS){
					fprintf(fout2, "  P:%s --> ", propStr);
				}

				// Get object value
				objOid = cs.lstObj[j]; 

				objType = (char) (objOid >> (sizeof(BUN)*8 - 4))  &  7 ; 

				if (objType == URI || objType == BLANKNODE){
					objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
					takeOid(objOid, &objStr); 
				}
				else{
					objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */ 
					bun = BUNfirst(mapbat);
					objStr = (str) BUNtail(mapi, bun + objOid); 
				}

				fprintf(fout, "  O: %s \n", objStr);
				if (cs.type == MAXCS){
					fprintf(fout2, "  O: %s \n", objStr);
				}


			}	
			fprintf(fout, "\n");
			if (cs.type == MAXCS){
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
static 
str printamergeCS(mergeCS cs, int mergecsid, CSset *freqCSset, oid* superCSFreqCSMap){
	int ret; 
	char*   schema = "rdf";
	int j; 
	CS freqcs; 
	str propStr; 

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

	printf("MergeCS %d - (numConsistsOf: %d) \n",mergecsid, cs.numConsistsOf);
	for (j = 0; j < cs.numConsistsOf; j++){
		freqcs = freqCSset->items[superCSFreqCSMap[cs.lstConsistsOf[j]]];
		printf(" " BUNFMT " ", freqcs.csId);
	}
	printf("\n");
	for (j = 0; j < cs.numProp; j++){
		takeOid(cs.lstProp[j], &propStr);	
		printf("          %s\n", propStr);
	}
	printf("\n");


	TKNZRclose(&ret);
	return MAL_SUCCEED;
}
*/

static 
str printmergeCSSet(CSset *freqCSset, int freqThreshold){

	int 	i,j; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];
	int 	ret;

	str 	propStr; 
	char*   schema = "rdf";
	int	nummergecs;	
	CS	freqcs; 

	nummergecs = freqCSset->numCSadded; 

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	

	strcpy(filename, "mergeCSFullInfo");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	for (i = freqCSset->numOrigFreqCS; i < nummergecs; i++){
		CS cs = (CS)freqCSset->items[i];
		if (cs.parentFreqIdx == -1){
			fprintf(fout, "MergeCS %d (Number of parent: %d) \n",i, cs.numConsistsOf);
			for (j = 0; j < cs.numConsistsOf; j++){
				freqcs = freqCSset->items[cs.lstConsistsOf[j]];
				fprintf(fout, " " BUNFMT " ", freqcs.csId);
			}
			fprintf(fout, "\n");
			for (j = 0; j < cs.numProp; j++){
				takeOid(cs.lstProp[j], &propStr);	
				fprintf(fout,"          %s\n", propStr);
			}
			fprintf(fout, "\n");
		}
	}

	fclose(fout);
	
	TKNZRclose(&ret);
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
void addNewCS(CSBats *csBats, BUN* csKey, oid* key, oid *csoid, int num, int numTriples){
	int freq = 1; 
	int coverage = numTriples; 
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
	BUNappend(csBats->coverageBat, &coverage, TRUE); 
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
oid putaCStoHash(CSBats *csBats, oid* key, int num, int numTriples,  
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset, oid subjectId, oid* buffObjs)
#else
static 
oid putaCStoHash(CSBats *csBats, oid* key, int num, int numTriples, 
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset)
#endif	
{
	BUN 	csKey; 
	int 	*freq; 
	oid	*coverage; 	//Total number of triples coverred by this CS
	CS	*freqCS; 
	BUN	bun; 
	oid	csId; 		/* Id of the characteristic set */
	char	isDuplicate = 0; 

	csKey = RDF_hash_oidlist(key, num);
	bun = BUNfnd(BATmirror(csBats->hsKeyBat),(ptr) &csKey);
	if (bun == BUN_NONE) {
		csId = *csoid; 
		addNewCS(csBats, &csKey, key, csoid, num, numTriples);
		
		//Handle the case when freqThreshold == 1 
		if (isStoreFreqCS ==1 && freqThreshold == 1){
			#if STOREFULLCS
			freqCS = creatCS(csId, num, key, subjectId, buffObjs, FREQCS, -1, 0,0);		
			#else
			freqCS = creatCS(csId, num, key, FREQCS,-1,0,0);			
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
			addNewCS(csBats, &csKey, key, csoid, num, numTriples);
			
			//Handle the case when freqThreshold == 1 
			if (isStoreFreqCS ==1 && freqThreshold == 1){
				
				#if STOREFULLCS
				freqCS = creatCS(csId, num, key, subjectId, buffObjs, FREQCS,-1,0,0);		
				#else
				freqCS = creatCS(csId, num, key, FREQCS,-1,0,0);			
				#endif
				addCStoSet(freqCSset, *freqCS);
			}

		}
		else{
			//printf(" Duplication (existed CS) at csId = " BUNFMT "\n", csId);	

			// Update freqCS value
			freq = (int *)Tloc(csBats->freqBat, csId);
			(*freq)++; 
			// Update number of coverred triples
			coverage = (oid *)Tloc(csBats->coverageBat, csId); 
			(*coverage) += numTriples;

			if (isStoreFreqCS == 1){	/* Store the frequent CS to the CSset*/
				//printf("FreqCS: Support = %d, Threshold %d  \n ", freq, freqThreshold);
				if (*freq == freqThreshold){
					#if STOREFULLCS
					freqCS = creatCS(csId, num, key, subjectId, buffObjs, FREQCS,-1,0,0);		
					#else
					freqCS = creatCS(csId, num, key, FREQCS,-1,0,0);			
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
 * Using TF-IDF for calculating the similarity score
 * See http://disi.unitn.it/~bernardi/Courses/DL/Slides_11_12/measures.pdf
 * tf(t,d): Number of times t occurs in d. --> For a CS, tf(prop, aCS) = 1; 
 * idf(t): The rarity of a term t in the whold document collection
 * idf(t) = log(#totalNumOfCSs / #numberCSs_containing_t +1)
 * tf-idf(t,d,D) = tf(t,d) * idf(t,D)
 *
 * Note that: If we use normalize tf by dividing with maximum tf 
 * in each CS, we still get the value 1. 
 * */

static 
float tfidfComp(int numContainedCSs, int totalNumCSs){
	return log((float)totalNumCSs/(1+numContainedCSs)); 
}

/*
 * Use Jaccard similarity coefficient for computing the 
 * similarity between two sets
 * sim(A,B) = |A  B| / |A U B|
 * Here each set contains distinct values only 
 * */

static 
float similarityScore(oid* arr1, oid* arr2, int m, int n, int *numCombineP){
	
	int i = 0, j = 0;
	int numOverlap = 0; 
	 
	while( i < n && j < m )
	{
		if( arr1[j] < arr2[i] )
			j++;
		else if( arr1[j] == arr2[i] )
		{
			j++;
			i++;
			numOverlap++;
		}
		else if( arr1[j] > arr2[i] )
			i++;
	}

	*numCombineP = m + n - numOverlap;
		
	return  ((float)numOverlap / (*numCombineP));
}


/*Using cosine similarity score with vector of tf-idfs for properties in each CS */
static 
float similarityScoreTFIDF(oid* arr1, oid* arr2, int m, int n, int *numCombineP, PropStat* propStat){
	
	int i = 0, j = 0;
	int numOverlap = 0; 
	float sumX2 = 0.0; 
	float sumY2 = 0.0;
	float sumXY = 0.0;
	BUN bun; 
	BUN	p; 
	float 	tfidfV; 

	// Get tf-idfs 
	float *tfidf1 = (float *)malloc(sizeof(float) * m) ;
	float *tfidf2 = (float *)malloc(sizeof(float) * n) ;

	for (i = 0; i < m; i++){
		p = arr1[i]; 
		bun = BUNfnd(BATmirror(propStat->pBat),(ptr) &p);
		if (bun == BUN_NONE) {
			printf("This prop must be there!!!!\n");
			return 0.0; 
		}
		else{
			tfidfV = propStat->tfidfs[bun]; 
			sumX2 +=  tfidfV*tfidfV;	
		}
	}

	for (i = 0; i < n; i++){
		p = arr2[i]; 
		bun = BUNfnd(BATmirror(propStat->pBat),(ptr) &p);
		if (bun == BUN_NONE) {
			printf("This prop must be there!!!!\n");
			return 0.0; 
		}
		else{
			tfidfV = propStat->tfidfs[bun]; 
			sumY2 +=  tfidfV*tfidfV;	
		}
	}
	
	i = 0;
	j = 0;
	while( i < n && j < m )
	{
		if( arr1[j] < arr2[i] ){
			j++;

		}
		else if( arr1[j] == arr2[i] )
		{
			p = arr1[j];
			bun = BUNfnd(BATmirror(propStat->pBat),(ptr) &p);

			if (bun == BUN_NONE) {
				printf("This prop must be there!!!!\n");
				return 0.0; 
			}
			else{
				tfidfV = propStat->tfidfs[bun];	 
				// We can do this because the tfidfs of a property in any CS
				// are the same
				sumXY += tfidfV*tfidfV;
			}
			j++;
			i++;
			numOverlap++;

		}
		else if( arr1[j] > arr2[i] )
			i++;
	}

	*numCombineP = m + n - numOverlap;
	
	free(tfidf1); 
	free(tfidf2); 

	return  ((float) sumXY / (sqrt(sumX2)*sqrt(sumY2)));
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
void getMaximumFreqCSs(CSset *freqCSset, BAT* coverageBat, BAT* freqBat, int numCS, int *nMaxCSs){

	int 	numFreqCS = freqCSset->numCSadded; 
	int 	i, j; 
	int 	numMaxCSs = 0;

	int 	tmpParentIdx; 
	int* 	coverage; 
	int* 	freq; 

	printf("Retrieving maximum frequent CSs: \n");

	for (i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].parentFreqIdx != -1) continue;
		for (j = (i+1); j < numFreqCS; j++){
			if (isSubset(freqCSset->items[i].lstProp, freqCSset->items[j].lstProp,  
					freqCSset->items[i].numProp,freqCSset->items[j].numProp) == 1) { 
				/* CSj is a subset of CSi */
				freqCSset->items[j].parentFreqIdx = i; 
			}
			else if (isSubset(freqCSset->items[j].lstProp, freqCSset->items[i].lstProp,  
					freqCSset->items[j].numProp,freqCSset->items[i].numProp) == 1) { 
				/* CSj is a subset of CSi */
				freqCSset->items[i].parentFreqIdx = j; 
				break; 
			}
			
		} 
		/* By the end, if this CS is not a subset of any other CS */
		if (freqCSset->items[i].parentFreqIdx == -1){
			numMaxCSs++;
			//printCS( freqCSset->items[i]); 
		}
	}

	*nMaxCSs = numMaxCSs;
	printf("Number of maximum CSs: %d / %d CSs \n", numMaxCSs, numCS);


	//Tunning
	for (i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].parentFreqIdx != -1){
			tmpParentIdx = freqCSset->items[i].parentFreqIdx; 
			while (freqCSset->items[tmpParentIdx].parentFreqIdx != -1){
				tmpParentIdx = freqCSset->items[tmpParentIdx].parentFreqIdx; // tracing to the maximum CS
			}

			//End. Update maximum CS for each frequent CS
			freqCSset->items[i].parentFreqIdx = tmpParentIdx; 
		}
	}

	// Update coverage for maximum CS
	
	for (i = 0; i < numFreqCS; i++){
		tmpParentIdx = freqCSset->items[i].parentFreqIdx; 

		coverage = (int*) Tloc(coverageBat, freqCSset->items[i].csId);
		freq = (int*) Tloc(freqBat, freqCSset->items[i].csId); 

		if (tmpParentIdx != -1){
			
			freqCSset->items[tmpParentIdx].coverage  += *coverage;
			freqCSset->items[tmpParentIdx].support  += *freq;
		}
		else{
			freqCSset->items[i].type = MAXCS; 	//Update type for this freqCS
			freqCSset->items[i].coverage += *coverage;
			freqCSset->items[i].support += *freq;

		}

	}



}


static 
PropStat* initPropStat(void){

	PropStat *propStat = (PropStat *) malloc(sizeof(PropStat));
	propStat->pBat = BATnew(TYPE_void, TYPE_oid, INIT_PROP_NUM);

	BATseqbase(propStat->pBat, 0);
	
	if (propStat->pBat == NULL) {
		return NULL; 
	}

	(void)BATprepareHash(BATmirror(propStat->pBat));
	if (!(propStat->pBat->T->hash)){
		return NULL;
	}

	propStat->freqs = (int*) malloc(sizeof(int) * INIT_PROP_NUM);
	if (propStat->freqs == NULL) return NULL; 

	propStat->tfidfs = (float*) malloc(sizeof(float) * INIT_PROP_NUM);
	if (propStat->tfidfs == NULL) return NULL; 
	
	propStat->numAdded = 0; 
	propStat->numAllocation = INIT_PROP_NUM; 

	// For posting list of each prop
	propStat->plCSidx = (Postinglist*) malloc(sizeof(Postinglist) * INIT_PROP_NUM); 
	if (propStat->plCSidx  == NULL) return NULL; 

	return propStat; 
}

static 
void addaProp(PropStat* propStat, oid prop, int csIdx){
	BUN	bun; 
	BUN	p; 

	int* _tmp1; 
	float* _tmp2; 
	Postinglist* _tmp3;
	
	p = prop; 
	bun = BUNfnd(BATmirror(propStat->pBat),(ptr) &prop);
	if (bun == BUN_NONE) {	/* New Prop */
	       if (propStat->pBat->T->hash && BATcount(propStat->pBat) > 4 * propStat->pBat->T->hash->mask) {
			HASHdestroy(propStat->pBat);
			BAThash(BATmirror(propStat->pBat), 2*BATcount(propStat->pBat));
		}

		propStat->pBat = BUNappend(propStat->pBat,&p, TRUE);
		
		if(propStat->numAdded == propStat->numAllocation){

			propStat->numAllocation += INIT_PROP_NUM;

			_tmp1 = realloc(propStat->freqs, ((propStat->numAllocation) * sizeof(int)));
			if (!_tmp1){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			
			propStat->freqs = (int*)_tmp1;
			
			_tmp2 = realloc(propStat->tfidfs, ((propStat->numAllocation) * sizeof(float)));
			if (!_tmp2){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			
			propStat->tfidfs = (float*)_tmp2;
			
			_tmp3 = realloc(propStat->plCSidx, ((propStat->numAllocation) * sizeof(Postinglist)));
			if (!_tmp3){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			
			propStat->plCSidx = (Postinglist*)_tmp3;
		}

		propStat->freqs[propStat->numAdded] = 1; 

		propStat->plCSidx[propStat->numAdded].lstIdx = (int *) malloc(sizeof(int) * INIT_CS_PER_PROP);
		if (propStat->plCSidx[propStat->numAdded].lstIdx  == NULL){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		} 
	
		propStat->plCSidx[propStat->numAdded].lstIdx[0] = csIdx;
		propStat->plCSidx[propStat->numAdded].numAdded = 1; 
		propStat->plCSidx[propStat->numAdded].numAllocation = INIT_CS_PER_PROP; 
		
		propStat->numAdded++;

	}
	else{		/*existing p*/
		propStat->freqs[bun]++;

		if (propStat->plCSidx[bun].numAdded == propStat->plCSidx[bun].numAllocation){
			
			propStat->plCSidx[bun].numAllocation += INIT_CS_PER_PROP;
		
			_tmp1 = realloc(propStat->plCSidx[bun].lstIdx, ((propStat->plCSidx[bun].numAllocation) * sizeof(int)));
			if (!_tmp1){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			propStat->plCSidx[bun].lstIdx = (int*) _tmp1; 
			
		}
		propStat->plCSidx[bun].lstIdx[propStat->plCSidx[bun].numAdded] = csIdx; 
		propStat->plCSidx[bun].numAdded++;
	}

}

static
void getPropStatisticsFromMaxCSs(PropStat* propStat, int numMaxCSs, oid* superCSFreqCSMap, CSset* freqCSset){

	int i, j; 
	oid freqId; 
	CS cs;

	for (i = 0; i < numMaxCSs; i++){
		freqId = superCSFreqCSMap[i];
		cs = (CS)freqCSset->items[freqId];

		for (j = 0; j < cs.numProp; j++){
			addaProp(propStat, cs.lstProp[j],freqId);
		}
	}

	for (i = 0; i < propStat->numAdded; i++){
		propStat->tfidfs[i] = tfidfComp(propStat->freqs[i],numMaxCSs);
	}

	//BATprint(propStat->pBat); 
	/*
	for (i = 0; i < (int)BATcount(propStat->pBat); i++){
		printf("Prop %d |||  freq: %d",i, propStat->freqs[i]);
		printf("   tfidf: %f \n",propStat->tfidfs[i] );
	}
	*/
}


static
PropStat* getPropStatisticsFromFreqCSs(CSset* freqCSset){

	int i, j; 
	CS cs;

	PropStat* propStat; 
	
	propStat = initPropStat(); 

	for (i = 0; i < freqCSset->numCSadded; i++){

		if (freqCSset->items[i].parentFreqIdx == -1){	// Only use the maximum or merge CS 
			cs = (CS)freqCSset->items[i];

			for (j = 0; j < cs.numProp; j++){
				addaProp(propStat, cs.lstProp[j], i);
			}
		}
	}

	/* Do not calculate the TFIDF score. May need in the future  
	 *  
	for (i = 0; i < propStat->numAdded; i++){
		propStat->tfidfs[i] = tfidfComp(propStat->freqs[i],numMaxCSs);
	}
	*/

	return propStat; 
}

static
void printPropStat(PropStat* propStat){
	int i, j; 
	oid	*pbt; 
	Postinglist ps; 

	printf("---- PropStat --- \n");
	for (i = 0; i < propStat->numAdded; i++){
		pbt = (oid *) Tloc(propStat->pBat, i);
		printf("Property " BUNFMT " :\n   FreqCSIdx: ", *pbt);

		ps = propStat->plCSidx[i]; 
		for (j = 0; j < ps.numAdded; j++){
			printf("  %d",ps.lstIdx[j]);
		}
		printf("\n");
	}
}

static 
void freePropStat(PropStat *propStat){
	int i; 
	BBPreclaim(propStat->pBat); 
	free(propStat->freqs); 
	free(propStat->tfidfs); 
	for (i = 0; i < propStat->numAdded; i++){
		free(propStat->plCSidx[i].lstIdx);
	}
	free(propStat->plCSidx); 
	free(propStat); 
}


static
void mergeMaximumFreqCSsAll(CSset *freqCSset, oid* superCSFreqCSMap, oid* superCSMergeMaxCSMap, int numMaxCSs, oid maxCSoid){
	int 		i, j, k; 
	int 		maxCSid = 0; 
	int 		freqId1, freqId2; 
	float 		simscore = 0.0; 
	CS     		*mergecs;
	oid		mercsId = 0; 
	oid		existMergecsId = BUN_NONE; 
	int 		numCombineP = 0; 
	CS		*cs1, *cs2;
	CS		*existmergecs, *mergecs1, *mergecs2; 

	PropStat	*propStat; 	/* Store statistics about properties */


	for (i = 0; i < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){
			superCSFreqCSMap[maxCSid] = i; 
			maxCSid++;
		}
	}

	//Initial superCSMergeMaxCSMap
	for (i = 0; i < numMaxCSs; i++){
		superCSMergeMaxCSMap[i] = BUN_NONE; 
	}

	
	propStat = initPropStat();
	getPropStatisticsFromMaxCSs(propStat, numMaxCSs, superCSFreqCSMap, freqCSset);
	


	for (i = 0; i < numMaxCSs; i++){
		freqId1 = superCSFreqCSMap[i];
		cs1 = (CS*) &(freqCSset->items[freqId1]);
	 	for (j = (i+1); j < numMaxCSs; j++){
			freqId2 = superCSFreqCSMap[j];
			cs2 = (CS*) &(freqCSset->items[freqId2]);
			
			if(USINGTFIDF == 0){
				simscore = similarityScore(cs1->lstProp, cs2->lstProp,
					cs1->numProp,cs2->numProp,&numCombineP);

				//printf("simscore Jaccard = %f \n", simscore);
			}
			else{
				simscore = similarityScoreTFIDF(cs1->lstProp, cs2->lstProp,
					cs1->numProp,cs2->numProp,&numCombineP, propStat);
				//printf("         Cosine = %f \n", simscore);
				
			}

			#if	USINGTFIDF	
			if (simscore > SIM_TFIDF_THRESHOLD){
			#else	
			if (simscore > SIM_THRESHOLD) {
			#endif				
				//Check whether these CS's belong to any mergeCS
				
				if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
					mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, mercsId + maxCSoid);
					//addmergeCStoSet(mergecsSet, *mergecs);
					cs1->parentFreqIdx = freqCSset->numCSadded;
					cs2->parentFreqIdx = freqCSset->numCSadded;
					addCStoSet(freqCSset,*mergecs);

					mercsId++;

				}
				else if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx != -1){
					existMergecsId = cs2->parentFreqIdx;
					existmergecs = (CS*) &(freqCSset->items[existMergecsId]);
					mergeACStoExistingmergeCS(*cs1,freqId1, existmergecs);
					cs1->parentFreqIdx = existMergecsId; 
				}
				
				else if (cs1->parentFreqIdx != -1 && cs2->parentFreqIdx == -1){
					existMergecsId = cs1->parentFreqIdx;
					existmergecs = (CS*)&(freqCSset->items[existMergecsId]);
					mergeACStoExistingmergeCS(*cs2,freqId2, existmergecs);
					cs2->parentFreqIdx = existMergecsId; 
					//printamergeCS(mergecsSet->items[existMergecsId] ,existMergecsId, freqCSset, superCSFreqCSMap);
				}
				else if (cs1->parentFreqIdx != cs2->parentFreqIdx){
					mergecs1 = (CS*)&(freqCSset->items[cs1->parentFreqIdx]);
					mergecs2 = (CS*)&(freqCSset->items[cs2->parentFreqIdx]);
					
					mergeTwomergeCS(mergecs1, mergecs2, cs1->parentFreqIdx);

					//Re-map for all maxCS in mergecs2
					for (k = 0; k < mergecs2->numConsistsOf; k++){
						freqCSset->items[mergecs2->lstConsistsOf[k]].parentFreqIdx = cs1->parentFreqIdx;
					}
				}
			}
		}
	}


	freePropStat(propStat);

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
#ifdef NDEBUG
		/* variable err is not used other than in above assertion */
		(void) err;
#endif
				
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


static void getStatisticCSsBySupports(BAT *pOffsetBat, BAT *freqBat, BAT *coverageBat, BAT *fullPBat, char isWriteToFile, int freqThreshold){

	//int 	*csPropNum; 
	//int	*csFreq; 
	FILE 	*fout; 
	oid 	*offset, *offset2; 
	int	numP; 
	BUN 	p, q; 
	BATiter	pi, freqi, coveri;
	int	*freq, *coverage; 
	char 	filename[100];
	char 	tmpStr[20];

	strcpy(filename, "csStatistic");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 
	fprintf(fout, " csId  #Prop   #frequency #coverage \n"); 

	pi = bat_iterator(pOffsetBat);
	freqi = bat_iterator(freqBat);
	coveri = bat_iterator(coverageBat); 

	BATloop(pOffsetBat, p, q){
		offset = (oid *) BUNtloc(pi, p);		

		if ((p+1) != BUNlast(pOffsetBat)){
			offset2 = (oid *)BUNtloc(pi, p + 1);
			numP = *offset2 - *offset;
		}
		else	//Last element
			numP = BUNlast(fullPBat) - *offset;

		freq = (int *) BUNtloc(freqi, p); 
		coverage = (int *) BUNtloc(coveri, p); 

		// Output the result 
		if (isWriteToFile == 0)
			printf(BUNFMT "  %d  %d %d \n", p, numP, *freq, *coverage); 
		else 
			fprintf(fout, BUNFMT " %d  %d %d \n", p, numP, *freq, *coverage); 
	}

	fclose(fout); 
	//free(csPropNum); 
}


static void getStatisticMaxCSs(CSset *freqCSset, char isWriteToFile, int freqThreshold){

	//int 	*csPropNum; 
	//int	*csFreq; 
	FILE 	*fout; 
	int	numFreqCS, i ; 
	char 	filename[100];
	char 	tmpStr[20];

	printf("Get statistics of Maximum CSs ....");

	numFreqCS = freqCSset->numCSadded; 

	strcpy(filename, "maxCSStatistic");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 
	fprintf(fout, " csId  #Prop   #frequency maxCSid coverage\n"); 

	for (i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){		// Check whether it is a maximumCS
			// Output the result 
			if (isWriteToFile == 0)
				printf(BUNFMT "  %d  %d  %d\n", freqCSset->items[i].csId, freqCSset->items[i].numProp,freqCSset->items[i].support, freqCSset->items[i].coverage); 
			else 
				fprintf(fout, BUNFMT " %d  %d  %d\n", freqCSset->items[i].csId, freqCSset->items[i].numProp,freqCSset->items[i].support, freqCSset->items[i].coverage); 

		}
	}

	fclose(fout); 
	//free(csPropNum); 
	printf("Done \n");
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

	csBats->coverageBat = BATnew(TYPE_void, TYPE_int, smallbatsz);
	
	if (csBats->coverageBat == NULL) {
		return NULL; 
	}

	return csBats; 
}



static 
void freeCSBats(CSBats *csBats){
	BBPreclaim(csBats->hsKeyBat); 
	BBPreclaim(csBats->hsValueBat); 
	BBPreclaim(csBats->freqBat); 
	BBPreclaim(csBats->coverageBat); 
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
				returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs); 
				#else
				returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset); 
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
	returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs); 
	#else
	returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset ); 
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

	//Update the numOrigFreqCS for freqCS
	freqCSset->numOrigFreqCS = freqCSset->numCSadded; 

	return MAL_SUCCEED; 
}

static 
str RDFrelationships(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi,  
		oid *subjCSMap, oid *subjSubCSMap, SubCSSet *csSubCSMap, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup){

	BUN	 	p, q; 
	oid 		*sbt = 0, *obt, *pbt;
	oid 		curS; 		/* current Subject oid */
	//oid 		CSoid = 0; 	/* Characteristic set oid */
	int 		numPwithDup;	/* Number of properties for current S */
	char 		objType;
	oid 		returnSubCSid; 
	char* 		buffTypes; 
	oid		realObjOid; 	
	char 		isBlankNode; 
	oid		curP; 

	if (BATcount(sbat) == 0) {
		throw(RDF, "rdf.RDFrelationships", "sbat must not be empty");
		/* otherwise, variable sbt is not initialized and thus
		 * cannot be dereferenced after the BATloop below */
	}

	buffTypes = (char *) malloc(sizeof(char) * (maxNumPwithDup + 1)); 

	numPwithDup = 0;
	curS = 0; 
	curP = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[curS], csSubCSMap);

				//Get the subCSId
				subjSubCSMap[*sbt] = returnSubCSid; 

			}
			curS = *sbt; 
			numPwithDup = 0;
			curP = 0; 
		}
				
		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = (char) ((*obt) >> (sizeof(BUN)*8 - 4))  &  7 ;	/* Get two bits 63th, 62nd from object oid */
	
		pbt = (oid *) BUNtloc(pi, p);

		/* Look at the referenced CS Id using subjCSMap */
		isBlankNode = 0;
		if (objType == URI || objType == BLANKNODE){
			realObjOid = (*obt) - ((oid) objType << (sizeof(BUN)*8 - 4));

			if (realObjOid <= maxSoid && subjCSMap[realObjOid] != BUN_NONE){
				if (objType == BLANKNODE) isBlankNode = 1;
				addReltoCSRel(subjCSMap[*sbt], subjCSMap[realObjOid], *pbt, &csrelSet[subjCSMap[*sbt]], isBlankNode);
			}
		}

		if (curP == *pbt){
			#if USE_MULTIPLICITY == 1	
			// Update the object type for this P as MULTIVALUES	
			buffTypes[numPwithDup-1] = MULTIVALUES; 
			#else
			buffTypes[numPwithDup] = objType;
			numPwithDup++;
			#endif
		}
		else{			
			buffTypes[numPwithDup] = objType; 
			numPwithDup++; 
			curP = *pbt; 
		}
	}
	
	/* Check for the last CS */
	returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[*sbt], csSubCSMap);
	subjSubCSMap[*sbt] = returnSubCSid; 

	free (buffTypes); 

	*ret = 1; 

	return MAL_SUCCEED; 
}

static
void initCsRelBetweenMergeFreqSet(CSmergeRel *csRelBetweenMergeFreqSet, int num){
	int i;
	for (i = 0; i < num; ++i) {
		csRelBetweenMergeFreqSet[i].origFreqIdx = i;
		csRelBetweenMergeFreqSet[i].lstRefFreqIdx = (int *) malloc (sizeof(int) * INIT_NUM_CSREL);
		csRelBetweenMergeFreqSet[i].lstPropId = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);

		csRelBetweenMergeFreqSet[i].lstCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);
		csRelBetweenMergeFreqSet[i].lstBlankCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);

		csRelBetweenMergeFreqSet[i].numRef = 0;
		csRelBetweenMergeFreqSet[i].numAllocation = INIT_NUM_CSREL;
	}
}

static
void addReltoCSmergeRel(int origFreqIdx, int refFreqIdx, oid propId, int freq, int numBlank, CSmergeRel *csmergerel)
{
	void *_tmp;
	void *_tmp1;
	void *_tmp2;
	void *_tmp3;

	int i = 0;

	assert (origFreqIdx == csmergerel->origFreqIdx);
#ifdef NDEBUG
	/* parameter origCSoid is not used other in about assertion */
	(void) origFreqIdx;
#endif

	while (i < csmergerel->numRef){
		if (refFreqIdx == csmergerel->lstRefFreqIdx[i] && propId == csmergerel->lstPropId[i]){
			//Existing
			break;
		}
		i++;
	}

	if (i != csmergerel->numRef){
		csmergerel->lstCnt[i] = csmergerel->lstCnt[i] + freq;
		csmergerel->lstBlankCnt[i] = csmergerel->lstBlankCnt[i] + numBlank;
		return;
	}
	else{	// New Ref
		if(csmergerel->numRef == csmergerel->numAllocation)
		{
			csmergerel->numAllocation += INIT_NUM_CSREL;

			_tmp = realloc(csmergerel->lstRefFreqIdx, (csmergerel->numAllocation * sizeof(int)));
			_tmp1 = realloc(csmergerel->lstPropId, (csmergerel->numAllocation * sizeof(oid)));
			_tmp2 = realloc(csmergerel->lstCnt, (csmergerel->numAllocation * sizeof(int)));
			_tmp3 = realloc(csmergerel->lstBlankCnt, (csmergerel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2 || !_tmp3){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csmergerel->lstRefFreqIdx = (int*)_tmp;
			csmergerel->lstPropId = (oid*)_tmp1;
			csmergerel->lstCnt = (int*)_tmp2;
			csmergerel->lstBlankCnt = (int*)_tmp3;
		}

		csmergerel->lstRefFreqIdx[csmergerel->numRef] = refFreqIdx;
		csmergerel->lstPropId[csmergerel->numRef] = propId;
		csmergerel->lstCnt[csmergerel->numRef] = freq;
		csmergerel->lstBlankCnt[csmergerel->numRef] = numBlank;
		csmergerel->numRef++;
	}
}

/* Create a new data structure to store relationships including merged CS */
static
void generateCsRelBetweenMergeFreqSet(CSmergeRel *csRelBetweenMergeFreqSet, CSrel *csrelBetweenMaxFreqSet, int numOid, int *csIdFreqIdxMap, CSset *freqCSset){
	int i,j;
	for (i = 0; i < numOid; ++i) {
		CSrel rel;
		int from;
		if (csrelBetweenMaxFreqSet[i].numRef == 0) continue; // ignore CS without relations
		rel = csrelBetweenMaxFreqSet[i];

		// update the 'from' value
		from = csIdFreqIdxMap[rel.origCSoid];
		assert (from != -1);
		if (freqCSset->items[from].parentFreqIdx != -1) {
			from = freqCSset->items[from].parentFreqIdx;
			assert (freqCSset->items[from].type = MERGECS);
		}

		for (j = 0; j < rel.numRef; ++j) {
			int to;
			// update the 'to' value
			to = csIdFreqIdxMap[rel.lstRefCSoid[j]];
			assert (to != -1);
			if (freqCSset->items[to].parentFreqIdx != -1) {
				to = freqCSset->items[to].parentFreqIdx;
				assert (freqCSset->items[to].type = MERGECS);
			}

			// add relation to new data structure
			addReltoCSmergeRel(from, to, rel.lstPropId[j], rel.lstCnt[j], rel.lstBlankCnt[j], &csRelBetweenMergeFreqSet[from]);
		}
	}
}

static
void printCSmergeRel(CSset *freqCSset, CSmergeRel *csRelBetweenMergeFreqSet, int freqThreshold){
	FILE 	*fout2,*fout2filter;
	char 	filename2[100];
	char 	tmpStr[20];
	str 	propStr;
	int		i,j;
	int		freq;

	strcpy(filename2, "csRelationshipBetweenMergeFreqCS");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename2, tmpStr);
	strcat(filename2, ".txt");

	fout2 = fopen(filename2,"wt");
	strcat(filename2, ".filter");
	fout2filter = fopen(filename2,"wt");

	for (i = 0; i < freqCSset->numCSadded; i++){
		if (csRelBetweenMergeFreqSet[i].numRef != 0){	//Only print CS with FK
			fprintf(fout2, "Relationship "BUNFMT": ", freqCSset->items[csRelBetweenMergeFreqSet[i].origFreqIdx].csId);
			fprintf(fout2filter, "Relationship "BUNFMT": ", freqCSset->items[csRelBetweenMergeFreqSet[i].origFreqIdx].csId);
			freq = freqCSset->items[csRelBetweenMergeFreqSet[i].origFreqIdx].support;
			fprintf(fout2, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", freqCSset->items[csRelBetweenMergeFreqSet[i].origFreqIdx].csId, freq, 1);
			fprintf(fout2filter, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", freqCSset->items[csRelBetweenMergeFreqSet[i].origFreqIdx].csId, freq, 1);

			for (j = 0; j < csRelBetweenMergeFreqSet[i].numRef; j++){
				#if SHOWPROPERTYNAME
				takeOid(csRelBetweenMergeFreqSet[i].lstPropId[j], &propStr);
				fprintf(fout2, BUNFMT "(P:" BUNFMT " - %s) (%d)(Blank:%d) ", freqCSset->items[csRelBetweenMergeFreqSet[i].lstRefFreqIdx[j]].csId,csRelBetweenMergeFreqSet[i].lstPropId[j], propStr, csRelBetweenMergeFreqSet[i].lstCnt[j], csRelBetweenMergeFreqSet[i].lstBlankCnt[j]);
				#else
				fprintf(fout2, BUNFMT "(P:" BUNFMT ") (%d)(Blank:%d) ", freqCSset->items[csRelBetweenMergeFreqSet[i].lstRefFreqIdx[j]].csId,csRelBetweenMergeFreqSet[i].lstPropId[j], csRelBetweenMergeFreqSet[i].lstCnt[j], csRelBetweenMergeFreqSet[i].lstBlankCnt[j]);
				#endif

				if (freq < csRelBetweenMergeFreqSet[i].lstCnt[j]*100){
					fprintf(fout2filter, BUNFMT "(P:" BUNFMT ") (%d)(Blank:%d) ", freqCSset->items[csRelBetweenMergeFreqSet[i].lstRefFreqIdx[j]].csId,csRelBetweenMergeFreqSet[i].lstPropId[j], csRelBetweenMergeFreqSet[i].lstCnt[j], csRelBetweenMergeFreqSet[i].lstBlankCnt[j]);
				}
			}
			fprintf(fout2, "\n");
			fprintf(fout2filter, "\n");
		}
	}

	fclose(fout2);
	fclose(fout2filter);
}

// for storing ontology data
str	**ontattributes = NULL;
int	ontattributesCount = 0;
str	**ontmetadata = NULL;
int	ontmetadataCount = 0;

/* Extract CS from SPO triples table */
str
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold, void *_freqCSset, oid **subjCSMap, oid *maxCSoid){

	BAT 		*sbat = NULL, *pbat = NULL, *obat = NULL, *mbat = NULL; 
	BATiter 	si, pi, oi; 	/*iterator for BAT of s,p,o columns in spo table */

	CSBats		*csBats; 
	oid		*subjSubCSMap;  /* Store the corresponding CS sub Id for each subject */
	BUN		*maxSoid; 	
	int		maxNumProp = 0;
	int		maxNumPwithDup = 0; 
	char		*csFreqMap; 
	CSrel   	*csrelSet;
	CSrel		*csrelToMaxFreqSet, *csrelFromMaxFreqSet;
	CSrel		*csrelBetweenMaxFreqSet; 
	CSmergeRel	*csRelBetweenMergeFreqSet;
	SubCSSet 	*csSubCSMap; 

	int*		csIdFreqIdxMap; /* Map a CSId to a freqIdx. Should be removed in the future .... */

	int		numMaxCSs = 0; 
	oid		*superCSFreqCSMap; 
	oid		*superCSMergeMaxCSMap;
	CSset		*freqCSset; 

	Labels		*labels;

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if (!(sbat->tsorted)){
		BBPreleaseref(sbat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", "sbat is not sorted");
	}

	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		BBPreleaseref(obat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}

	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat);

	csBats = initCSBats();

	freqCSset = ((CSset *) _freqCSset); 


	maxSoid = (BUN *) Tloc(sbat, BUNlast(sbat) - 1);
	printf("Max S oid: " BUNFMT "\n", *maxSoid);

	assert(*maxSoid != BUN_NONE); 

	*subjCSMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	subjSubCSMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	
	initArray(*subjCSMap, (*maxSoid) + 1, BUN_NONE);


	//Phase 1: Assign an ID for each CS
	#if STOREFULLCS
	RDFassignCSId(ret, sbat, si, pi, oi, freqCSset, freqThreshold, csBats, *subjCSMap, maxCSoid, &maxNumProp, &maxNumPwithDup);
	#else
	RDFassignCSId(ret, sbat, si, pi, freqCSset, freqThreshold, csBats, *subjCSMap, maxCSoid, &maxNumProp, &maxNumPwithDup);
	#endif
	


	//Phase 2: Check the relationship	

	printf("Max CS oid: " BUNFMT "\n", *maxCSoid);

	printf("Max Number of P (considering duplicated P): %d \n", maxNumPwithDup);

	csFreqMap = (char*) malloc(sizeof(char) * (*maxCSoid +1)); 
	initCharArray(csFreqMap, *maxCSoid +1, 0); 


	generateFreqCSMap(freqCSset,csFreqMap); 


	csrelSet = initCSrelset(*maxCSoid + 1);


	csSubCSMap = initCS_SubCSMap(*maxCSoid +1); 

	RDFrelationships(ret, sbat, si, pi, oi, *subjCSMap, subjSubCSMap, csSubCSMap, csrelSet, *maxSoid, maxNumPwithDup);


	printCSrelSet(csrelSet,csFreqMap, csBats->freqBat, *maxCSoid + 1, 1, *freqThreshold);  

	printSubCSInformation(csSubCSMap, csBats->freqBat, *maxCSoid + 1, 1, *freqThreshold); 

	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	/*get the statistic */

	//getTopFreqCSs(csMap,*freqThreshold);

	getMaximumFreqCSs(freqCSset, csBats->coverageBat,  csBats->freqBat, *maxCSoid + 1, &numMaxCSs); 

	//printf("Number of maximumCS: %d", numMaxCSs);
	
	printFreqCSSet(freqCSset, csBats->freqBat, mbat, 1, *freqThreshold); 


	csrelToMaxFreqSet = initCSrelset(*maxCSoid + 1);	// CS --> Reference MaxCSs
	csrelFromMaxFreqSet = initCSrelset(*maxCSoid + 1);	// CS --> Reference MaxCSs
	csrelBetweenMaxFreqSet = initCSrelset(*maxCSoid + 1);	// MaxCS --> Reference MaxCSs


	csIdFreqIdxMap = (int *) malloc (sizeof(int) * (*maxCSoid + 1)); 
	initcsIdFreqIdxMap(csIdFreqIdxMap, *maxCSoid + 1, -1, freqCSset);

	printCSrelWithMaxSet(freqCSset, csIdFreqIdxMap, csrelToMaxFreqSet, csrelFromMaxFreqSet, csrelBetweenMaxFreqSet, csrelSet,csFreqMap, csBats->freqBat, *maxCSoid + 1, *freqThreshold);  

	superCSFreqCSMap = (oid*) malloc(sizeof(oid) * numMaxCSs); 
	superCSMergeMaxCSMap = (oid*) malloc(sizeof(oid) * numMaxCSs);

	//mergeMaximumFreqCSs(freqCSset, superCSFreqCSMap, superCSMergeMaxCSMap, mergecsSet, numMaxCSs);

	mergeMaximumFreqCSsAll(freqCSset, superCSFreqCSMap, superCSMergeMaxCSMap, numMaxCSs, *maxCSoid);

	csRelBetweenMergeFreqSet = (CSmergeRel *) malloc (sizeof(CSmergeRel) * freqCSset->numCSadded);
	initCsRelBetweenMergeFreqSet(csRelBetweenMergeFreqSet, freqCSset->numCSadded);
	generateCsRelBetweenMergeFreqSet(csRelBetweenMergeFreqSet, csrelBetweenMaxFreqSet, *maxCSoid + 1, csIdFreqIdxMap, freqCSset);
	printCSmergeRel(freqCSset, csRelBetweenMergeFreqSet, *freqThreshold);

	printmergeCSSet(freqCSset, *freqThreshold);
	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csBats->pOffsetBat, csBats->freqBat, csBats->coverageBat, csBats->fullPBat, 1, *freqThreshold);
	getStatisticMaxCSs(freqCSset, 1, *freqThreshold);



	// Phase 3: Labels
	labels = createLabels(freqCSset, csRelBetweenMergeFreqSet, sbat, si, pi, oi, *subjCSMap, mbat, csIdFreqIdxMap, *freqThreshold, ontattributes, ontattributesCount, ontmetadata, ontmetadataCount);

	(void) labels; // TODO use

	freeLabels(labels, freqCSset);



	BBPreclaim(sbat); 
	BBPreclaim(pbat); 
	BBPreclaim(obat);
	BBPreclaim(mbat);

	free (subjSubCSMap);
	free (csFreqMap);
	free (superCSFreqCSMap);
	free (superCSMergeMaxCSMap); 

	freeCS_SubCSMapSet(csSubCSMap, *maxCSoid + 1); 

	free(csIdFreqIdxMap); 
	free(csRelBetweenMergeFreqSet);
	freeCSrelSet(csrelSet, *maxCSoid + 1); 
	freeCSrelSet(csrelToMaxFreqSet, *maxCSoid + 1); 
	freeCSrelSet(csrelBetweenMaxFreqSet, *maxCSoid + 1);  

	freeCSBats(csBats);



	//testBatHash(); 

	return MAL_SUCCEED;
}


/* Extract Properties and their supports from PSO table */
str
RDFextractPfromPSO(int *ret, bat *pbatid, bat *sbatid){
	BUN 	p, q; 
	BAT 	*sbat = NULL, *pbat = NULL; 
	BATiter si, pi; 	/*iterator for BAT of s,p columns in spo table */
	oid 	*bt = 0, *sbt;
	oid 	curS = 0; 	/* current Subject oid */
	oid 	curP; 		/* current Property oid */
	map_t 	pMap; 		
	int 	supportP; 	/* Support value for P */
	oid 	Poid = 0; 	/* Characteristic set oid */

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}
	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}

	if (BATcount(pbat) == 0) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		throw(RDF, "rdf.RDFextractPfromPSO", "pbat must not be empty");
		/* otherwise, variable bt is not initialized and thus
		 * cannot be dereferenced after the BATloop below */
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

static 
BAT* getOriginalOBat(BAT *obat){
	BAT*	origobat; 
	BATiter	oi; 
	BUN	p,q; 
	oid 	*obt; 
	char	objType; 

	origobat = BATcopy(obat,  obat->htype, obat->ttype, TRUE);
	oi = bat_iterator(origobat); 
	
	BATloop(origobat, p, q){

		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = (char) ((*obt) >> (sizeof(BUN)*8 - 4))  &  7 ;	/* Get two bits 63th, 62nd from object oid */
	
		if (objType == URI || objType == BLANKNODE){
			*obt = (*obt) - ((oid)objType << (sizeof(BUN)*8 - 4));
		}
		// Note that for the oid of literal data type, we do not need
		// to remove the object oid since the map bat also use this
		// oid

	}
	
	return origobat; 
}
/*
static 
oid getTblidFromSoid(oid Soid){
	int	freqCSid; 	
	
	return freqCSid; 
}
*/

static
str triplesubsort(BAT **sbat, BAT **pbat, BAT **obat){

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


str
RDFreorganize(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold){

	CSset		*freqCSset; 	/* Set of frequent CSs */
	oid		*subjCSMap = NULL;  	/* Store the corresponding CS Id for each subject */
	int 		i; 
	oid 		maxCSoid = 0; 
	BAT		*sbat = NULL, *obat = NULL, *pbat = NULL;
	BATiter		si; 
	BUN		p,q; 
	BAT		*sNewBat, *lmap, *rmap, *oNewBat, *origobat, *pNewBat; 
	BUN		newId; 
	oid		*sbt; 
	oid		*lastSubjId; 	/* Store the last subject Id in each freqCS */
	oid		freqId; 
	oid		lastS;
	oid		l,r; 
	bat		oNewBatid, pNewBatid; 
	oid		*csMFreqCSMap;	/* Store the mapping from a CS id to an index of a maxCS or mergeCS in freqCSset. */
	PropStat	*propStat; 

	freqCSset = initCSset();

	if (RDFextractCSwithTypes(ret, sbatid, pbatid, obatid, mapbatid, freqThreshold, freqCSset,&subjCSMap, &maxCSoid) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in extracting CSs");
	} 
	
	printf("Start re-organizing triple store for " BUNFMT " CSs \n", maxCSoid);
	csMFreqCSMap = (oid *) malloc (sizeof (oid) * (maxCSoid + 1)); 
	initArray(csMFreqCSMap, (maxCSoid + 1), BUN_NONE);


	lastSubjId = (oid *) malloc (sizeof(oid) * freqCSset->numOrigFreqCS); 
	for (i = 0; i < freqCSset->numOrigFreqCS; i++){
		if (freqCSset->items[i].parentFreqIdx != -1){	// Use the maximum or merge CS instead 	
			csMFreqCSMap[freqCSset->items[i].csId] = freqCSset->items[i].parentFreqIdx;
		}
		else
			csMFreqCSMap[freqCSset->items[i].csId] = i; 

		lastSubjId[i] = 0; 
	}

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(obat->batCacheid);
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	sNewBat = BATnew(TYPE_void, TYPE_oid, BATcount(sbat));
	if (sNewBat== NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	BATseqbase(sNewBat, 0);
	
	lmap = BATnew(TYPE_void, TYPE_oid, smallbatsz);

	if (lmap == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	lmap->tsorted = TRUE;

	BATseqbase(lmap, 0);
	
	rmap = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	if (rmap == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	BATseqbase(rmap, 0);
	
	si = bat_iterator(sbat); 

	printf("Re-assigning Subject oids ... ");
	lastS = -1; 
	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);
		freqId = csMFreqCSMap[subjCSMap[*sbt]];

		if (freqId != BUN_NONE){

			newId = lastSubjId[freqId];
			newId |= (BUN)freqId << (sizeof(BUN)*8 - NBITS_FOR_CSID);

			if (lastS != *sbt){	//new subject
				lastS = *sbt; 

				l = *sbt; 
				r = newId; 

				lmap = BUNappend(lmap, &l, TRUE);
				rmap = BUNappend(rmap, &r, TRUE);
				lastSubjId[freqId]++;
			}

		}
		else{	// Use original subject Id
			newId = *sbt; 
		}

		sNewBat = BUNappend(sNewBat, &newId, TRUE);

	}


	//BATprint(VIEWcreate(BATmirror(lmap),rmap)); 
	
	origobat = getOriginalOBat(obat); 

	//BATprint(origobat);
	
	if (RDFpartialjoin(&oNewBatid, &lmap->batCacheid, &rmap->batCacheid, &origobat->batCacheid) == MAL_SUCCEED){
		if ((oNewBat = BATdescriptor(oNewBatid)) == NULL) {
			throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
		}
	}
	else
		throw(RDF, "rdf.RDFreorganize", "Problem in using RDFpartialjoin for obat");


	if (RDFpartialjoin(&pNewBatid, &lmap->batCacheid, &rmap->batCacheid, &pbat->batCacheid) == MAL_SUCCEED){
		if ((pNewBat = BATdescriptor(pNewBatid)) == NULL) {
			throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
		}
	}
	else
		throw(RDF, "rdf.RDFreorganize", "Problem in using RDFpartialjoin for obat");

	//BATprint(oNewBat);
	printf("Done! \n");
	
	printf("Sort triple table according to P, S, O order ... ");
	if (triplesubsort(&pNewBat, &sNewBat, &oNewBat) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in sorting PSO");	
	}	
	printf("Done  \n");

	BATprint(pNewBat);

	BATprint(sNewBat);

	propStat = getPropStatisticsFromFreqCSs(freqCSset); 
	printPropStat(propStat); 
		
	freeCSset(freqCSset); 
	free(subjCSMap); 
	free(csMFreqCSMap);
	
	BBPreclaim(lmap);
	BBPreclaim(rmap); 
	BBPreclaim(sbat);
	BBPreclaim(sNewBat);
	BBPreclaim(obat); 
	BBPreclaim(origobat);
	BBPreclaim(oNewBat); 
	BBPreclaim(pbat); 
	BBPreclaim(pNewBat); 

	return MAL_SUCCEED; 
}
