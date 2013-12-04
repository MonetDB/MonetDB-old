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

/* This contains algebra functions used for RDF store only */

#include "monetdb_config.h"
#include "rdf.h"
#include "rdftypes.h"
#include "rdfschema.h"
#include "rdflabels.h"
#include "rdfretrieval.h"
#include "algebra.h"
#include <gdk.h>
#include <hashmap/hashmap.h>
#include <math.h>
#include <time.h>
#include <trie/trie.h>
#include <string.h>
#include "rdfminheap.h"

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


#if NEEDSUBCS
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

static void initCharArray(char* inputArr, int num, char defaultValue){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}
}
#endif /* if NEEDSUBCS */
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


static void initIntArray(int* inputArr, int num, oid defaultValue){
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



char isCSTable(CS item){
	if (item.parentFreqIdx != -1) return 0; 

	if (item.type == DIMENSIONCS) return 1; 

	#if REMOVE_SMALL_TABLE
	if (item.coverage < MINIMUM_TABLE_SIZE) return 0;
	#endif

	return 1; 
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
void creataCSrel(oid freqIdx, CSrel *csrel){
	//CSrel *csrel = (CSrel*) malloc(sizeof(CSrel));
	csrel->origFreqIdx = freqIdx; 
	csrel->lstRefFreqIdx = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);
	csrel->lstPropId = (oid*) malloc(sizeof(oid) * INIT_NUM_CSREL);
	csrel->lstCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);		
	csrel->lstBlankCnt = (int*) malloc(sizeof(int) * INIT_NUM_CSREL);		
	csrel->numRef = 0;
	csrel->numAllocation = INIT_NUM_CSREL;

	//return csrel; 
}


static 
void addReltoCSRel(oid origFreqIdx, oid refFreqIdx, oid propId, CSrel *csrel, char isBlankNode)
{
	void *_tmp; 
	void *_tmp1; 
	void *_tmp2;
	void *_tmp3; 

	int i = 0; 

	assert (origFreqIdx == csrel->origFreqIdx);
#ifdef NDEBUG
	/* parameter FreqIdx is not used other than in above assertion */
	(void) origFreqIdx;
#endif

	while (i < csrel->numRef){
		if (refFreqIdx == csrel->lstRefFreqIdx[i] && propId == csrel->lstPropId[i]){
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
			
			_tmp = realloc(csrel->lstRefFreqIdx, (csrel->numAllocation * sizeof(oid)));
			_tmp1 = realloc(csrel->lstPropId, (csrel->numAllocation * sizeof(oid)));
			_tmp2 = realloc(csrel->lstCnt, (csrel->numAllocation * sizeof(int)));
			_tmp3 = realloc(csrel->lstBlankCnt, (csrel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2 || !_tmp3){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csrel->lstRefFreqIdx = (oid*)_tmp;
			csrel->lstPropId = (oid*)_tmp1; 
			csrel->lstCnt = (int*)_tmp2; 
			csrel->lstBlankCnt = (int*)_tmp3; 
		}

		csrel->lstRefFreqIdx[csrel->numRef] = refFreqIdx;
		csrel->lstPropId[csrel->numRef] = propId;
		csrel->lstCnt[csrel->numRef] = 1; 
		csrel->lstBlankCnt[csrel->numRef] = (int) isBlankNode; 
		csrel->numRef++;
	}
}


static 
void addReltoCSRelWithFreq(oid origFreqIdx, oid refFreqIdx, oid propId, int freq, int numBlank, CSrel *csrel)
{
	void *_tmp; 
	void *_tmp1; 
	void *_tmp2; 
	void *_tmp3; 

	int i = 0; 

	assert (origFreqIdx == csrel->origFreqIdx);
#ifdef NDEBUG
	/* parameter origFreqIdx is not used other than in above assertion */
	(void) origFreqIdx;
#endif

	while (i < csrel->numRef){
		if (refFreqIdx == csrel->lstRefFreqIdx[i] && propId == csrel->lstPropId[i]){
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
			
			_tmp = realloc(csrel->lstRefFreqIdx, (csrel->numAllocation * sizeof(oid)));
			_tmp1 = realloc(csrel->lstPropId, (csrel->numAllocation * sizeof(oid)));		
			_tmp2 = realloc(csrel->lstCnt, (csrel->numAllocation * sizeof(int)));
			_tmp3 = realloc(csrel->lstBlankCnt, (csrel->numAllocation * sizeof(int)));

			if (!_tmp || !_tmp2 || !_tmp3){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			csrel->lstRefFreqIdx = (oid*)_tmp;
			csrel->lstPropId = (oid*)_tmp1; 
			csrel->lstCnt = (int*)_tmp2; 
			csrel->lstBlankCnt = (int*)_tmp3; 
		}

		csrel->lstRefFreqIdx[csrel->numRef] = refFreqIdx;
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
	//CSrel *csrel; 
	for (i = 0; i < numCSrel; i++){
		//csrel = creataCSrel(i); 
		//csrelSet[i] = (CSrel) *csrel;
		creataCSrel(i, &csrelSet[i]);
	}
	return csrelSet;
}

static 
void freeCSrelSet(CSrel *csrelSet, int numCSrel){
	int i; 

	for (i = 0; i < numCSrel; i++){
		free(csrelSet[i].lstRefFreqIdx);
		free(csrelSet[i].lstPropId);
		free(csrelSet[i].lstCnt); 
		free(csrelSet[i].lstBlankCnt);
	}
	free(csrelSet);
}

static 
void printCSrelSet(CSrel *csrelSet, CSset *freqCSset,  int num,  int freqThreshold){

	int 	i; 
	int 	j; 
	int 	freq; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];

	strcpy(filename, "csRelationship");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	for (i = 0; i < num; i++){
		if (csrelSet[i].numRef != 0){	//Only print CS with FK
			fprintf(fout, "Relationship %d: ", i);
			freq  = freqCSset->items[i].support;
			fprintf(fout, "FreqCS " BUNFMT " (Freq: %d) --> ", csrelSet[i].origFreqIdx, freq);
			for (j = 0; j < csrelSet[i].numRef; j++){
				fprintf(fout, BUNFMT " (%d) ", csrelSet[i].lstRefFreqIdx[j],csrelSet[i].lstCnt[j]);	
			}	
			fprintf(fout, "\n");
		}
	}


	fclose(fout);
	
}


static 
void getOrigRefCount(CSrel *csrelSet, CSset *freqCSset, int num,  int* refCount){

	int 	i, j; 
	int	freqId; 

	for (i = 0; i < num; i++){
		if (csrelSet[i].numRef != 0){	
			for (j = 0; j < csrelSet[i].numRef; j++){
				freqId = csrelSet[i].lstRefFreqIdx[j]; 
				#if FILTER_INFREQ_FK_FOR_IR
				if (csrelSet[i].lstCnt[j] < FILTER_THRESHOLD_FK_FOR_IR * freqCSset->items[freqId].support) continue; 
				#endif
				//Do not count the self-reference
				if (freqId != i) refCount[freqId] += csrelSet[i].lstCnt[j];
			}	
		}
	}

}

/* Get the number of indirect references to a CS */
static 
void getIRNums(CSrel *csrelSet, CSset *freqCSset, int num,  int* refCount, float *curIRScores, int noIter){

	int 	i, j, k; 
	int	freqId; 
	float	*lastIRScores;
	
	lastIRScores = (float *) malloc(sizeof(float) * num);
	for (i = 0; i < num; i++){
		curIRScores[i] = 0.0; 
		lastIRScores[i] = 0.0; 
	}
   
	for (k = 0; k < noIter; k++){
		for (i = 0; i < num; i++){
			curIRScores[i] = 0.0;
		}
		for (i = 0; i < num; i++){
			if (csrelSet[i].numRef != 0){	
				for (j = 0; j < csrelSet[i].numRef; j++){
					freqId = csrelSet[i].lstRefFreqIdx[j]; 
					#if FILTER_INFREQ_FK_FOR_IR
					if (csrelSet[i].lstCnt[j] < FILTER_THRESHOLD_FK_FOR_IR * freqCSset->items[freqId].support) continue; 
					#endif
					if (freqId != i){	//Do not count the self-reference
						//curIRScores[freqId] += (lastIRScores[i] * (float)csrelSet[i].lstCnt[j]/(float)refCount[freqId]) +  csrelSet[i].lstCnt[j];
						curIRScores[freqId] += (lastIRScores[i] * (float)csrelSet[i].lstCnt[j]/(float)refCount[freqId] * (float)csrelSet[i].lstCnt[j]/freqCSset->items[i].support) +  csrelSet[i].lstCnt[j];
					}
				}	
			}
		}
		
		//Update the last Indirect reference scores
		for (i = 0; i < num; i++){
			lastIRScores[i] = curIRScores[i]; 
		}

		/*
		printf(" ======== After %d iteration \n", k); 
		for (i = 0; i < num; i++){
			printf("IR score[%d] is %f \n", i, curIRScores[i]);
		}
		*/
	}
	


	free(lastIRScores);
}


static 
void updateFreqCStype(CSset *freqCSset, int num,  float *curIRScores, int *refCount){

	int 	i; 
	int	numDimensionCS = 0; 
	int 	totalSupport = 0; 	/* Total CS frequency */
	float	threshold  = 0.0; 
	
	for (i = 0; i < num; i++){	
		totalSupport += freqCSset->items[i].support; 
	}
	threshold = (float)totalSupport * IR_DIMENSION_THRESHOLD_PERCENTAGE; 
	printf("Total support %d --> Threshold for dimension table is: %f \n", totalSupport, threshold);

	printf("List of dimension tables: \n");
	for (i = 0; i < num; i++){
		if (refCount[i] < freqCSset->items[i].support) continue; 
		if (curIRScores[i] < threshold) continue; 

		freqCSset->items[i].type = DIMENSIONCS;
		//printf("A dimension CS with IR score = %f \n", curIRScores[i]);
		printf(" %d  ", i);
		numDimensionCS++;
	}
	
	printf("\n"); 
	printf("There are %d dimension CSs \n", numDimensionCS); 

}

#if NEEDSUBCS
static 
void setdefaultSubCSs(SubCSSet *subcsset, int num, BAT *sbat, oid *subjSubCSMap,oid *subjCSMap, char *subjdefaultMap){

	int i; 
	int j; 
	int	tmpmaxfreq; 
	int	defaultidx; 
	BUN	p,q; 
	BATiter	si; 
	oid	*sbt; 
	oid 	csId; 
	oid	subId; 

	for (i = 0; i < num; i++){
		if (subcsset[i].numSubCS != 0){	
			tmpmaxfreq = 0; 
			defaultidx = -1; 
			for (j = 0; j < subcsset[i].numSubCS; j++){
				if (subcsset[i].freq[j] > tmpmaxfreq){
					tmpmaxfreq = subcsset[i].freq[j];
					defaultidx = j; 
				}	
			}

			//Update default value
			subcsset[i].subCSs[defaultidx].isdefault = 1; 

		}
	}

	si = bat_iterator(sbat);

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);
		csId = subjCSMap[*sbt];
		subId = subjSubCSMap[*sbt];
		//printf("csId = " BUNFMT " | subId = " BUNFMT " \n", csId, subId);
		if (subcsset[csId].subCSs[subId].isdefault == 1){
			subjdefaultMap[*sbt] = 1; 
		}
	}
}

#endif

#if NEEDSUBCS
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
				fprintf(fout, "CS " BUNFMT " (Freq: %d) : ", subcsset[i].csId, *freq);
					
				if (*freq > freqThreshold){
					fprintf(foutfreq, BUNFMT "  ", subcsset[i].csId);
					fprintf(foutfreqfilter, BUNFMT "  ", subcsset[i].csId);
				}
				numSubCSFilter = 0;
				for (j = 0; j < subcsset[i].numSubCS; j++){
					if (subcsset[i].subCSs[j].isdefault == 1)
						fprintf(fout, "(default) "BUNFMT " (%d) ", subcsset[i].subCSs[j].subCSId, subcsset[i].freq[j]);	
					else
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

#endif  /* NEEDSUBCS */

char
getObjType(oid objOid){
	char objType = (char) (objOid >> (sizeof(BUN)*8 - 4))  &  7 ;

	return objType; 

}




/*
 * Init property types for each CS in FreqCSset (after merging)
 * For each property, init with all possible types (MULTIVALUES + 1))
 * 
 * */
static 
void initCSPropTypes(CSPropTypes* csPropTypes, CSset* freqCSset, int numMergedCS){
	int numFreqCS = freqCSset->numCSadded;
	int i, j, k ;
	int id; 
	
	id = 0; 
	for (i = 0; i < numFreqCS; i++){
		if ( isCSTable(freqCSset->items[i])  ){   // Only use the maximum or merge CS		
			csPropTypes[id].freqCSId = i; 
			csPropTypes[id].numProp = freqCSset->items[i].numProp;
			csPropTypes[id].numInfreqProp = 0; 
			csPropTypes[id].numNonDefTypes = 0;
			csPropTypes[id].lstPropTypes = (PropTypes*) GDKmalloc(sizeof(PropTypes) * csPropTypes[id].numProp);
			for (j = 0; j < csPropTypes[id].numProp; j++){
				csPropTypes[id].lstPropTypes[j].prop = freqCSset->items[i].lstProp[j]; 
				#if STAT_ANALYZE
				csPropTypes[id].lstPropTypes[j].numNull = 0;
				csPropTypes[id].lstPropTypes[j].numMVType = 0;
				csPropTypes[id].lstPropTypes[j].numSingleType = 0;		
				#endif
				csPropTypes[id].lstPropTypes[j].propFreq = 0; 
				csPropTypes[id].lstPropTypes[j].propCover = 0; 
				csPropTypes[id].lstPropTypes[j].numType = MULTIVALUES + 1;
				csPropTypes[id].lstPropTypes[j].defaultType = STRING; 
				csPropTypes[id].lstPropTypes[j].isMVProp = 0; 
				csPropTypes[id].lstPropTypes[j].isPKProp = 0; 
				csPropTypes[id].lstPropTypes[j].numMvTypes = 0; 
				csPropTypes[id].lstPropTypes[j].defColIdx = -1; 
				csPropTypes[id].lstPropTypes[j].isFKProp = 0;
				csPropTypes[id].lstPropTypes[j].refTblId = -1; 
				csPropTypes[id].lstPropTypes[j].refTblSupport = 0;
				csPropTypes[id].lstPropTypes[j].numReferring = 0;
				csPropTypes[id].lstPropTypes[j].numDisRefValues = 0;
				csPropTypes[id].lstPropTypes[j].isDirtyFKProp = 0; 
				csPropTypes[id].lstPropTypes[j].lstTypes = (char*)GDKmalloc(sizeof(char) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].lstFreq = (int*)GDKmalloc(sizeof(int) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].lstFreqWithMV = (int*)GDKmalloc(sizeof(int) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].colIdxes = (int*)GDKmalloc(sizeof(int) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].TableTypes = (char*)GDKmalloc(sizeof(char) * csPropTypes[id].lstPropTypes[j].numType);

				for (k = 0; k < csPropTypes[id].lstPropTypes[j].numType; k++){
					csPropTypes[id].lstPropTypes[j].lstFreq[k] = 0; 
					csPropTypes[id].lstPropTypes[j].lstFreqWithMV[k] = 0; 
					csPropTypes[id].lstPropTypes[j].TableTypes[k] = 0; 
					csPropTypes[id].lstPropTypes[j].colIdxes[k] = -1; 
				}

			}

			id++;
		}
	}

	assert(id == numMergedCS);

	//return csPropTypes;
}

static 
char isMultiValueCol(PropTypes pt){
	double tmpRatio;

	tmpRatio = ((double)pt.propCover / (pt.numSingleType + pt.numMVType));
	//printf("NumMVType = %d  | Ratio %f \n", pt.numMVType, tmpRatio);
	if ((pt.numMVType > 0) && (tmpRatio > IS_MULVALUE_THRESHOLD)){
		return 1; 
	}
	else return 0; 
}

static
char isInfrequentProp(PropTypes pt, CS cs){
	if (pt.propFreq < cs.support * INFREQ_PROP_THRESHOLD) return 1; 
	else return 0;

}

static 
void genCSPropTypesColIdx(CSPropTypes* csPropTypes, int numMergedCS, CSset* freqCSset){
	int i, j, k; 
	int tmpMaxFreq;  
	int defaultIdx;	 /* Index of the default type for a property */
	int curTypeColIdx = 0;
	int curDefaultColIdx = 0; 
	int curNumTypeMVTbl = 0; 
	int	freqId; 

	(void) freqCSset;
	(void) freqId; 

	for (i = 0; i < numMergedCS; i++){
		curTypeColIdx = 0; 
		curDefaultColIdx = -1; 
		for(j = 0; j < csPropTypes[i].numProp; j++){
			#if REMOVE_INFREQ_PROP
			freqId = csPropTypes[i].freqCSId;
			if (isInfrequentProp(csPropTypes[i].lstPropTypes[j], freqCSset->items[freqId])){
				for (k = 0; k < (MULTIVALUES+1); k++){
					csPropTypes[i].lstPropTypes[j].TableTypes[k] = PSOTBL;
					csPropTypes[i].lstPropTypes[j].colIdxes[k] = -1; 
				}	
				csPropTypes[i].lstPropTypes[j].isMVProp = 0;
				csPropTypes[i].numInfreqProp++;

				continue; 
			}
			#endif
			curDefaultColIdx++;
			csPropTypes[i].lstPropTypes[j].defColIdx = curDefaultColIdx;

			//printf("genCSPropTypesColIdx: Table: %d | Prop: %d \n", i, j);
			if (isMultiValueCol(csPropTypes[i].lstPropTypes[j])){
				//if this property is a Multi-valued prop
				csPropTypes[i].lstPropTypes[j].TableTypes[MULTIVALUES] = MAINTBL;
				csPropTypes[i].lstPropTypes[j].colIdxes[MULTIVALUES] = curDefaultColIdx;
				csPropTypes[i].lstPropTypes[j].isMVProp = 1; 

				//Find the default type for this MV col
				tmpMaxFreq = csPropTypes[i].lstPropTypes[j].lstFreqWithMV[0];
				defaultIdx = 0; 
				//find the default type of the multi-valued prop
				for (k = 0; k < MULTIVALUES; k++){
					if (csPropTypes[i].lstPropTypes[j].lstFreqWithMV[k] > tmpMaxFreq){
						tmpMaxFreq =  csPropTypes[i].lstPropTypes[j].lstFreqWithMV[k];
						defaultIdx = k; 	
					}
					
				}

				/* One type is set to be the default type (in the mv table) */
				csPropTypes[i].lstPropTypes[j].defaultType = defaultIdx;
				csPropTypes[i].lstPropTypes[j].colIdxes[defaultIdx] = 0; 	//The default type is the first col in the MV table
				csPropTypes[i].lstPropTypes[j].TableTypes[defaultIdx] = MVTBL;
				
				curNumTypeMVTbl = 1; //One default column for MV Table
				for (k = 0; k < MULTIVALUES; k++){
					if (csPropTypes[i].lstPropTypes[j].lstFreqWithMV[k] > 0){
						csPropTypes[i].lstPropTypes[j].TableTypes[k] = MVTBL;
						if (k != defaultIdx){
							csPropTypes[i].lstPropTypes[j].colIdxes[k] = curNumTypeMVTbl; 
							curNumTypeMVTbl++;
						}
					}
					else{
						csPropTypes[i].lstPropTypes[j].TableTypes[k] = NOTBL;
					}
				}
				csPropTypes[i].lstPropTypes[j].numMvTypes = curNumTypeMVTbl;
				//printf("Table %d with MV col %d has %d types",i, j, curNumTypeMVTbl);
				/* Count the number of column for MV table needed */

			}
			else{
				csPropTypes[i].lstPropTypes[j].isMVProp = 0;


				tmpMaxFreq = csPropTypes[i].lstPropTypes[j].lstFreq[0];
				defaultIdx = 0; 
				for (k = 0; k < MULTIVALUES; k++){
					if (csPropTypes[i].lstPropTypes[j].lstFreq[k] > tmpMaxFreq){
						tmpMaxFreq =  csPropTypes[i].lstPropTypes[j].lstFreq[k];
						defaultIdx = k; 	
					}
					//TODO: Check the case of single value col has a property with multi-valued objects
					if (csPropTypes[i].lstPropTypes[j].lstFreq[k] < csPropTypes[i].lstPropTypes[j].propFreq * INFREQ_TYPE_THRESHOLD){
						//non-frequent type goes to PSO
						csPropTypes[i].lstPropTypes[j].TableTypes[k] = PSOTBL; 
					}
					else
						csPropTypes[i].lstPropTypes[j].TableTypes[k] =TYPETBL;
				}
				/* One type is set to be the default type (in the main table) */
				csPropTypes[i].lstPropTypes[j].TableTypes[defaultIdx] = MAINTBL; 
				csPropTypes[i].lstPropTypes[j].colIdxes[defaultIdx] = curDefaultColIdx;
				csPropTypes[i].lstPropTypes[j].defaultType = defaultIdx; 
				
				//Multi-valued prop go to PSO
				csPropTypes[i].lstPropTypes[j].TableTypes[MULTIVALUES] = PSOTBL;
		
				/* Count the number of column needed */
				for (k = 0; k < csPropTypes[i].lstPropTypes[j].numType; k++){
					if (csPropTypes[i].lstPropTypes[j].TableTypes[k] == TYPETBL){
						csPropTypes[i].lstPropTypes[j].colIdxes[k] = curTypeColIdx; 
						curTypeColIdx++;
					}	
				}
			}
		}
		csPropTypes[i].numNonDefTypes = curTypeColIdx;

	}


}

#if     COLORINGPROP
static 
void updatePropSupport(CSPropTypes* csPropTypes, int numMergedCS, CSset* freqCSset){
	int i, j; 
	int freqId; 
	for (i = 0; i < numMergedCS; i++){
		freqId = csPropTypes[i].freqCSId; 
		freqCSset->items[freqId].lstPropSupport = (int*) malloc (sizeof(int) * freqCSset->items[freqId].numProp);
		for (j = 0; j < freqCSset->items[freqId].numProp; j++){
			freqCSset->items[freqId].lstPropSupport[j] = csPropTypes[i].lstPropTypes[j].propFreq; 
		}
	}
}



#endif /* #if COLORINGPROP */
static 
void printCSPropTypes(CSPropTypes* csPropTypes, int numMergedCS, CSset* freqCSset, int freqThreshold){
	char filename[100]; 
	char tmpStr[50]; 
	FILE *fout; 
	int i, j, k; 
	int	numMVCS = 0; 
	int	numMVCSFilter = 0; 
	int	numMVCols = 0; 
	int 	numMVColsFilter = 0;
	int	numNonMVCS = 0; 
	char	tmpIsMVCS = 0; 
	char 	tmpIsMVCSFilter = 0; 


	strcpy(filename, "csPropTypes");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	/* Print cspropTypes */
	for (i = 0; i < numMergedCS; i++){
		fprintf(fout, "MergedCS %d (Freq: %d): \n", i, freqCSset->items[csPropTypes[i].freqCSId].support);
		tmpIsMVCS = 0;
		tmpIsMVCSFilter = 0; 
		for(j = 0; j < csPropTypes[i].numProp; j++){
			if (csPropTypes[i].lstPropTypes[j].numMVType > 0){
				tmpIsMVCS = 1; 
				numMVCols++;
			}

			if (isMultiValueCol(csPropTypes[i].lstPropTypes[j])){
				tmpIsMVCSFilter = 1; 
				numMVColsFilter++;
			}

			fprintf(fout, "  P " BUNFMT "(%d | freq: %d | cov:%d | Null: %d | Single: %d | Multi: %d) \n", 
					csPropTypes[i].lstPropTypes[j].prop, csPropTypes[i].lstPropTypes[j].defaultType,
					csPropTypes[i].lstPropTypes[j].propFreq, csPropTypes[i].lstPropTypes[j].propCover,
					csPropTypes[i].lstPropTypes[j].numNull, csPropTypes[i].lstPropTypes[j].numSingleType, csPropTypes[i].lstPropTypes[j].numMVType);
			fprintf(fout, "         ");
			for (k = 0; k < csPropTypes[i].lstPropTypes[j].numType; k++){
				fprintf(fout, " Type %d (%d)(+MV: %d) | ", k, csPropTypes[i].lstPropTypes[j].lstFreq[k],csPropTypes[i].lstPropTypes[j].lstFreqWithMV[k]);
			}
			fprintf(fout, "\n");
			fprintf(fout, "         ");
			for (k = 0; k < csPropTypes[i].lstPropTypes[j].numType; k++){
				fprintf(fout, " Tbl %d (cl%d) | ", csPropTypes[i].lstPropTypes[j].TableTypes[k], csPropTypes[i].lstPropTypes[j].colIdxes[k]);
			}
			fprintf(fout, "\n");
		}

		if (tmpIsMVCS == 1){
			numMVCS++;
		}

		if (tmpIsMVCSFilter == 1){
			numMVCSFilter++;
		}
	}
	numNonMVCS = numMergedCS - numMVCS;
	fprintf(fout, "Number of tables with MV col: %d \n", numMVCS);
	fprintf(fout, "Number of tables with NO MV col: %d \n", numNonMVCS);
	fprintf(fout, "Number of MV cols: %d \n", numMVCols);

	fprintf(fout, "==== With filtering ==== \n");
	fprintf(fout, "Number of tables with MV col: %d \n", numMVCSFilter);
	fprintf(fout, "Number of tables with NO MV col: %d \n", (numMergedCS - numMVCSFilter));
	fprintf(fout, "Number of MV cols: %d \n", numMVColsFilter);


	fclose(fout); 

}


static 
void getTableStatisticViaCSPropTypes(CSPropTypes* csPropTypes, int numMergedCS, CSset* freqCSset, int freqThreshold){
	char filename[100]; 
	char tmpStr[50]; 
	FILE *fout; 
	int i, j; 
	int	totalTblNo = 0; 
	int	totalDefColNo = 0; 
	int	totalExColNo = 0; 
	int	totalMVTbl = 0;
	int	totalColInMVTbl = 0; 

	int	tmpNumMVCols; 
	int	tmpRealCov; 
	int	tmpTotalCols; 

	strcpy(filename, "tblStatisticByPropType");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 
	fprintf(fout, "TblId	NumProp	#InfreqProp	#DefCol	#ExCol	#MVColInMain(#MVtbl)	#totalCols  #RealCoverage	#MaxPossibleCoverage\n");
	/* Print cspropTypes */
	totalTblNo = numMergedCS;
	for (i = 0; i < numMergedCS; i++){
		fprintf(fout, "%d	%d	%d	%d	%d	", i, csPropTypes[i].numProp, csPropTypes[i].numInfreqProp, 
				csPropTypes[i].numProp - csPropTypes[i].numInfreqProp, csPropTypes[i].numNonDefTypes);

		totalDefColNo += csPropTypes[i].numProp - csPropTypes[i].numInfreqProp;
		totalExColNo += csPropTypes[i].numNonDefTypes;


		tmpNumMVCols = 0;
		tmpRealCov = 0;
		tmpTotalCols = 0; 
		tmpTotalCols = csPropTypes[i].numProp - csPropTypes[i].numInfreqProp + csPropTypes[i].numNonDefTypes;
		for(j = 0; j < csPropTypes[i].numProp; j++){
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1) continue; 	

			if (csPropTypes[i].lstPropTypes[j].isMVProp == 1){
				totalTblNo++;
				tmpNumMVCols++;
				totalColInMVTbl += csPropTypes[i].lstPropTypes[j].numMvTypes;	// Do not count the key column
				tmpTotalCols = csPropTypes[i].lstPropTypes[j].numMvTypes; 	//TODO: +1 for the key column in MVtable

			}

			tmpRealCov += csPropTypes[i].lstPropTypes[j].propCover;	
		}

		if (csPropTypes[i].numNonDefTypes > 0)	totalTblNo++;
		totalMVTbl += tmpNumMVCols;

		fprintf(fout, "%d       %d      %d	%d\n", tmpNumMVCols, tmpTotalCols, tmpRealCov, freqCSset->items[csPropTypes[i].freqCSId].coverage); 
		

	}
		
	fprintf(fout, "Total number of tables: %d \n", totalTblNo);
	fprintf(fout, "Total number of MV tables (=MVcols in maintbl): %d \n", totalMVTbl);
	fprintf(fout, "Total number of Default cols: %d \n", totalDefColNo);
	fprintf(fout, "Total number of Ex cols: %d \n", totalExColNo);
	fprintf(fout, "Total number of MV cols in MVTable: %d \n", totalColInMVTbl);
	
	fclose(fout); 

}

/*
 * Add types of properties 
 * Note that the property list is sorted by prop's oids
 * E.g., buffP = {3, 5, 7}
 * csPropTypes[tbIdx] contains properties {1,3,4,5,7} with types for each property and frequency of each <property, type>
 * */
static 
void addPropTypes(char *buffTypes, oid* buffP, int numP, int* buffCover, int **buffTypesCoverMV, int csId, int* csTblIdxMapping, CSPropTypes* csPropTypes){
	int i,j,k; 
	int tblId = csTblIdxMapping[csId];
	
	//printf("Add %d prop from CS %d to table %d \n", numP, csId, tblId);
	j = 0;
	if (tblId != -1){
		for (i = 0; i < numP; i++){
			//printf("  P: " BUNFMT " Type: %d ", buffP[i], buffTypes[i]);
			while (csPropTypes[tblId].lstPropTypes[j].prop != buffP[i]){
				#if STAT_ANALYZE
				csPropTypes[tblId].lstPropTypes[j].numNull++;
				#endif
				j++;
			}	
			//j is position of the property buffP[i] in csPropTypes[tblId]

			csPropTypes[tblId].lstPropTypes[j].propFreq++;
			csPropTypes[tblId].lstPropTypes[j].propCover += buffCover[i]; 
			csPropTypes[tblId].lstPropTypes[j].lstFreq[(int)buffTypes[i]]++; 
			csPropTypes[tblId].lstPropTypes[j].lstFreqWithMV[(int)buffTypes[i]]++;
			
			if (buffTypes[i] == MULTIVALUES){	
				//Add the number of triples per type (e.g., int, string
				//in this multi-valued prop to the freq of each type
				for  (k = 0; k < MULTIVALUES; k++){	
					csPropTypes[tblId].lstPropTypes[j].lstFreqWithMV[k] += buffTypesCoverMV[i][k]; 
				}
			}

			#if STAT_ANALYZE
			if (buffTypes[i] == MULTIVALUES){
				csPropTypes[tblId].lstPropTypes[j].numMVType++;
			}
			else{
				csPropTypes[tblId].lstPropTypes[j].numSingleType++;
			}
			#endif

			j++;

		}
		#if STAT_ANALYZE
		while (j < csPropTypes[tblId].numProp){
			csPropTypes[tblId].lstPropTypes[j].numNull++;
			j++;
		}
		#endif
	}
	//printf("\n");
}

static
void freeCSPropTypes(CSPropTypes* csPropTypes, int numCS){
	int i,j; 

	for (i = 0; i < numCS; i++){
		for (j = 0; j < csPropTypes[i].numProp; j++){
			GDKfree(csPropTypes[i].lstPropTypes[j].lstTypes); 
			GDKfree(csPropTypes[i].lstPropTypes[j].lstFreq);
			GDKfree(csPropTypes[i].lstPropTypes[j].lstFreqWithMV);
			GDKfree(csPropTypes[i].lstPropTypes[j].colIdxes);
			GDKfree(csPropTypes[i].lstPropTypes[j].TableTypes);
		}
		GDKfree(csPropTypes[i].lstPropTypes); 
	}
	GDKfree(csPropTypes);
}

#if NEEDSUBCS
static 
SubCS* creatSubCS(oid subCSId, int numP, char* buff, oid subCSsign){
	SubCS *subcs = (SubCS*) malloc(sizeof(SubCS)); 
	subcs->subTypes =  (char*) malloc(sizeof(char) * numP);
	
	copyTypesSet(subcs->subTypes, buff, numP); 
	subcs->subCSId = subCSId;
	subcs->numSubTypes = numP; 
	subcs->sign = subCSsign; 
	subcs->isdefault = 0; 
	return subcs; 
}

static 
void createaSubCSSet(oid csId, SubCSSet* subCSset){
	subCSset->csId = csId; 
	subCSset->numAllocation = INIT_NUM_SUBCS;
	subCSset->numSubCS = 0;
	subCSset->subCSs = (SubCS*) malloc(sizeof(SubCS) * INIT_NUM_SUBCS);
	subCSset->freq = (int*) malloc(sizeof(int) * INIT_NUM_SUBCS);

}

static 
SubCSSet* initCS_SubCSSets(oid numSubCSSet){
	oid i; 
	SubCSSet *subcssets = (SubCSSet*) malloc(sizeof(SubCSSet) * numSubCSSet); 
	for (i = 0; i < numSubCSSet;i++){
		createaSubCSSet(i, &subcssets[i]); 
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
oid addSubCS(char *buff, int numP, int csId, SubCSSet* csSubCSSet){
	SubCSSet *subcsset;
	oid subCSsign; 
	char isFound; 
	oid  subCSId; 
	SubCS *subCS; 


	subcsset = &(csSubCSSet[csId]);

	// Check the duplication
	subCSsign = RDF_hash_Tyleslist(buff, numP);

	isFound = checkExistsubCS(subCSsign, buff, numP, subcsset, &subCSId);
	
	if (isFound == 0){	// Add new 
		subCS = creatSubCS(subCSId, numP, buff, subCSsign);
		addSubCStoSet(subcsset, *subCS);
		free(subCS);
	}
	else{			// Exist
		//Update frequency
		subcsset->freq[subCSId]++;
	}

	return subCSId; 

}
#endif /*if NEEDSUBCS*/

static
int countNumberMergeCS(CSset *csSet){
	int i; 
	int num = 0;
	int maxNumProp = 0; 
	for (i = 0; i < csSet->numCSadded; i ++){
		if (csSet->items[i].parentFreqIdx == -1){
			num++;	
			if (csSet->items[i].numProp > maxNumProp) maxNumProp = csSet->items[i].numProp;
		}
	}

	printf("Max number of prop among %d merged CS is: %d \n", num, maxNumProp);

	return num; 

}

static
int countNumberConsistOfCS(CSset *csSet){
	int i; 
	int num = 0;
	for (i = 0; i < csSet->numCSadded; i ++){
		if (csSet->items[i].parentFreqIdx == -1){
			num += csSet->items[i].numConsistsOf; 	
			//printf("FCS %d: %d  ",i,csSet->items[i].numConsistsOf);
		}
	}
	printf("\n");

	return num; 

}
static
void freeCSset(CSset *csSet){
	int i;
	for(i = 0; i < csSet->numCSadded; i ++){
		free(csSet->items[i].lstProp);
		#if COLORINGPROP
		if (csSet->items[i].lstPropSupport != NULL)
			free(csSet->items[i].lstPropSupport);
		#endif
		free(csSet->items[i].lstConsistsOf);
	}

	#if STOREFULLCS
	for(i = 0; i < csSet->numOrigFreqCS; i ++){
		if (csSet->items[i].lstObj != NULL)
			free(csSet->items[i].lstObj);
	}
	#endif


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
CS* creatCS(oid csId, int freqIdx, int numP, oid* buff, oid subjectId, oid* lstObject, char type, int parentfreqIdx, int support, int coverage)
#else
static 
CS* creatCS(oid csId, int freqIdx, int numP, oid* buff, char type,  int parentfreqIdx, int support, int coverage)
#endif	
{
	CS *cs = (CS*)malloc(sizeof(CS)); 
	#if COLORINGPROP
	cs->lstPropSupport = NULL; 
	#endif
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
	cs->subject = subjectId; 
	if (subjectId != BUN_NONE){
		cs->lstObj =  (oid*) malloc(sizeof(oid) * numP);
		if (cs->lstObj == NULL){
			printf("Malloc failed. at %d", numP);
			exit(-1); 
		}
		copyOidSet(cs->lstObj, lstObject, numP); 
		}
	else
		cs->lstObj = NULL; 
	//printf("Create a CS with subjectId: " BUNFMT "\n", subjectId);
	#endif

	cs->type = type; 

	// This value is set for the 
	cs->parentFreqIdx = parentfreqIdx; 
	cs->support = support;
	cs->coverage = coverage; 

	// For using in the merging process
	cs->numConsistsOf = 1;
	cs->lstConsistsOf = (int *) malloc(sizeof(int)); 
	cs->lstConsistsOf[0]= freqIdx; 

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
	
	int numCombineP = 0; 

	CS *mergecs = (CS*) malloc (sizeof (CS)); 
	mergecs->type = (char)MERGECS; 
	mergecs->numConsistsOf = 2; 
	mergecs->lstConsistsOf = (int*) malloc(sizeof(int) * 2);

	//mergecs->lstConsistsOf[0] = cs1->csId;  
	//mergecs->lstConsistsOf[1] = cs2->csId; 

	mergecs->lstConsistsOf[0] = freqIdx1;  
	mergecs->lstConsistsOf[1] = freqIdx2; 
	
	mergecs->lstProp = (oid*) malloc(sizeof(oid) * (cs1.numProp + cs2.numProp));  // will be redundant

	if (mergecs->lstProp == NULL){
		printf("Malloc failed in merging two CSs \n");
		exit(-1);
	}

	mergeOidSets(cs1.lstProp, cs2.lstProp, mergecs->lstProp, cs1.numProp, cs2.numProp, &numCombineP); 

	mergecs->numProp = numCombineP;
	#if     COLORINGPROP 
	mergecs->lstPropSupport = NULL; 
	#endif
	#if STOREFULLCS
	mergecs->lstObj = NULL; 
	#endif
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

/* Merge list of consistsOf into mergecs1 */
static
void mergeConsistsOf(CS *mergecs1, CS *mergecs2){
	int* _tmp1;
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

#if USE_MULTIWAY_MERGING
/*Multi-way merging */
/* Get distinct list of mergedFreqIdx */
static  
int* getDistinctList(int *lstMergeCSFreqId, int num, int *numDistinct){
	int i; 
	int *lstDistinctFreqId;
	int 	*first; 
	int 	last; 

	lstDistinctFreqId = (int*) malloc(sizeof(int) * num); /* A bit redundant */
	
	GDKqsort(lstMergeCSFreqId, NULL, NULL, num, sizeof(int), 0, TYPE_int);
	
	first = lstMergeCSFreqId;
	last = *first; 
	*numDistinct = 1; 
	lstDistinctFreqId[0] = *first; 
	for (i =1; i < num; i++){
		first++; 
		if (last != *first){	/*new value*/
			lstDistinctFreqId[*numDistinct] = *first; 
			(*numDistinct)++;
			last = *first; 
		}
	}

	return lstDistinctFreqId; 

}

/* Calculate number of consistsOf in the merged CS 
 and  Update support and coverage: Total of all suppors */

static
void updateConsistsOfListAndSupport(CSset *freqCSset, CS *newmergeCS, int *lstDistinctFreqId, int numDistinct, char isExistingMergeCS, int mergecsFreqIdx){
	int 	i, j, tmpIdx, tmpFreqIdx;
	int 	mergeNumConsistsOf = 0;
	int	*_tmp; 
	int	totalSupport = 0;
	int	totalCoverage = 0; 
	int 	tmpConsistFreqIdx; 

	
	//printf("Distinct: \n");
	#if MINIMIZE_CONSISTSOF
	tmpIdx = newmergeCS->numConsistsOf;
	mergeNumConsistsOf = newmergeCS->numConsistsOf + numDistinct - isExistingMergeCS;
	#else
	tmpIdx = 0;
	for (i = 0; i < numDistinct; i++){
		tmpFreqIdx = lstDistinctFreqId[i]; 
		//printf("CS%d (%d)  ", tmpFreqIdx, freqCSset->items[tmpFreqIdx].numConsistsOf);
		mergeNumConsistsOf += freqCSset->items[tmpFreqIdx].numConsistsOf; 
	}
	
	#endif

	//printf("Number of freqCS consisted in mergeCS %d:  %d \n", mergecsFreqIdx, mergeNumConsistsOf);
	if (isExistingMergeCS){
		_tmp = realloc(newmergeCS->lstConsistsOf, sizeof(int) * mergeNumConsistsOf); 
        	if (!_tmp){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		}
		newmergeCS->lstConsistsOf = (int*)_tmp;
	}
	else{
		newmergeCS->lstConsistsOf = (int*)malloc(sizeof(int)  * mergeNumConsistsOf);
	}

	
	/*Update the parentIdx of the CS to-be-merged and its children */
	for (i = 0; i < numDistinct; i++){
		tmpFreqIdx = lstDistinctFreqId[i];
		for (j = 0; j < freqCSset->items[tmpFreqIdx].numConsistsOf; j++){
			tmpConsistFreqIdx =  freqCSset->items[tmpFreqIdx].lstConsistsOf[j];

			#if !MINIMIZE_CONSISTSOF
			newmergeCS->lstConsistsOf[tmpIdx] = tmpConsistFreqIdx; 
			tmpIdx++;
			#endif

			//Reset the parentFreqIdx
			freqCSset->items[tmpConsistFreqIdx].parentFreqIdx = mergecsFreqIdx;
		}
		freqCSset->items[tmpFreqIdx].parentFreqIdx = mergecsFreqIdx;
		#if MINIMIZE_CONSISTSOF
		if (tmpFreqIdx != mergecsFreqIdx) {
			newmergeCS->lstConsistsOf[tmpIdx] = tmpFreqIdx;
			tmpIdx++;
		}
		#endif
		//Update support
		totalSupport += freqCSset->items[tmpFreqIdx].support; 
		totalCoverage += freqCSset->items[tmpFreqIdx].coverage;
	}
	assert(tmpIdx == mergeNumConsistsOf);
	newmergeCS->numConsistsOf = mergeNumConsistsOf;

	newmergeCS->support = totalSupport;
	newmergeCS->coverage = totalCoverage; 

}
/*
Multi-way merging for list of freqCS
*/
static 
int* mergeMultiCS(CSset *freqCSset, int *lstFreqId, int num, oid *mergecsId,int *retNumDistinct, int *isNew, int *retFreqIdx){
	
	int 	i; 
	int 	*lstMergeCSFreqId;
	int	*lstDistinctFreqId = NULL; 
	int 	numDistinct = 0; 
	CS	*newmergeCS; 
	char 	isExistingMergeCS = 0;
	int	mergecsFreqIdx = -1;
	oid	*_tmp2; 
	oid	*tmpPropList; 
	int 	numCombinedP = 0; 
	int	tmpParentIdx;	



	/* Get the list of merge FreqIdx */
	lstMergeCSFreqId = (int*) malloc(sizeof(int) * num); 
	
	for (i = 0; i < num; i++){
		if (freqCSset->items[lstFreqId[i]].parentFreqIdx != -1){
			tmpParentIdx = freqCSset->items[lstFreqId[i]].parentFreqIdx;
			//Go to the last mergeCS
			while(freqCSset->items[tmpParentIdx].parentFreqIdx != -1)
				tmpParentIdx = freqCSset->items[tmpParentIdx].parentFreqIdx;

			lstMergeCSFreqId[i] = tmpParentIdx;
			mergecsFreqIdx = lstMergeCSFreqId[i]; //An existing one
			isExistingMergeCS = 1; 
		}
		else
			lstMergeCSFreqId[i] = lstFreqId[i];

		//printf(" %d --> %d  | ", lstFreqId[i], lstMergeCSFreqId[i]);
	}

	if (isExistingMergeCS == 0) mergecsFreqIdx = freqCSset->numCSadded; 

	lstDistinctFreqId = getDistinctList(lstMergeCSFreqId,num, &numDistinct);

	if (numDistinct < 2){
		free(lstMergeCSFreqId);
		free(lstDistinctFreqId);
		return NULL;
	}
	
	/* Create or not create a new CS */
	if (isExistingMergeCS){
		newmergeCS = (CS*) &(freqCSset->items[mergecsFreqIdx]);
	}
	else{
		newmergeCS = (CS*) malloc (sizeof (CS));
		newmergeCS->support = 0;
		newmergeCS->coverage = 0; 
		newmergeCS->numConsistsOf = 0;
	}


	updateConsistsOfListAndSupport(freqCSset, newmergeCS, lstDistinctFreqId, numDistinct, isExistingMergeCS,mergecsFreqIdx);

	/*Reset parentIdx */
	newmergeCS->parentFreqIdx = -1;
	newmergeCS->type = MERGECS;

	/*Merge the list of prop list */
	tmpPropList = mergeMultiPropList(freqCSset, lstDistinctFreqId, numDistinct, &numCombinedP);

	if (isExistingMergeCS){		//For existed mergeCS, reallocate lstProp	
		_tmp2 = realloc(newmergeCS->lstProp, sizeof(oid) * numCombinedP); 
		
        	if (!_tmp2){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		}
		newmergeCS->lstProp = (oid*)_tmp2; 
		memcpy(newmergeCS->lstProp, tmpPropList, numCombinedP * sizeof(oid));
		newmergeCS->numProp = numCombinedP;
		newmergeCS->numAllocation = numCombinedP;
	}
	else{
		newmergeCS->lstProp =  (oid*) malloc(sizeof(oid) * numCombinedP);
		memcpy(newmergeCS->lstProp, tmpPropList, numCombinedP * sizeof(oid));		
		newmergeCS->numProp = numCombinedP;
		newmergeCS->numAllocation = numCombinedP;
		newmergeCS->csId = *mergecsId;
		mergecsId[0]++;
	}

	free(tmpPropList); 

	
	#if     COLORINGPROP 
	newmergeCS->lstPropSupport = NULL; 
	#endif

	if (isExistingMergeCS == 0){
		addCStoSet(freqCSset, *newmergeCS);
		free(newmergeCS); 
	}

	free(lstMergeCSFreqId);

	*retNumDistinct = numDistinct;
	*isNew = 1 -  isExistingMergeCS;
	*retFreqIdx = mergecsFreqIdx;

	return lstDistinctFreqId;
}

#endif /* USE_MULTIWAY_MERGING */

static 
str printFreqCSSet(CSset *freqCSset, BAT *freqBat, BAT *mapbat, char isWriteTofile, int freqThreshold, CSlabel* labels){

	int 	i; 
	int 	j; 
	int 	*freq; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];

#if SHOWPROPERTYNAME
	str 	propStr; 
	#if STOREFULLCS
	str	subStr; 
	str	objStr; 
	oid 	objOid; 
	char 	objType; 
	BUN	bun; 
	#endif
	int	ret; 
	char*   schema = "rdf";
	
	BATiter mapi; 
	(void) mapi;
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

			printf("CS " BUNFMT " (Freq: %d) | Parent " BUNFMT " \n", cs.csId, *freq, freqCSset->items[cs.parentFreqIdx].csId);
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
		strcat(filename, ".txt");

		fout = fopen(filename,"wt"); 

		for (i = 0; i < freqCSset->numCSadded; i++){
			CS cs = (CS)freqCSset->items[i];
			freq  = (int *) Tloc(freqBat, cs.csId);
			if (cs.type != MAXCS) assert(*freq == cs.support);

			#if STOREFULLCS	
			if (cs.subject != BUN_NONE){
				takeOid(cs.subject, &subStr);

				if (labels[i].name == BUN_NONE) {
					fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d) | Subject: %s  | FreqParentIdx %d \n", cs.csId, i, "DUMMY", *freq, subStr, cs.parentFreqIdx);
				} else {
					str labelStr;
					takeOid(labels[i].name, &labelStr);
					fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d) | Subject: %s  | FreqParentIdx %d \n", cs.csId, i, labelStr, *freq, subStr, cs.parentFreqIdx);
					GDKfree(labelStr); 
				}

				GDKfree(subStr);
			}
			else{
				if (labels[i].name == BUN_NONE) {
					fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d) | FreqParentIdx %d \n", cs.csId, i, "DUMMY", *freq, cs.parentFreqIdx);
				} else {
					str labelStr;
					takeOid(labels[i].name, &labelStr);
					fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d) | FreqParentIdx %d \n", cs.csId, i, labelStr, *freq, cs.parentFreqIdx);
					GDKfree(labelStr);
				}
			}
			#endif	

			for (j = 0; j < cs.numProp; j++){
				takeOid(cs.lstProp[j], &propStr);
				//fprintf(fout, "  P:" BUNFMT " --> ", cs.lstProp[j]);	
				fprintf(fout, "  P(" BUNFMT "):%s --> ", cs.lstProp[j],propStr);	

				GDKfree(propStr);
				
				#if STOREFULLCS
				// Get object value
				if (cs.lstObj != NULL){
					objOid = cs.lstObj[j]; 

					objType = getObjType(objOid); 

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

					if (objType == URI || objType == BLANKNODE){
						GDKfree(objStr);
					}
				}
				#endif


			}	
			fprintf(fout, "\n");
		}

		fclose(fout);
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
	int	totalCoverage = 0; 

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
				fprintf(fout,"PropId: "BUNFMT"  --->    %s\n", cs.lstProp[j],  propStr);
				GDKfree(propStr);
			}
			fprintf(fout, "\n");
		}
	}

	fclose(fout);
	
	/*Asserting the total number of coverage by mergeCS */

	for (i = 0; i < nummergecs; i++){
		CS cs = (CS)freqCSset->items[i];
		if (cs.parentFreqIdx == -1){
			totalCoverage += cs.coverage; 
		}
	}

	printf("Total coverage by merged CS's: %d \n", totalCoverage);

	TKNZRclose(&ret);
	return MAL_SUCCEED;
}


static 
str printsubsetFromCSset(CSset *freqCSset, BAT* subsetIdxBat, int num, int* mergeCSFreqCSMap, CSlabel *label){

	int 	i,j; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];
	int 	ret;
	int	*tblIdx; 
	int	freqIdx; 

	str 	propStr; 
	char*   schema = "rdf";
	CS	cs; 
	int	tmpNumcand;
	str	canStr; 


	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	

	strcpy(filename, "selectedSubset");
	sprintf(tmpStr, "%d", num);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	for (i = 0; i < num; i++){
		tblIdx = (int*) Tloc(subsetIdxBat, i); 
		freqIdx = mergeCSFreqCSMap[*tblIdx];
		cs = (CS)freqCSset->items[freqIdx];
		assert (cs.parentFreqIdx == -1);
		tmpNumcand = (NUM_SAMPLE_CANDIDATE > label[freqIdx].candidatesCount)?label[freqIdx].candidatesCount:NUM_SAMPLE_CANDIDATE;
		fprintf(fout, "Table %d  | Candidates: ",i);
		
		for (j = 0; j < tmpNumcand; j++){
			//fprintf(fout,"  "  BUNFMT,sample.candidates[j]);
			if (label[freqIdx].candidates[j] != BUN_NONE){
#if USE_SHORT_NAMES
				str canStrShort = NULL;
#endif
				takeOid(label[freqIdx].candidates[j], &canStr); 
#if USE_SHORT_NAMES
				getPropNameShort(&canStrShort, canStr);
				fprintf(fout," %s  ::: ",  canStrShort);
				GDKfree(canStrShort);
#else
				fprintf(fout," %s  ::: ",  canStr);
#endif
				GDKfree(canStr); 
			
			}
		}
		fprintf(fout, "\n");
		fprintf(fout, "Coverage: %d, NumProp: %d) \n",cs.coverage, cs.numProp);

		for (j = 0; j < cs.numProp; j++){
			takeOid(cs.lstProp[j], &propStr);	
			fprintf(fout,"          %s\n", propStr);
			GDKfree(propStr);
		}
		fprintf(fout, "\n");
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
void addaProp(PropStat* propStat, oid prop, int csIdx, int invertIdx){
	BUN	bun; 
	BUN	p; 

	int* _tmp1; 
	float* _tmp2; 
	Postinglist* _tmp3;
	int* _tmp4; 
	
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
		propStat->plCSidx[propStat->numAdded].lstInvertIdx = (int *) malloc(sizeof(int) * INIT_CS_PER_PROP);


		if (propStat->plCSidx[propStat->numAdded].lstIdx  == NULL){
			fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		} 
	
		propStat->plCSidx[propStat->numAdded].lstIdx[0] = csIdx;
		propStat->plCSidx[propStat->numAdded].lstInvertIdx[0] = invertIdx;
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
			
			_tmp4 = realloc(propStat->plCSidx[bun].lstInvertIdx, ((propStat->plCSidx[bun].numAllocation) * sizeof(int)));
			if (!_tmp4){
				fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			propStat->plCSidx[bun].lstInvertIdx = (int*) _tmp4; 

		}
		propStat->plCSidx[bun].lstIdx[propStat->plCSidx[bun].numAdded] = csIdx; 
		propStat->plCSidx[bun].lstInvertIdx[propStat->plCSidx[bun].numAdded] = invertIdx; 

		propStat->plCSidx[bun].numAdded++;
	}

}


static 
void addNewCS(CSBats *csBats, PropStat* fullPropStat, BUN* csKey, oid* key, oid *csoid, int num, int numTriples){
	int freq = 1; 
	int coverage = numTriples; 
	BUN	offset; 
	#if FULL_PROP_STAT
	int	i; 
	#endif
	
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

	#if FULL_PROP_STAT == 1		// add property to fullPropStat
	for (i = 0; i < num; i++){
		addaProp(fullPropStat, key[i], *csoid, i);
	}
	if (num > fullPropStat->maxNumPPerCS)
		fullPropStat->maxNumPPerCS = num; 
	#else
	(void) fullPropStat; 
	#endif

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
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset, oid subjectId, oid* buffObjs, PropStat *fullPropStat)
#else
static 
oid putaCStoHash(CSBats *csBats, oid* key, int num, int numTriples, 
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset, PropStat *fullPropStat)
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
		addNewCS(csBats, fullPropStat, &csKey, key, csoid, num, numTriples);
		
		//Handle the case when freqThreshold == 1 
		if (isStoreFreqCS ==1 && freqThreshold == 1){
			#if STOREFULLCS
			freqCS = creatCS(csId, freqCSset->numCSadded, num, key, subjectId, buffObjs, FREQCS, -1, 0,0);		
			#else
			freqCS = creatCS(csId, freqCSset->numCSadded, num, key, FREQCS,-1,0,0);			
			#endif
			addCStoSet(freqCSset, *freqCS);
			free(freqCS);
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
			addNewCS(csBats, fullPropStat, &csKey, key, csoid, num, numTriples);
			
			//Handle the case when freqThreshold == 1 
			if (isStoreFreqCS ==1 && freqThreshold == 1){
				
				#if STOREFULLCS
				freqCS = creatCS(csId, freqCSset->numCSadded, num, key, subjectId, buffObjs, FREQCS,-1,0,0);		
				#else
				freqCS = creatCS(csId, freqCSset->numCSadded, num, key, FREQCS,-1,0,0);			
				#endif
				addCStoSet(freqCSset, *freqCS);
				free(freqCS);
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
					freqCS = creatCS(csId, freqCSset->numCSadded, num, key, subjectId, buffObjs, FREQCS,-1,0,0);		
					#else
					freqCS = creatCS(csId, freqCSset->numCSadded, num, key, FREQCS,-1,0,0);			
					#endif
					addCStoSet(freqCSset, *freqCS);
					free(freqCS);
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
 * m > n
 * */
static int isSubset(oid* arr1, oid* arr2, int m, int n)
{
	int i = 0, j = 0;
	 

	if (arr2[n-1] > arr1[m-1]) return 0; 

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

/*Recursively update the parentFreqIdx so that 
 *a cs.parentFreqIdx points to the last merged CS
 * */
static
void updateParentIdxAll(CSset *freqCSset){
	int i; 
	int tmpParentIdx;

	for (i = 0; i < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx != -1){
			tmpParentIdx = freqCSset->items[i].parentFreqIdx; 
			while (freqCSset->items[tmpParentIdx].parentFreqIdx != -1){
				tmpParentIdx = freqCSset->items[tmpParentIdx].parentFreqIdx; // tracing to the maximum CS
			}

			//End. Update maximum CS for each frequent CS
			freqCSset->items[i].parentFreqIdx = tmpParentIdx;

		}
	}
}

/*
 * Get the maximum frequent CSs from a CSset
 * Here maximum frequent CS is a CS that there exist no other CS which contains that CS
 * */
static 
void mergeCSbyS4(CSset *freqCSset, CSlabel** labels, oid *mergeCSFreqCSMap, int curNumMergeCS, oid **ontmetadata, int ontmetadataCount){

	int 	numMergeCS = curNumMergeCS; 
	int 	i, j; 
	int 	numMaxCSs = 0;

	int 	tmpParentIdx; 
	int	freqId1, freqId2; 
	#if USE_LABEL_FINDING_MAXCS
	char	isLabelComparable = 0;
	#endif
	char	isDiffLabel = 0;
	int	numP1, numP2; 
	CS	*mergecs1, *mergecs2; 
	(void) labels;

	printf("Retrieving maximum frequent CSs: \n");

	for (i = 0; i < numMergeCS; i++){
		freqId1 = mergeCSFreqCSMap[i]; 
		if (freqCSset->items[freqId1].parentFreqIdx != -1) continue;
		#if	NOT_MERGE_DIMENSIONCS
		if (freqCSset->items[freqId1].type == DIMENSIONCS) continue; 
		#endif

		#if USE_LABEL_FINDING_MAXCS
		isLabelComparable = 0;
		if ((*labels)[i].name != BUN_NONE) isLabelComparable = 1; // no "DUMMY"
		#endif

		for (j = (i+1); j < numMergeCS; j++){
			freqId2 = mergeCSFreqCSMap[j];
			#if	NOT_MERGE_DIMENSIONCS
			if (freqCSset->items[freqId2].type == DIMENSIONCS) continue; 
			#endif

			isDiffLabel = 0; 
			#if USE_LABEL_FINDING_MAXCS
			if (isLabelComparable == 0 || strcmp((*labels)[freqId1].name, (*labels)[freqId2].name) != 0) {
				isDiffLabel = 1; 
			}
			#endif

			if (isDiffLabel == 0){
				numP2 = freqCSset->items[freqId2].numProp;
				numP1 = freqCSset->items[freqId1].numProp;
				if (numP2 > numP1 && (numP2-numP1)< MAX_SUB_SUPER_NUMPROP_DIF){
					if (isSubset(freqCSset->items[freqId2].lstProp, freqCSset->items[freqId1].lstProp, numP2,numP1) == 1) { 
						/* CSj is a superset of CSi */
						freqCSset->items[freqId1].parentFreqIdx = freqId2; 
						updateLabel(S3, freqCSset, labels, 0, freqId2, freqId1, freqId2, BUN_NONE, ontmetadata, ontmetadataCount, NULL, -1);
						break; 
					}
				}
				else if (numP2 < numP1 && (numP1-numP2)< MAX_SUB_SUPER_NUMPROP_DIF){
					if (isSubset(freqCSset->items[freqId1].lstProp, freqCSset->items[freqId2].lstProp,  
							numP1,numP2) == 1) { 
						/* CSj is a subset of CSi */
						freqCSset->items[freqId2].parentFreqIdx = freqId1; 
						updateLabel(S3, freqCSset, labels, 0, freqId1, freqId1, freqId2, BUN_NONE, ontmetadata, ontmetadataCount, NULL, -1);
					}		
				
				}
			}

			//Do not need to consider the case that the numProps are the same
		} 
		/* By the end, if this CS is not a subset of any other CS */
		if (freqCSset->items[freqId1].parentFreqIdx == -1){
			numMaxCSs++;
		}
	}

	printf("Number of maximum CSs: %d / %d CSs \n", numMaxCSs, numMergeCS);

	//Update the parentFreqIdx for all freqCS
	updateParentIdxAll(freqCSset);
	
	// Update freq/coverage for maximumCS and merge ListConsists of
	
	for (i = 0; i < numMergeCS; i++){
		freqId1 = mergeCSFreqCSMap[i];
		tmpParentIdx = freqCSset->items[freqId1].parentFreqIdx; 
		
		if (tmpParentIdx != -1){

			freqCSset->items[tmpParentIdx].coverage  += freqCSset->items[freqId1].coverage;
			freqCSset->items[tmpParentIdx].support  += freqCSset->items[freqId1].support;
			//printf("NumProp differences between sub-super CS: %d / %d \n", freqCSset->items[tmpParentIdx].numProp - freqCSset->items[freqId1].numProp, freqCSset->items[tmpParentIdx].numProp);

			mergecs1 = (CS*)&(freqCSset->items[tmpParentIdx]);
			mergecs2 = (CS*)&(freqCSset->items[freqId1]);
			//printf("MaxCS: Merge freqCS %d into freqCS %d \n", freqId1, tmpParentIdx);
			mergeConsistsOf(mergecs1, mergecs2);
		}

	}

	

}

static int 
trie_insertwithFreqCS(struct trie_node* root, oid* key, int key_len, int val, CSset *freqCSset)
{

	int i, found_child;
	int	returnvalue; 

	struct trie_node* curr_node = NULL;
	struct trie_node* new_node = NULL;
	struct trie_node* iter = NULL;

	/*
	printf("Insert: \n");
	for (i = 0; i < key_len; i++){
		printf(BUNFMT " ", key[i]);
	}
	printf("\n");
	*/


	/* Negative values nor NULL keys are allowed */
	if(val < 0 || root == NULL)
		return -1; 

	curr_node = root;

	/* iterates over all key's elements. For each one,
	tries to advance in the trie reusing the path. When 
	the reusable part ends, start to add new nodes.
	*/

	for(i=0; i <= key_len; i++){ 
		if(i == key_len){
			returnvalue = curr_node->value;
			if(curr_node->children != NULL){
				// Go to the leaf of this path
				while(curr_node->children != NULL)
					curr_node = curr_node->children; 
				

				freqCSset->items[val].parentFreqIdx = curr_node->value;
			}
			return returnvalue;
		}
	
		found_child = 0; 
		for(iter=curr_node->children; iter != NULL; iter=iter->right)
		{ 
			if(iter->key == key[i])
			{
				found_child = 1;
				curr_node = iter;
				if (iter->value != TRIE_PATH){
					freqCSset->items[iter->value].parentFreqIdx = val; 
				}
				break;
			}
		}

		/* Adds a new node on the trie */
		if(!found_child){	
			new_node = malloc(sizeof(struct trie_node));
			new_node->key = key[i];
			/*If we are in the end of key, this node should get the
			value*/
			new_node->value = i == key_len-1 ? val:TRIE_PATH;	
			new_node->parent = curr_node;
 			new_node->children = NULL; 		// DUC: CHECK. Force the children of newnode to be NULL
			/*Updates the children linked list*/
			new_node->left = NULL;
			new_node->right = curr_node->children; 	
			if(curr_node->children != NULL)
				curr_node->children->left = new_node;
	
			curr_node->children = new_node; 	
	
			/*Next loop iteration consider the new node*/
			curr_node = new_node; 
		}
		else {
			if(i == key_len -1)
				assert(curr_node->key == key[i]);
				//curr_node->key = key[i];
		}
	}

	return 1;
}

void createTreeForCSset(CSset *freqCSset){
	struct trie_node* root;
	int 	i; 	
	int	numFreqCS;
	int	numMaxCS = 0; 
	int	numLeaf = 0; 

	
	printf("Build tree from frequent CSs \n");
	init_trie(&root);
	numFreqCS = freqCSset->numCSadded;

	for(i = 0; i < numFreqCS; i++){
	   	trie_insertwithFreqCS(root,freqCSset->items[i].lstProp, freqCSset->items[i].numProp,i,freqCSset);
	}

	count_trieleaf(&root, &numLeaf);
	printf("Num of leaf is %d \n", numLeaf);

	for(i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){
			numMaxCS++;
		}
	}
		
	printf("Num max freqCSs explored from prefix tree: %d \n", numMaxCS);
	delete_trie(&root);
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

	propStat->maxNumPPerCS = 0; 

	return propStat; 
}



static
void getPropStatisticsFromMergeCSs(PropStat* propStat,int  curNumMergeCS, oid* mergeCSFreqCSMap, CSset* freqCSset){

	int i, j; 
	oid freqId; 
	CS cs;

	for (i = 0; i < curNumMergeCS; i++){
		freqId = mergeCSFreqCSMap[i];
		cs = (CS)freqCSset->items[freqId];

		for (j = 0; j < cs.numProp; j++){
			addaProp(propStat, cs.lstProp[j],freqId, j);
		}

		if (cs.numProp > propStat->maxNumPPerCS)
			propStat->maxNumPPerCS = cs.numProp;
	}

	for (i = 0; i < propStat->numAdded; i++){
		propStat->tfidfs[i] = tfidfComp(propStat->freqs[i],curNumMergeCS);
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
PropStat* getPropStatisticsByTable(int numTables, int* mTblIdxFreqIdxMapping, CSset* freqCSset, int *numdistinctMCS){

	int i, j, k; 
	CS cs;
	int freqId; 

	PropStat* propStat; 
	
	propStat = initPropStat(); 

	k = 0; 

	for (i = 0; i < numTables; i++){
		freqId = mTblIdxFreqIdxMapping[i];
		cs = (CS)freqCSset->items[freqId];
		k++; 
		for (j = 0; j < cs.numProp; j++){
			addaProp(propStat, cs.lstProp[j], i, j);
		}

		if (cs.numProp > propStat->maxNumPPerCS)
			propStat->maxNumPPerCS = cs.numProp;
	}

	/* Do not calculate the TFIDF score. May need in the future  
	 *  
	for (i = 0; i < propStat->numAdded; i++){
		propStat->tfidfs[i] = tfidfComp(propStat->freqs[i],numMaxCSs);
	}
	*/

	*numdistinctMCS = k; 

	return propStat; 
}


void printPropStat(PropStat* propStat, int printToFile){
	int i, j; 
	oid	*pbt; 
	Postinglist ps; 

	FILE 	*fout; 
	char 	filename[100];

	if (printToFile == 0){
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
	else{

		strcpy(filename, "fullPropStat");
		strcat(filename, ".txt");

		fout = fopen(filename,"wt"); 
		fprintf(fout, "PropertyOid #ofCSs ");	
		for (i = 0; i < propStat->numAdded; i++){
			pbt = (oid *) Tloc(propStat->pBat, i);
			fprintf(fout, BUNFMT "  %d\n", *pbt, propStat->plCSidx[i].numAdded);
		}
		fclose(fout);
	
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
		free(propStat->plCSidx[i].lstInvertIdx); 
	}
	free(propStat->plCSidx); 
	free(propStat); 
}

static 
void initMergeCSFreqCSMap(CSset *freqCSset, oid *mergeCSFreqCSMap){
	int i; 
	int	mergeCSid = 0; 

	for (i = 0; i < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){
			mergeCSFreqCSMap[mergeCSid] = i; 
			mergeCSid++;
		}
	}
}

static
CSrelSum* initCSrelSum(int maxNumProp, int maxNumRefPerCS){
	int i; 
	CSrelSum *csRelSum; 	
	csRelSum = (CSrelSum*)malloc(sizeof(CSrelSum)); 
	csRelSum->origFreqIdx = -1;
	csRelSum->numProp = 0;		/* Initially there is no prop */
	csRelSum->lstPropId = (oid*)malloc(sizeof(oid) * maxNumProp); 
	csRelSum->numPropRef = (int*)malloc(sizeof(int) * maxNumProp);		
	csRelSum->freqIdList = (int**)malloc(sizeof(int*) * maxNumProp);
	for (i = 0; i < maxNumProp; i++){
		csRelSum->numPropRef[i] = 0;
		csRelSum->freqIdList[i] = (int*)malloc(sizeof(int) * maxNumRefPerCS);
	}	

	return csRelSum;
}

static 
void freeCSrelSum(int maxNumProp, CSrelSum *csRelSum){
	int i; 
	for (i = 0; i < maxNumProp; i++){
		free(csRelSum->freqIdList[i]);
	}
	free(csRelSum->freqIdList);
	free(csRelSum->numPropRef);
	free(csRelSum->lstPropId);
	free(csRelSum);
}

static 
void generatecsRelSum(CSrel csRel, int freqId, CSset* freqCSset, CSrelSum *csRelSum){
	int i; 
	int propIdx; 
	int refIdx; 
	int freq; 

	csRelSum->origFreqIdx = freqId;
	csRelSum->numProp = freqCSset->items[freqId].numProp;
	copyOidSet(csRelSum->lstPropId, freqCSset->items[freqId].lstProp, csRelSum->numProp);

	for (i = 0; i < csRelSum->numProp; i++){
		csRelSum->numPropRef[i] = 0;
	}

	for (i = 0; i < csRel.numRef; i++){
		freq = freqCSset->items[csRel.origFreqIdx].support; 
		if (freq > MIN_FROMTABLE_SIZE_S6 && freq < csRel.lstCnt[i] * MIN_PERCETAGE_S6){			
			propIdx = 0;
			while (csRelSum->lstPropId[propIdx] != csRel.lstPropId[i])
				propIdx++;
		
			//Add to this prop
			refIdx = csRelSum->numPropRef[propIdx];
			csRelSum->freqIdList[propIdx][refIdx] = csRel.lstRefFreqIdx[i]; 
			csRelSum->numPropRef[propIdx]++;
		}
	}

}

static
LabelStat* initLabelStat(void){
	LabelStat *labelStat = (LabelStat*) malloc(sizeof(LabelStat)); 
	labelStat->labelBat = BATnew(TYPE_void, TYPE_oid, INIT_DISTINCT_LABEL);	
	if (labelStat->labelBat == NULL){
		return NULL; 
	}
	(void)BATprepareHash(BATmirror(labelStat->labelBat));
	if (!(labelStat->labelBat->T->hash)) 
		return NULL; 
	labelStat->lstCount = (int*)malloc(sizeof(int) * INIT_DISTINCT_LABEL);

	labelStat->freqIdList = NULL;	
	labelStat->numLabeladded = 0;
	labelStat->numAllocation = INIT_DISTINCT_LABEL;

	return labelStat; 
}

/*
 * 
 * */
#if USE_ALTERNATIVE_NAME 
static
oid getMostSuitableName(CSlabel *labels, int freqIdx, int candIdx){
	oid candidate; 
	int i; 
	candidate = labels[freqIdx].candidates[candIdx];

	if (labels[freqIdx].hierarchyCount > 1){
		for (i = 0; i < labels[freqIdx].hierarchyCount; i++){
			if (labels[freqIdx].hierarchy[i] == candidate) break;
		}

	}
	
	if (i == labels[freqIdx].hierarchyCount)	// Not appears in the hierarchy
		return candidate; 
	else if (i > TOP_GENERAL_NAME)		// Not a too general candidate
		return candidate; 
	else if ((candIdx+1) < labels[freqIdx].candidatesCount){
		//printf("Use another candidate \n");
		return labels[freqIdx].candidates[candIdx+1];
	}
		
	//No choice			
	return candidate; 

}
#endif

static
void buildLabelStat(LabelStat *labelStat, CSlabel *labels, CSset *freqCSset, int k){
	int 	i,j; 
	BUN 	bun; 
	int 	*_tmp; 
	int	freqIdx;
	int	numDummy = 0; 
	int	numCheck = 0;
	oid	candidate; 

	//Preparation
	for (i = 0; i  < freqCSset->numCSadded; i++){
		if (labels[i].name != BUN_NONE){
			numCheck = (labels[i].candidatesCount > k)?k:labels[i].candidatesCount;
			for (j = 0; j < numCheck; j++){
				#if USE_ALTERNATIVE_NAME
				candidate = getMostSuitableName(labels, i, j);
				#else
				candidate = labels[i].candidates[j];
				#endif
				bun = BUNfnd(BATmirror(labelStat->labelBat),(ptr) &candidate);
				if (bun == BUN_NONE) {
					/*New string*/
					if (labelStat->labelBat->T->hash && BATcount(labelStat->labelBat) > 4 * labelStat->labelBat->T->hash->mask) {
						HASHdestroy(labelStat->labelBat);
						BAThash(BATmirror(labelStat->labelBat), 2*BATcount(labelStat->labelBat));
					}

					labelStat->labelBat = BUNappend(labelStat->labelBat, (ptr) &candidate, TRUE);
							
					if(labelStat->numLabeladded == labelStat->numAllocation) 
					{ 
						labelStat->numAllocation += INIT_DISTINCT_LABEL; 
						
						_tmp = realloc(labelStat->lstCount, (labelStat->numAllocation * sizeof(int)));
					
						if (!_tmp){
							fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						}
						labelStat->lstCount = (int*)_tmp;
					}
					labelStat->lstCount[labelStat->numLabeladded] = 1; 
					labelStat->numLabeladded++;
				}
				else{
					labelStat->lstCount[bun]++;
				}
			}
		}
		else
			numDummy++;
	}
	
	printf("Total number of distinct labels in Top%d is %d \n", k, labelStat->numLabeladded);
	printf("Number of DUMMY freqCS: %d \n",numDummy);
	//Build list of FreqCS
	labelStat->freqIdList = (int**) malloc(sizeof(int*) * labelStat->numLabeladded);
	for (i =0; i < labelStat->numLabeladded; i++){
		labelStat->freqIdList[i] = (int*)malloc(sizeof(int) * labelStat->lstCount[i]);
		//reset the lstCount
		labelStat->lstCount[i] = 0;
	}
	
	for (i = 0; i  < freqCSset->numCSadded; i++){
		if (labels[i].name != BUN_NONE){
			numCheck = (labels[i].candidatesCount > k)?k:labels[i].candidatesCount;
			for (j = 0; j < numCheck; j++){
				#if USE_ALTERNATIVE_NAME
				candidate = getMostSuitableName(labels, i, j);
				#else
				candidate = labels[i].candidates[j];
				#endif
				bun = BUNfnd(BATmirror(labelStat->labelBat),(ptr) &candidate);
				if (bun == BUN_NONE) {
					fprintf(stderr, "All the name should be stored already!\n");
				}
				else{
					freqIdx = labelStat->lstCount[bun];
					labelStat->freqIdList[bun][freqIdx] = i; 
					labelStat->lstCount[bun]++;
				}
			}
		}
	}

}
static 
void freeLabelStat(LabelStat *labelStat){
	int i; 
	if (labelStat->freqIdList != NULL){
		for (i = 0; i < labelStat->numLabeladded;i++){
			free(labelStat->freqIdList[i]);
		} 
		free(labelStat->freqIdList);
	}	
	free(labelStat->lstCount);
	BBPreclaim(labelStat->labelBat);
	free(labelStat);
}

static 
void doMerge(CSset *freqCSset, int ruleNum, CS* cs1, CS* cs2, int freqId1, int freqId2, oid *mergecsId, CSlabel** labels, oid** ontmetadata, int ontmetadataCount, oid name){
	CS 	*mergecs; 
	int		existMergecsId; 
	CS		*existmergecs, *mergecs1, *mergecs2; 
	int	k; 

	//Check whether these CS's belong to any mergeCS
	if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
		mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, *mergecsId);
		//addmergeCStoSet(mergecsSet, *mergecs);
		cs1->parentFreqIdx = freqCSset->numCSadded;
		cs2->parentFreqIdx = freqCSset->numCSadded;
		addCStoSet(freqCSset,*mergecs);
		updateLabel(ruleNum, freqCSset, labels, 1, freqCSset->numCSadded - 1, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
		free(mergecs);
		
		mergecsId[0]++;
	}
	else if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx != -1){
		existMergecsId = cs2->parentFreqIdx;
		existmergecs = &(freqCSset->items[existMergecsId]);
		mergeACStoExistingmergeCS(*cs1,freqId1, existmergecs);
		cs1->parentFreqIdx = existMergecsId; 
		updateLabel(ruleNum, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
	}
	
	else if (cs1->parentFreqIdx != -1 && cs2->parentFreqIdx == -1){
		existMergecsId = cs1->parentFreqIdx;
		existmergecs = &(freqCSset->items[existMergecsId]);
		mergeACStoExistingmergeCS(*cs2,freqId2, existmergecs);
		cs2->parentFreqIdx = existMergecsId; 
		updateLabel(ruleNum, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
	}
	else if (cs1->parentFreqIdx != cs2->parentFreqIdx){
		mergecs1 = &(freqCSset->items[cs1->parentFreqIdx]);
		mergecs2 = &(freqCSset->items[cs2->parentFreqIdx]);
		
		mergeTwomergeCS(mergecs1, mergecs2, cs1->parentFreqIdx);

		//Re-map for all maxCS in mergecs2
		for (k = 0; k < mergecs2->numConsistsOf; k++){
			freqCSset->items[mergecs2->lstConsistsOf[k]].parentFreqIdx = cs1->parentFreqIdx;
		}
		updateLabel(ruleNum, freqCSset, labels, 0, cs1->parentFreqIdx, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
	}

}

static
str mergeMaxFreqCSByS1(CSset *freqCSset, CSlabel** labels, oid *mergecsId, oid** ontmetadata, int ontmetadataCount){
	int 		i; 

	#if !USE_MULTIWAY_MERGING
	int		j, k;
	int 		freqId1, freqId2;
	CS		*cs1, *cs2;
	#else
	int		*lstDistinctFreqId = NULL;		
	int		numDistinct = 0;
	int		isNew = 0; 
	int  		mergeFreqIdx = -1; 
	#endif
	LabelStat	*labelStat = NULL; 
	oid		*name;
	#if OUTPUT_FREQID_PER_LABEL
	FILE    	*fout;
	char*   	schema = "rdf";
	int		ret = 0;
	str		tmpLabel; 
	int		tmpCount; 
	
	#if USE_SHORT_NAMES
	str canStrShort = NULL;
	#endif
	#endif
	(void) name; 
	(void) ontmetadata;
	(void) ontmetadataCount;
	labelStat = initLabelStat(); 
	buildLabelStat(labelStat, (*labels), freqCSset, TOPK);
	printf("Num FreqCSadded before using S1 = %d \n", freqCSset->numCSadded);

	#if OUTPUT_FREQID_PER_LABEL

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	
	fout = fopen("freqIdPerLabel.txt","wt");
	#endif
	for (i = 0; i < labelStat->numLabeladded; i++){
		name = (oid*) Tloc(labelStat->labelBat, i);
		if (labelStat->lstCount[i] > 1){
			/*TODO: Multi-way merge */
			#if USE_MULTIWAY_MERGING	
			lstDistinctFreqId = mergeMultiCS(freqCSset,  labelStat->freqIdList[i], labelStat->lstCount[i], mergecsId, &numDistinct, &isNew, &mergeFreqIdx); 
			if (lstDistinctFreqId != NULL){
				updateLabel(S1, freqCSset, labels, isNew, mergeFreqIdx, -1, -1, *name, ontmetadata, ontmetadataCount, lstDistinctFreqId, numDistinct);
			}
			#else

			//For ontology name
			tmpCount = 0; 
			for (k = 0; k < labelStat->lstCount[i]; k++){
				freqId1 = labelStat->freqIdList[i][k];
				if ((*labels)[freqId1].isOntology == 1) {
					cs1 = &(freqCSset->items[freqId1]);
					#if     NOT_MERGE_DIMENSIONCS
					if (cs1->type == DIMENSIONCS) continue;
					#endif
					tmpCount++;
					break; 
				}
			}
			for (j = k+1; j < labelStat->lstCount[i]; j++){
				freqId2 = labelStat->freqIdList[i][j];
				cs2 = &(freqCSset->items[freqId2]);
				#if	NOT_MERGE_DIMENSIONCS
				if (cs2->type == DIMENSIONCS) 
					continue; 
				#endif
				if ((*labels)[freqId2].isOntology == 1){
					//printf("Merge FreqCS %d and FreqCS %d by Ontology name \n", freqId1, freqId2);
					doMerge(freqCSset, S1, cs1, cs2, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, *name);
					//printf("Number of added cs in freqCS: %d \n", freqCSset->numCSadded); 
					tmpCount++;
				}
			}
			fprintf(fout, " %d freqCS merged as having same name by Ontology. MergedCS has %d prop. \n", tmpCount, freqCSset->items[freqCSset->numCSadded -1].numProp);

			//For Type
			tmpCount = 0;
			for (k = 0; k < labelStat->lstCount[i]; k++){
				freqId1 = labelStat->freqIdList[i][k];
				if ((*labels)[freqId1].isType == 1) {
					cs1 = &(freqCSset->items[freqId1]);
					#if     NOT_MERGE_DIMENSIONCS
					if (cs1->type == DIMENSIONCS) continue;
					#endif
					tmpCount++;
					break; 
				}
			}
			for (j = k+1; j < labelStat->lstCount[i]; j++){
				freqId2 = labelStat->freqIdList[i][j];
				cs2 = &(freqCSset->items[freqId2]);
				#if	NOT_MERGE_DIMENSIONCS
				if (cs2->type == DIMENSIONCS) continue; 
				#endif
				if ((*labels)[freqId2].isType == 1){
					//printf("Merge FreqCS %d and FreqCS %d by Type name \n", freqId1, freqId2);
					doMerge(freqCSset, S1, cs1, cs2, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, *name);
					//printf("Number of added cs in freqCS: %d \n", freqCSset->numCSadded); 				
					tmpCount++;
				}
			}
			fprintf(fout, " %d freqCS merged as having same name by TYPE. MergedCS has %d prop. \n", tmpCount, freqCSset->items[freqCSset->numCSadded -1].numProp);

			//For FK
			tmpCount = 0;
			for (k = 0; k < labelStat->lstCount[i]; k++){
				freqId1 = labelStat->freqIdList[i][k];
				if ((*labels)[freqId1].isFK == 1) {
					cs1 = &(freqCSset->items[freqId1]);
					#if     NOT_MERGE_DIMENSIONCS
					if (cs1->type == DIMENSIONCS) continue;
					#endif
					tmpCount++;
					break; 
				}
			}
			for (j = k+1; j < labelStat->lstCount[i]; j++){
				freqId2 = labelStat->freqIdList[i][j];
				cs2 = &(freqCSset->items[freqId2]);
				#if	NOT_MERGE_DIMENSIONCS
				if (cs2->type == DIMENSIONCS) continue; 
				#endif
				if ((*labels)[freqId2].isFK == 1){
					//printf("Merge FreqCS %d and FreqCS %d by FK name \n", freqId1, freqId2);
					doMerge(freqCSset, S1, cs1, cs2, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, *name);
					//printf("Number of added cs in freqCS: %d \n", freqCSset->numCSadded); 					
					tmpCount++;
				}
			}
			#endif /* USE_MULTIWAY_MERGING */
			fprintf(fout, " %d freqCS merged as having same name by FK. MergedCS has %d prop. \n", tmpCount, freqCSset->items[freqCSset->numCSadded -1].numProp);

			#if OUTPUT_FREQID_PER_LABEL
			
			takeOid(*name, &tmpLabel); 
			#if USE_SHORT_NAMES
			getPropNameShort(&canStrShort, tmpLabel);
			fprintf(fout,"Label %d:  %s \n", i, canStrShort);
			GDKfree(canStrShort);
			#else
			fprintf(fout,"Label %d:  %s \n", i, tmpLabel);
			#endif  /* USE_SHORT_NAMES */
			GDKfree(tmpLabel); 
			fprintf(fout,"Totally contains %d freqCSs. MergedCS has %d prop. \n", labelStat->lstCount[i], freqCSset->items[freqCSset->numCSadded -1].numProp); 
			for (j = 0; j < labelStat->lstCount[i]; j++){
				fprintf(fout," %d  | ", labelStat->freqIdList[i][j]); 
			}
			
		 	fprintf(fout,"\n");
			#endif /*OUTPUT_FREQID_PER_LABEL*/
		}
		

	}

	#if OUTPUT_FREQID_PER_LABEL
	fclose(fout);
	TKNZRclose(&ret);
	#endif

	freeLabelStat(labelStat);

	return MAL_SUCCEED; 
}

static
void mergeMaxFreqCSByS6(CSrel *csrelMergeFreqSet, CSset *freqCSset, CSlabel** labels, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId, oid** ontmetadata, int ontmetadataCount){
	int 		i; 
	int 		freqId, refFreqId;
	//int 		relId; 
	//CS*		cs1;
	CSrelSum 	*csRelSum; 
	int		maxNumRefPerCS = 0; 
	int 		j, k; 
	#if 		!USE_MULTIWAY_MERGING
	int 		freqId1, freqId2;
	CS		*cs1, *cs2;
	#else
	int		*lstDistinctFreqId = NULL;		
	int		numDistinct = 0;
	int		isNew = 0; 
	int  		mergeFreqIdx = -1; 
	#endif	

	char 		filename[100];
	FILE		*fout; 
	int		maxNumPropInMergeCS =0;
	//int 		numCombinedP = 0; 
	int		startIdx = 0; 
	
	printf("Start merging CS by using S6 \n");

	strcpy(filename, "csRelSum.txt");

	fout = fopen(filename,"wt"); 

	for (i = 0; i < curNumMergeCS; i++){
		freqId = mergeCSFreqCSMap[i];
		if (csrelMergeFreqSet[freqId].numRef > maxNumRefPerCS)
		 	maxNumRefPerCS = csrelMergeFreqSet[freqId].numRef ; 		

		if (freqCSset->items[freqId].numProp > maxNumPropInMergeCS)
			maxNumPropInMergeCS = freqCSset->items[freqId].numProp;
	}
	printf("maxNumRefPerCS = %d \n", maxNumRefPerCS);
	printf("max number of prop in mergeCS: %d \n", maxNumPropInMergeCS);

	csRelSum = initCSrelSum(maxNumPropInMergeCS,maxNumRefPerCS);
	
	for (i = 0; i < curNumMergeCS; i++){
		freqId = mergeCSFreqCSMap[i];
		if (csrelMergeFreqSet[freqId].numRef != 0){
			generatecsRelSum(csrelMergeFreqSet[freqId], freqId, freqCSset, csRelSum);
			/* Check the number of */
			fprintf(fout, "csRelSum " BUNFMT " (support: %d, coverage %d ): ",csRelSum->origFreqIdx, freqCSset->items[freqId].support, freqCSset->items[freqId].coverage);
			for (j = 0; j < csRelSum->numProp; j++){
				if ( csRelSum->numPropRef[j] > 1){
					fprintf(fout, "  P " BUNFMT " -->",csRelSum->lstPropId[j]);
					for (k = 0; k < csRelSum->numPropRef[j]; k++){
						refFreqId = csRelSum->freqIdList[j][k];
						fprintf(fout, " %d | ", refFreqId);
					}	
					/* Merge each refCS into the first CS. 
					 * TODO: The Multi-way merging should be better
					 * */ 
					//mergeMultiPropList(freqCSset, csRelSum->freqIdList[j],csRelSum->numPropRef[j] , &numCombinedP);
					#if USE_MULTIWAY_MERGING	
					lstDistinctFreqId = mergeMultiCS(freqCSset, csRelSum->freqIdList[j],csRelSum->numPropRef[j], mergecsId, &numDistinct, &isNew, &mergeFreqIdx); 
					
					if (lstDistinctFreqId != NULL){
						updateLabel(S5, freqCSset, labels, isNew, mergeFreqIdx, -1, -1, BUN_NONE, ontmetadata, ontmetadataCount, lstDistinctFreqId, numDistinct);
					}
					#else

					startIdx = 0;
					#if	NOT_MERGE_DIMENSIONCS
					while(startIdx < csRelSum->numPropRef[j]) {
						freqId1 = csRelSum->freqIdList[j][startIdx];
						cs1 = (CS*) &(freqCSset->items[freqId1]);
						if (cs1->type == DIMENSIONCS) 
							startIdx++;
						else 
							break;
					}
					#else
					freqId1 = csRelSum->freqIdList[j][startIdx];
					cs1 = (CS*) &(freqCSset->items[freqId1]);
					#endif

					for (k = (startIdx+1); k < csRelSum->numPropRef[j]; k++){
						freqId2 = csRelSum->freqIdList[j][k];
						cs2 = (CS*) &(freqCSset->items[freqId2]);

						#if	NOT_MERGE_DIMENSIONCS
						if (cs2->type == DIMENSIONCS) continue; 
						#endif

						doMerge(freqCSset, S5, cs1, cs2, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, BUN_NONE);

					}

					#endif /*If USE_MULTIWAY_MERGING */
				}
			}
			fprintf(fout, "\n");
		}
	}
	


	fclose(fout); 


	freeCSrelSum(maxNumPropInMergeCS, csRelSum);

}

static
char isSemanticSimilar(int freqId1, int freqId2, CSlabel* labels, OntoUsageNode *tree, int numOrigFreqCS, oid *ancestor){	/*Rule S1 S2 S3*/
	int i, j; 
	//int commonHierarchy = -1;
	int minCount = 0; 
	int hCount1, hCount2; 
	int level; 
	OntoUsageNode *tmpNode; 
	/*
	int k1, k2; 
	if (labels[freqId1].name == labels[freqId2].name)
		return 1;
	else{ 
		k1 =  (labels[freqId1].candidatesCount < TOPK)?labels[freqId1].candidatesCount:TOPK;
		k2 =  (labels[freqId2].candidatesCount < TOPK)?labels[freqId2].candidatesCount:TOPK;	

		for (i = 0; i < k1; i++){
			for (j = 0; j < k2; j++){
				if (labels[freqId1].candidates[i] == labels[freqId2].candidates[j])
				{
					(*ancestor) = labels[freqId1].candidates[i];
					return 1; 
				}
			}
		}
	}
	*/

	// Check for the most common ancestor
	hCount1 = labels[freqId1].hierarchyCount;
	hCount2 = labels[freqId2].hierarchyCount;
	minCount = (hCount1 > hCount2)?hCount2:hCount1;
	
	/*
	if (minCount > 0){
	printf("minCount = %d \n", minCount);
	printf("Finding common ancestor for %d and %d \n", freqId1, freqId2 );
	printf("FreqCS1: ");
	for (i = 0; i < hCount1; i++){
		printf("  %s", labels[freqId1].hierarchy[hCount1-1-i]);
	}
	printf(" \n ");
	printf("FreqCS2: ");
	for (i = 0; i < hCount2; i++){
		printf("  %s", labels[freqId2].hierarchy[hCount2-1-i]);
	}
	printf(" \n ");
	}

	*/

	if ((freqId1 > numOrigFreqCS -1) || (freqId2 > numOrigFreqCS -1))
		return 0;

	for (i = 0; i < minCount; i++){
		if (labels[freqId1].hierarchy[hCount1-1-i] != labels[freqId2].hierarchy[hCount2-1-i])
				break;
	}

	//printf("The common ancestor of freqCS %d and %d is at %d (minCount = %d) \n",freqId1, freqId2,i, minCount);
	if (i !=0 && i != minCount){ /*There is a common ancestor at i */
		level = 0;
		tmpNode = tree; 
		while(level < i){
			for (j = 0; j < tmpNode->numChildren; j++) {	
				if (tmpNode->lstChildren[j]->uri == labels[freqId1].hierarchy[hCount1-1-level]){
					tmpNode = tmpNode->lstChildren[j];
					break;
				}
			}
			level++;
		}
		//printf("The common ancestor of freqCS %d (%s) and freqCS %d (%s) is: %s --- %f \n", freqId1, labels[freqId1].name, freqId2, labels[freqId2].name, tmpNode->uri, tmpNode->percentage);
		if (tmpNode->percentage < IMPORTANCE_THRESHOLD) {
			//printf("Merge two CS's %s and %s using the common ancestor (%s) at level %d (score: %f)\n",labels[freqId1].name,labels[freqId2].name,tmpNode->uri, i,tmpNode->percentage);
			(*ancestor) = tmpNode->uri;
			return 1;
		}

	}


	return 0;
}

static
void mergeCSByS3S5(CSset *freqCSset, CSlabel** labels, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId,OntoUsageNode *ontoUsageTree, oid **ontmetadata, int ontmetadataCount){
	int 		i, j, k; 
	int 		freqId1, freqId2; 
	float 		simscore = 0.0; 
	CS     		*mergecs;
	int		existMergecsId; 
	int 		numCombineP = 0; 
	CS		*cs1, *cs2;
	CS		*existmergecs, *mergecs1, *mergecs2; 

	PropStat	*propStat; 	/* Store statistics about properties */
	char		isLabelComparable = 0; 
	char		isSameLabel = 0; 
	oid		name;		/* Name of the common ancestor */

	

	
	(void) labels;
	(void) isLabelComparable;


	propStat = initPropStat();
	getPropStatisticsFromMergeCSs(propStat, curNumMergeCS, mergeCSFreqCSMap, freqCSset); /*TODO: Get PropStat from MaxCSs or From mergedCS only*/

	for (i = 0; i < curNumMergeCS; i++){		
		freqId1 = mergeCSFreqCSMap[i];
		//printf("Label of %d CS is %s \n", freqId1, (*labels)[freqId1].name);
		isLabelComparable = 0; 
		if ((*labels)[freqId1].name != BUN_NONE) isLabelComparable = 1; // no "DUMMY"

		cs1 = (CS*) &(freqCSset->items[freqId1]);
				
		#if	NOT_MERGE_DIMENSIONCS
		if (cs1->type == DIMENSIONCS) continue; 
		#endif
	 	for (j = (i+1); j < curNumMergeCS; j++){
			freqId2 = mergeCSFreqCSMap[j];
			cs2 = (CS*) &(freqCSset->items[freqId2]);
			#if	NOT_MERGE_DIMENSIONCS
			if (cs2->type == DIMENSIONCS) continue; 
			#endif
			isSameLabel = 0; 

			#if	USE_LABEL_FOR_MERGING
			if (isLabelComparable == 1 && isSemanticSimilar(freqId1, freqId2, (*labels), ontoUsageTree,freqCSset->numOrigFreqCS, &name) == 1){
				//printf("Same labels between freqCS %d and freqCS %d - Old simscore is %f \n", freqId1, freqId2, simscore);
				isSameLabel = 1;
				simscore = 1; 
			}
			#endif

			if (isSameLabel == 0){
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
			}
			
			//simscore = 0.0;
			#if	USINGTFIDF	
			if (simscore > SIM_TFIDF_THRESHOLD){
			#else	
			if (simscore > SIM_THRESHOLD) {
			#endif		
				//printf("S3S5: merge freqCS %d and freqCS %d \n", freqId1, freqId2);
				//Check whether these CS's belong to any mergeCS
				if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
					mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, *mergecsId);
					//addmergeCStoSet(mergecsSet, *mergecs);
					cs1->parentFreqIdx = freqCSset->numCSadded;
					cs2->parentFreqIdx = freqCSset->numCSadded;
					addCStoSet(freqCSset,*mergecs);
					if (isSameLabel) {
						// rule S2
						updateLabel(S2, freqCSset, labels, 1, freqCSset->numCSadded - 1, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
					} else {
						// rule S4
						updateLabel(S4, freqCSset, labels, 1, freqCSset->numCSadded - 1, freqId1, freqId2, BUN_NONE, ontmetadata, ontmetadataCount, NULL, -1);
					}
					free(mergecs);

					mergecsId[0]++;


				}
				else if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx != -1){
					existMergecsId = cs2->parentFreqIdx;
					existmergecs = (CS*) &(freqCSset->items[existMergecsId]);
					mergeACStoExistingmergeCS(*cs1,freqId1, existmergecs);
					cs1->parentFreqIdx = existMergecsId; 
					if (isSameLabel) {
						// rule S2
						updateLabel(S2, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
					} else {
						// rule S4
						updateLabel(S4, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, BUN_NONE, ontmetadata, ontmetadataCount, NULL, -1);
					}
				}
				
				else if (cs1->parentFreqIdx != -1 && cs2->parentFreqIdx == -1){
					existMergecsId = cs1->parentFreqIdx;
					existmergecs = (CS*)&(freqCSset->items[existMergecsId]);
					mergeACStoExistingmergeCS(*cs2,freqId2, existmergecs);
					cs2->parentFreqIdx = existMergecsId; 
					if (isSameLabel) {
						// rule S2
						updateLabel(S2, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
					} else {
						// rule S4
						updateLabel(S4, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, BUN_NONE, ontmetadata, ontmetadataCount, NULL, -1);
					}
				}
				else if (cs1->parentFreqIdx != cs2->parentFreqIdx){
					mergecs1 = (CS*)&(freqCSset->items[cs1->parentFreqIdx]);
					mergecs2 = (CS*)&(freqCSset->items[cs2->parentFreqIdx]);
					
					mergeTwomergeCS(mergecs1, mergecs2, cs1->parentFreqIdx);

					//Re-map for all maxCS in mergecs2
					for (k = 0; k < mergecs2->numConsistsOf; k++){
						freqCSset->items[mergecs2->lstConsistsOf[k]].parentFreqIdx = cs1->parentFreqIdx;
					}
					if (isSameLabel) {
						// rule S2
						updateLabel(S2, freqCSset, labels, 0, cs1->parentFreqIdx, freqId1, freqId2, name, ontmetadata, ontmetadataCount, NULL, -1);
					} else {
						// rule S4
						updateLabel(S4, freqCSset, labels, 0, cs1->parentFreqIdx, freqId1, freqId2, BUN_NONE, ontmetadata, ontmetadataCount, NULL, -1);
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

static void getStatisticFinalCSs(CSset *freqCSset, BAT *sbat, int freqThreshold, int numTables, int* mergeCSFreqCSMap, CSPropTypes* csPropTypes){

	//int 	*csPropNum; 
	//int	*csFreq; 
	FILE 	*fout; 
	int	i,j, k ; 
	char 	filename[100];
	char 	tmpStr[20];
	int	maxNumtriple = 0; 
	int	minNumtriple = INT_MAX; 
	int	numMergeCS = 0; 
	int 	totalCoverage = 0; 
	int	totalCoverage10[10]; 
	int	tmpNumProp10[10], maxNumProp10[10]; 
	int	freqId; 
	int	maxNumProp, tmpNumProp; 

	printf("Get statistics of final CSs ....");

	strcpy(filename, "finalCSStatistic");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 
	fprintf(fout, " csId  #Prop   #frequency #coverage\n"); 

	for (i = 0; i < numTables; i++){
		freqId = mergeCSFreqCSMap[i]; 
		if (isCSTable(freqCSset->items[freqId])){		// Check whether it is a maximumCS
			// Output the result 
			fprintf(fout, BUNFMT " %d  %d  %d\n", freqCSset->items[freqId].csId, freqCSset->items[freqId].numProp,freqCSset->items[freqId].support, freqCSset->items[freqId].coverage); 
			if (freqCSset->items[freqId].coverage > maxNumtriple) maxNumtriple = freqCSset->items[freqId].coverage;
			if (freqCSset->items[freqId].coverage < minNumtriple) minNumtriple = freqCSset->items[freqId].coverage;
			
			totalCoverage += freqCSset->items[freqId].coverage;
			numMergeCS++;
		}
	}
	
	fclose(fout); 
	printf("\nTotal " BUNFMT " triples, coverred by %d final CSs: %d  (%f percent) \n", BATcount(sbat), numTables, totalCoverage, 100 * ((float)totalCoverage/BATcount(sbat)));
	printf("Max number of triples coverred by one final CS: %d \n", maxNumtriple);
	printf("Min number of triples coverred by one final CS: %d \n", minNumtriple);
	printf("Avg number of triples coverred by one final CS: %f \n", (float)(totalCoverage/numMergeCS));

	//Check if remove all non-frequent Prop
	maxNumtriple = 0;
	minNumtriple = INT_MAX;
	maxNumProp = 0; 
	tmpNumProp = 0;
	for (k = 1; k < 10; k++) {
		totalCoverage10[k] = totalCoverage;
		maxNumProp10[k] = 0; 
	}
	for (i = 0; i < numTables; i++){
		freqId = mergeCSFreqCSMap[i]; 
		if (isCSTable(freqCSset->items[freqId])){		// Check whether it is a maximumCS
			// Output the result 
			tmpNumProp = freqCSset->items[freqId].numProp;	
			for (k = 1; k < 10; k++) {
				tmpNumProp10[k] = freqCSset->items[freqId].numProp;
			}
			for (j = 0; j < freqCSset->items[freqId].numProp; j++){
				//Check infrequent Prop
				if (isInfrequentProp(csPropTypes[i].lstPropTypes[j], freqCSset->items[freqId])){
					totalCoverage = totalCoverage -  csPropTypes[i].lstPropTypes[j].propCover;
					tmpNumProp--; 
				}

				for (k = 1; k < 10; k++) {
					if ((csPropTypes[i].lstPropTypes[j].propFreq * k)  < freqCSset->items[freqId].support * INFREQ_PROP_THRESHOLD){
						totalCoverage10[k] = totalCoverage10[k] - csPropTypes[i].lstPropTypes[j].propCover;
						tmpNumProp10[k]--; 
					};
				}
			}

			if (tmpNumProp > maxNumProp) maxNumProp = tmpNumProp; 
			for (k = 1; k < 10; k++){
				if (tmpNumProp10[k] > maxNumProp10[k]) maxNumProp10[k] = tmpNumProp10[k]; 
			}
		}
	}

	printf("If Removing all INFREQUENT Prop \n");
	printf("Max number of props: %d \n", maxNumProp);
	printf("Total " BUNFMT " triples, coverred by final CSs: %d  (%f percent) \n", BATcount(sbat), totalCoverage, 100 * ((float)totalCoverage/BATcount(sbat)));

	printf("If Removing all INFREQUENT Prop (k times smaller threshold) \n");
	for (k = 1; k < 10; k++) {
		printf("k = %d  |  Max # props: %d | coverred by final CSs: %d  (%f percent)  \n", k, maxNumProp10[k], totalCoverage10[k],  100 * ((float)totalCoverage10[k]/BATcount(sbat)));
	}

	//Check if remove all the final CS covering less than 10000 triples
	
	totalCoverage = 0;
	maxNumtriple = 0;
	minNumtriple = INT_MAX;
	numMergeCS = 0;

	for (i = 0; i < numTables; i++){
		freqId = mergeCSFreqCSMap[i]; 
		if (isCSTable(freqCSset->items[freqId])){		// Check whether it is a maximumCS
			// Output the result 
			if (freqCSset->items[freqId].coverage > maxNumtriple) maxNumtriple = freqCSset->items[freqId].coverage;
			if (freqCSset->items[freqId].coverage < minNumtriple) minNumtriple = freqCSset->items[freqId].coverage;
			
			totalCoverage += freqCSset->items[freqId].coverage;
			numMergeCS++;
		}
	}
	
	printf("IF Removing all the 'SMALL' final CSs  ==> Only %d final CSs \n", numMergeCS);
	printf("Total " BUNFMT " triples, coverred by final CSs: %d  (%f percent) \n", BATcount(sbat), totalCoverage, 100 * ((float)totalCoverage/BATcount(sbat)));
	printf("Max number of triples coverred by one final CS: %d \n", maxNumtriple);
	printf("Min number of triples coverred by one final CS: %d \n", minNumtriple);
	if (numMergeCS != 0) printf("Avg number of triples coverred by one final CS: %f \n", (float)(totalCoverage/numMergeCS));

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

static
BAT* generateTablesForEvaluating(CSset *freqCSset, int numTbl, int* mergeCSFreqCSMap, int curNumMergeCS){
	int	*cumDist; 
	int	totalCoverage = 0; 
	int	curCoverage = 0;
	int	randValue = 0;
	int	tmpIdx; 
	int	freqId; 
	BAT	*outputBat; 
	int	minIdx, maxIdx; 
	int	i;
	BUN	bun = BUN_NONE;
	int	numLoop; 

	cumDist = (int*)malloc(sizeof(int) * curNumMergeCS);
	outputBat = BATnew(TYPE_void, TYPE_int, numTbl);
	if (outputBat == NULL){
		return NULL; 
	}
	(void)BATprepareHash(BATmirror(outputBat));
	if (!(outputBat->T->hash)) 
		return NULL; 

	for (i = 0; i < curNumMergeCS; i++){		
		freqId = mergeCSFreqCSMap[i];
		totalCoverage += freqCSset->items[freqId].coverage; 
	}

	for (i = 0; i < curNumMergeCS; i++){		
		freqId = mergeCSFreqCSMap[i];
		//printf("CS%d --> cover %d  | ",i,freqCSset->items[freqId].coverage);
		curCoverage += freqCSset->items[freqId].coverage; 
		cumDist[i] = curCoverage; 
	}

	srand(123456);
	i = 0; 
	numLoop = 0;
	while(i < numTbl){
		//Get the index of freqCS for a random value [0-> totalCoverage -1]
		//Using binary search
		randValue = rand() % totalCoverage; 
		minIdx = 0;
		maxIdx = curNumMergeCS - 1;
			
		if (randValue < cumDist[minIdx]){
			tmpIdx = minIdx; 
		}
		
		while ((maxIdx - minIdx) > 1){
			tmpIdx = minIdx + (maxIdx - minIdx)/2;
			if (randValue > cumDist[tmpIdx] ){
				minIdx =  minIdx + (maxIdx - minIdx)/2;
			}
			else{
				maxIdx =  minIdx + (maxIdx - minIdx)/2;
			}
		}

		tmpIdx = maxIdx; 

		//printf("tmpIdx = %d --> FreqCS %d \n",tmpIdx, output[i]);
		bun = BUNfnd(BATmirror(outputBat),(ptr) &tmpIdx);
		if (bun == BUN_NONE) {
			/*New FreqIdx*/
			if (outputBat->T->hash && BATcount(outputBat) > 4 * outputBat->T->hash->mask) {
				HASHdestroy(outputBat);
				BAThash(BATmirror(outputBat), 2*BATcount(outputBat));
			}
			outputBat = BUNappend(outputBat, (ptr) &tmpIdx, TRUE);
			i++;
		}
		numLoop++;
	}

	//Print the results
	printf("Get the sample tables after %d loop \n",numLoop );

	free(cumDist); 

	return outputBat; 
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

	PropStat *fullPropStat; 	

	fullPropStat = initPropStat();
	
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
				returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs, fullPropStat); 
				#else
				returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset, fullPropStat); 
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
	returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs, fullPropStat); 
	#else
	returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, &CSoid, 1, *freqThreshold, freqCSset, fullPropStat ); 
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

	#if FULL_PROP_STAT
	printPropStat(fullPropStat,1);
	#endif	

	freePropStat(fullPropStat); 

	*ret = 1; 

	//Update the numOrigFreqCS for freqCS
	freqCSset->numOrigFreqCS = freqCSset->numCSadded; 

	return MAL_SUCCEED; 
}


static 
str RDFgetRefCounts(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, int maxNumProp, BUN maxSoid, int *refCount){

	BUN 		p, q; 
	oid 		*sbt, *pbt, *obt; 
	oid 		curS; 		/* current Subject oid */
	oid 		curP; 		/* current Property oid */
	int 		numP; 		/* Number of properties for current S */
	oid*		buff; 	 

	char 		objType;
	oid		realObjOid; 	

	buff = (oid *) malloc (sizeof(oid) * maxNumProp);

	numP = 0;
	curP = 0; 
	curS = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			curS = *sbt; 
			curP = 0;
			numP = 0;
		}
			
		obt = (oid *) BUNtloc(oi, p);
		objType = getObjType(*obt);
	
		pbt = (oid *) BUNtloc(pi, p);

		/* Look at the referenced CS Id using subjCSMap */
		if (objType == URI || objType == BLANKNODE){
			realObjOid = (*obt) - ((oid) objType << (sizeof(BUN)*8 - 4));

			if (realObjOid <= maxSoid && subjCSMap[realObjOid] != BUN_NONE){
				refCount[subjCSMap[realObjOid]]++; 
			}

		}

		if (curP != *pbt){	/* Multi values property */		
			buff[numP] = *pbt; 
			numP++; 
			curP = *pbt; 
		}

		
	}
	
	free (buff); 

	*ret = 1; 

	return MAL_SUCCEED; 
}

#if NEEDSUBCS
static 
str RDFrelationships(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi,  
		oid *subjCSMap, oid *subjSubCSMap, SubCSSet *csSubCSSet, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup,int *csIdFreqIdxMap){
#else
static
str RDFrelationships(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi,
		oid *subjCSMap, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup,int *csIdFreqIdxMap){
#endif	
	BUN	 	p, q; 
	oid 		*sbt = 0, *obt, *pbt;
	oid 		curS; 		/* current Subject oid */
	//oid 		CSoid = 0; 	/* Characteristic set oid */
	int 		numPwithDup;	/* Number of properties for current S */
	char 		objType;
	#if NEEDSUBCS
	oid 		returnSubCSid; 
	#endif
	char* 		buffTypes; 
	oid		realObjOid; 	
	char 		isBlankNode; 
	oid		curP;
	int		from, to; 



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
		from = csIdFreqIdxMap[subjCSMap[*sbt]];
		if ( from == -1) continue; /* Do not consider infrequentCS */
		if (*sbt != curS){
			#if NEEDSUBCS
			if (p != 0){	/* Not the first S */
				returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[curS], csSubCSSet);

				//Get the subCSId
				subjSubCSMap[curS] = returnSubCSid; 

			}
			#endif
			curS = *sbt; 
			numPwithDup = 0;
			curP = 0; 
		}
				
		pbt = (oid *) BUNtloc(pi, p);

		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = getObjType(*obt);

		/* Look at the referenced CS Id using subjCSMap */
		isBlankNode = 0;
		if (objType == URI || objType == BLANKNODE){
			realObjOid = (*obt) - ((oid) objType << (sizeof(BUN)*8 - 4));

			/* Only consider references to freqCS */	
			if (realObjOid <= maxSoid && subjCSMap[realObjOid] != BUN_NONE && csIdFreqIdxMap[subjCSMap[realObjOid]] != -1){
				to = csIdFreqIdxMap[subjCSMap[realObjOid]];
				if (objType == BLANKNODE) isBlankNode = 1;
				addReltoCSRel(from, to, *pbt, &csrelSet[from], isBlankNode);
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
	
	#if NEEDSUBCS
	/* Check for the last CS */
	returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[*sbt], csSubCSSet);
	subjSubCSMap[*sbt] = returnSubCSid; 
	#endif

	free (buffTypes); 

	*ret = 1; 

	return MAL_SUCCEED; 
}


/* 
 * Add highly referred CS to freqCSset, and update the frequency + coverage for each freqCS
 * */
static
str addHighRefCSsToFreqCS(BAT *pOffsetBat, BAT *freqBat, BAT *coverageBat, BAT *fullPBat, 
		int* refCount, CSset *freqCSset, int *csIdFreqIdxMap, int numCS, int threshold){
	int 	i; 
	int 	numP; 
	CS      *freqCS;
	BATiter pi, freqi, coveri; 
	int	freqId = -1; 
	int 	*freq, *coverage;
	oid	*buffP; 
	oid	*offset, *offset2;
	int	totalCoverByFreqCS = 0;

	
	pi = bat_iterator(pOffsetBat);
	freqi = bat_iterator(freqBat);
	coveri = bat_iterator(coverageBat); 
	
	assert((BUN)numCS == BATcount(pOffsetBat)); 
	for (i = 0; i < numCS; i ++){			
		//printf("refCount[%d] = %d \n", i,refCount[i]);
		freqId = csIdFreqIdxMap[i];
		if (freqId == -1){  /* Not a freqCS */
			if (refCount[i] > threshold){	/* Add highly referred CS to freqCSset*/
				
				offset = (oid *) BUNtloc(pi, i);		

				if ((BUN)(i+1) != BUNlast(pOffsetBat)){
					offset2 = (oid *)BUNtloc(pi, (BUN)i + 1);
					numP = *offset2 - *offset;
				}
				else	//Last element
					numP = BUNlast(fullPBat) - *offset;

				freq = (int *) BUNtloc(freqi, (BUN)i); 
				coverage = (int *) BUNtloc(coveri, (BUN)i); 	

			
				buffP = (oid *)Tloc(fullPBat, *offset);	
				#if STOREFULLCS
				/* use BUN_NONE for subjId --> do not have this information*/
				freqCS = creatCS(i, freqCSset->numCSadded, numP, buffP, BUN_NONE, NULL, FREQCS,-1,  *freq,*coverage);
				#else
				freqCS = creatCS(i, freqCSset->numCSadded, numP, buffP, FREQCS,-1,*freq,*coverage);
				#endif
				//printf("Add highly referred CS \n"); 
				addCStoSet(freqCSset, *freqCS);
				csIdFreqIdxMap[i] = freqCSset->numCSadded - 1; 
				free(freqCS);

			}
		}
		else{	/* Update */
			freq = (int *) BUNtloc(freqi, (BUN)i);
			coverage = (int *) BUNtloc(coveri, (BUN)i);
			freqCSset->items[freqId].support = *freq;
			freqCSset->items[freqId].coverage = *coverage;
		}
	}

	/* Update number of original FreqCS*/
	freqCSset->numOrigFreqCS = freqCSset->numCSadded;

	/* Check the total number of triples covered by freqCS */	
	for (i = 0; i < freqCSset->numOrigFreqCS; i++){
		CS cs = (CS)freqCSset->items[i];
		totalCoverByFreqCS += cs.coverage; 
	}

	printf("Total coverage by freq CS's: %d \n", totalCoverByFreqCS);

	return MAL_SUCCEED; 
}


static 
str RDFExtractCSPropTypes(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi,  
		oid *subjCSMap, int* csTblIdxMapping, CSPropTypes* csPropTypes, int maxNumPwithDup){

	BUN	 	p, q; 
	oid 		*sbt = 0, *obt, *pbt;
	oid 		curS; 		/* current Subject oid */
	//oid 		CSoid = 0; 	/* Characteristic set oid */
	int 		numPwithDup;	/* Number of properties for current S */
	int*		buffCoverage;	/* Number of triples coverage by each property. For deciding on MULTI-VALUED P */
	char 		objType;
	char* 		buffTypes; 
	int		**buffTypesCoverMV; /*Store the types of each value in a multi-value prop */		
	oid*		buffP;
	oid		curP; 
	int 		i;

	buffTypes = (char *) malloc(sizeof(char) * (maxNumPwithDup + 1)); 
	buffTypesCoverMV = (int **)malloc(sizeof(int*) * (maxNumPwithDup + 1));
	for (i = 0; i < (maxNumPwithDup + 1); i++){
		buffTypesCoverMV[i] = (int *) malloc(sizeof(int) * (MULTIVALUES)); 
	}
	buffP = (oid *) malloc(sizeof(oid) * (maxNumPwithDup + 1));
	buffCoverage = (int *)malloc(sizeof(int) * (maxNumPwithDup + 1));

	numPwithDup = 0;
	curS = 0; 
	curP = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				addPropTypes(buffTypes, buffP, numPwithDup, buffCoverage, buffTypesCoverMV, subjCSMap[curS], csTblIdxMapping, csPropTypes);
			}
			curS = *sbt; 
			numPwithDup = 0;
			curP = 0; 
		}
				
		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = getObjType(*obt);	/* Get two bits 63th, 62nd from object oid */
		
		if (objType == BLANKNODE){	//BLANKNODE object values will be stored in the same column with URI object values	
			objType = URI; 
		}

		pbt = (oid *) BUNtloc(pi, p);

		if (curP == *pbt){
			#if USE_MULTIPLICITY == 1	
			// Update the object type for this P as MULTIVALUES	
			buffTypes[numPwithDup-1] = MULTIVALUES; 
			buffCoverage[numPwithDup-1]++;
			buffTypesCoverMV[numPwithDup-1][(int)objType]++;
			#else
			buffTypes[numPwithDup] = objType;
			numPwithDup++;
			#endif
		}
		else{			
			for (i = 0; i < MULTIVALUES; i++){
				buffTypesCoverMV[numPwithDup][i] = 0;
			}
			buffTypesCoverMV[numPwithDup][(int)objType] = 1;
			buffTypes[numPwithDup] = objType; 
			buffP[numPwithDup] = *pbt;
			buffCoverage[numPwithDup] = 1; 
			numPwithDup++; 
			curP = *pbt; 
		}


	}
	
	/* Check for the last CS */
	addPropTypes(buffTypes, buffP, numPwithDup, buffCoverage,buffTypesCoverMV, subjCSMap[curS], csTblIdxMapping, csPropTypes);

	for (i = 0; i < (maxNumPwithDup + 1); i++){
		free(buffTypesCoverMV[i]);
	}
	free(buffTypesCoverMV); 

	free (buffTypes); 
	free (buffP); 
	free (buffCoverage);

	*ret = 1; 

	return MAL_SUCCEED; 
}
static 
void initSampleData(CSSample *csSample,BAT *candBat,CSset *freqCSset, int *mergeCSFreqCSMap, CSlabel *label){
	int 	i, j, k; 
	int	numCand = 0; 
	int	freqId; 
	int	*tblId; 
	CS	cs; 
	int	tmpNumcand; 
	oid	tmpCandidate; 
	int 	randValue = 0; 

	numCand = BATcount(candBat); 
	srand(123456); 
	for (i = 0; i < numCand; i++){
		tblId = (int*) Tloc(candBat, i); 
		freqId = mergeCSFreqCSMap[*tblId];
		cs = freqCSset->items[freqId];
		csSample[i].freqIdx = freqId;
		tmpNumcand = (NUM_SAMPLE_CANDIDATE > label[freqId].candidatesCount)?label[freqId].candidatesCount:NUM_SAMPLE_CANDIDATE;
		csSample[i].name = label[freqId].name; 
		csSample[i].candidateCount = tmpNumcand;
		csSample[i].candidates = (oid*)malloc(sizeof(oid) * tmpNumcand); 
		for (k = 0; k < tmpNumcand; k++){
			csSample[i].candidates[k] = label[freqId].candidates[k]; 
		}
		//Randomly exchange the value, change the position k with a random pos
		for (k = 0; k < tmpNumcand; k++){
			randValue = rand() % tmpNumcand;
			tmpCandidate = csSample[i].candidates[k];
			csSample[i].candidates[k] = csSample[i].candidates[randValue];
			csSample[i].candidates[randValue] = tmpCandidate;
		}

		csSample[i].numProp = cs.numProp;
		csSample[i].lstProp = (oid*)malloc(sizeof(oid) * cs.numProp); 
		memcpy(csSample[i].lstProp, cs.lstProp, cs.numProp * sizeof(oid));
		csSample[i].lstSubjOid = (oid*)malloc(sizeof(oid) * NUM_SAMPLE_INSTANCE);
		for (k = 0; k < NUM_SAMPLE_INSTANCE; k++)
			csSample[i].lstSubjOid[k] = BUN_NONE; 

		csSample[i].lstObj = (oid**)malloc(sizeof(oid*) * cs.numProp); 
		for (j = 0; j < cs.numProp; j++){
			csSample[i].lstObj[j] = (oid*)malloc(sizeof(oid) * NUM_SAMPLE_INSTANCE);
			for (k = 0; k < NUM_SAMPLE_INSTANCE; k++)
				csSample[i].lstObj[j][k] = BUN_NONE; 
		}
		csSample[i].numInstances = 0;

	}
}
static 
void freeSampleData(CSSample *csSample, int numCand){
	int i, j; 
	for (i = 0; i < numCand; i++){
		free(csSample[i].lstProp);
		free(csSample[i].candidates); 
		free(csSample[i].lstSubjOid);
		for (j = 0; j < csSample[i].numProp; j++){
			free(csSample[i].lstObj[j]);
		}
		free(csSample[i].lstObj);
	}

	free(csSample);
}

static 
void addSampleInstance(oid subj, oid *buffO, oid* buffP, int numP, int sampleIdx, CSSample *csSample){
	int i,j; 
	int curPos; 
	
	j = 0;
	curPos= csSample[sampleIdx].numInstances;
	csSample[sampleIdx].lstSubjOid[curPos] = subj;
	for (i = 0; i < numP; i++){
		//printf("  P: " BUNFMT " Type: %d ", buffP[i], buffTypes[i]);
		while (csSample[sampleIdx].lstProp[j] != buffP[i]){
			j++;
		}	
		assert(j < csSample[sampleIdx].numProp);
		//j is position of the property buffP[i] in csPropTypes[tblId]
		csSample[sampleIdx].lstObj[j][curPos] = buffO[i]; 
	}
	csSample[sampleIdx].numInstances++;
}

static 
void getObjStr(BAT *mapbat, BATiter mapi, oid objOid, str *objStr, char *retObjType){
	BUN bun; 

	char objType = getObjType(objOid); 

	if (objType == URI || objType == BLANKNODE){
		objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
		takeOid(objOid, objStr); 
	}
	else{
		objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */ 
		bun = BUNfirst(mapbat);
		*objStr = (str) BUNtail(mapi, bun + objOid); 
	}

	*retObjType = objType; 




}

//Assume Tokenizer is openned 
//
void getTblName(char *name, oid nameId){
	str canStr = NULL; 
	str canStrShort = NULL;
	char    *pch;

	if (nameId != BUN_NONE){
		takeOid(nameId, &canStr);
		getPropNameShort(&canStrShort, canStr);

		if (strstr (canStrShort,".") != NULL || 
			strcmp(canStrShort,"") == 0 || 
			strstr(canStrShort,"-") != NULL	){	// WEBCRAWL specific problem with Table name a.jpg, b.png....

			strcpy(name,"NONAME");
		}
		else {
			pch = strstr (canStrShort,"(");
			if (pch != NULL) *pch = '\0';	//Remove (...) characters from table name
		}

		GDKfree(canStr);
		if (strlen(canStrShort) < 50){
			strcpy(name,canStrShort);
		}
		else{
			strncpy (name, canStrShort, 50);
		}

		GDKfree(canStrShort); 
	}
	else 
		strcpy(name,"NONAME");


}

static 
str printSampleData(CSSample *csSample, CSset *freqCSset, BAT *mbat, int num){

	int 	i,j, k; 
	FILE 	*fout, *fouttb, *foutis; 
	char 	filename[100];
	char 	tmpStr[20];
	int 	ret;

	str 	propStr; 
	str	subjStr; 
	char*   schema = "rdf";
	CSSample	sample; 
	CS		freqCS; 
	char	objType = 0; 
	str	objStr; 	
	oid	objOid = BUN_NONE; 
	BATiter mapi;
	str	canStr; 
	char	isTitle = 0; 
	char	isUrl = 0;
	char 	isType = 0;
	char	isDescription = 0;
	char 	isImage = 0; 
	char	isSite = 0;
	char	isEmail = 0; 
	char 	isCountry = 0; 
	char 	isLocality = 0;
#if USE_SHORT_NAMES
	str	propStrShort = NULL;
	char 	*pch; 
#endif

	mapi = bat_iterator(mbat);

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	

	strcpy(filename, "sampleData");
	sprintf(tmpStr, "%d", num);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");
	

	fout = fopen(filename,"wt"); 
	fouttb = fopen("createSampleTable.sh","wt");
	foutis = fopen("loadSampleToMonet.sh","wt");

	for (i = 0; i < num; i++){
		sample = csSample[i];
		freqCS = freqCSset->items[sample.freqIdx];
		fprintf(fout,"Sample table %d Candidates: ", i);
		for (j = 0; j < (int)sample.candidateCount; j++){
			//fprintf(fout,"  "  BUNFMT,sample.candidates[j]);
			if (sample.candidates[j] != BUN_NONE){
#if USE_SHORT_NAMES
				str canStrShort = NULL;
#endif
				takeOid(sample.candidates[j], &canStr); 
#if USE_SHORT_NAMES
				getPropNameShort(&canStrShort, canStr);
				fprintf(fout,";%s",  canStrShort);
				GDKfree(canStrShort);
#else
				fprintf(fout,";%s",  canStr);
#endif
				GDKfree(canStr); 
			
			}
		}
		fprintf(fout, "\n");
		

		if (sample.name != BUN_NONE){
			str canStrShort = NULL;
			takeOid(sample.name, &canStr);
			getPropNameShort(&canStrShort, canStr);

			if (strstr (canStrShort,".") != NULL || 
				strcmp(canStrShort,"") == 0 || 
				strstr(canStrShort,"-") != NULL	){	// WEBCRAWL specific problem with Table name a.jpg, b.png....
				fprintf(fouttb,"CREATE TABLE tbSample%d \n (\n", i);			
			}
			else if (strcmp(canStrShort,"page") == 0){
				fprintf(fouttb,"CREATE TABLE %s%d \n(\n",  canStrShort, i);
			}
			else {
				pch = strstr (canStrShort,"(");
				if (pch != NULL) *pch = '\0';	//Remove (...) characters from table name
				fprintf(fouttb,"CREATE TABLE %s \n(\n",  canStrShort);
			}

			GDKfree(canStrShort);
			GDKfree(canStr);
		}
		else 
			fprintf(fouttb,"CREATE TABLE tbSample%d \n (\n", i);

		//List of columns
		fprintf(fout,"Subject");
		fprintf(fouttb,"SubjectCol string");
		isTitle = 0; 
		isUrl = 0;
		isType = 0; 
		isDescription = 0; 
		isImage = 0;
		isSite = 0; 
		for (j = 0; j < sample.numProp; j++){
			if (freqCS.lstPropSupport[j] * 100 < freqCS.support * SAMPLE_FILTER_THRESHOLD) continue; 
#if USE_SHORT_NAMES
			propStrShort = NULL;
#endif
			takeOid(sample.lstProp[j], &propStr);	
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			fprintf(fout,";%s", propStrShort);

			pch = strstr (propStrShort,"-");
			if (pch != NULL) *pch = '\0';	//Remove - characters from prop  //WEBCRAWL specific problem

			if ((strcmp(propStrShort,"type") == 0 && isType == 1)|| 
					strcmp(propStrShort,"position") == 0 ||
					strcmp(propStrShort,"order") == 0 || 
					(strcmp(propStrShort,"title") == 0 && isTitle == 1) ||
					(strcmp(propStrShort,"url") == 0 && isUrl == 1) ||
					(strcmp(propStrShort,"description") == 0 && isDescription == 1) ||
					(strcmp(propStrShort,"site_name") == 0 && isSite == 1) ||
					(strcmp(propStrShort,"image") == 0 && isImage == 1)  ||
					(strcmp(propStrShort,"email") == 0 && isEmail == 1)  ||
					(strcmp(propStrShort,"country") == 0 && isCountry == 1)  ||
					(strcmp(propStrShort,"locality") == 0 && isLocality == 1)  ||
					strcmp(propStrShort,"fbmladmins") == 0 ||
					strcmp(propStrShort,"latitude") == 0 ||
					strcmp(propStrShort,"fbmlapp_id") == 0 ||
					strcmp(propStrShort,"locale") == 0 ||
					strcmp(propStrShort,"longitude") == 0 ||
					strcmp(propStrShort,"phone_number") == 0 ||
					strcmp(propStrShort,"postal") == 0 ||
					strcmp(propStrShort,"street") == 0 ||
					strcmp(propStrShort,"region") == 0 ||
					strcmp(propStrShort,"fax_number") == 0 ||
					strcmp(propStrShort,"app_id") == 0 
					)
				fprintf(fouttb,",\n%s_%d string",propStrShort,j);
			else
				fprintf(fouttb,",\n%s string",propStrShort);

			if (strcmp(propStrShort,"title") == 0) isTitle = 1; //WEBCRAWL specific problem, duplicate title
			if (strcmp(propStrShort,"url") == 0) isUrl = 1; //WEBCRAWL specific problem, duplicate url
			if (strcmp(propStrShort,"type") == 0) isType = 1; //WEBCRAWL specific problem, duplicate type
			if (strcmp(propStrShort,"description") == 0) isDescription = 1; //WEBCRAWL specific problem, duplicate type
			if (strcmp(propStrShort,"image") == 0) isImage = 1; //WEBCRAWL specific problem, duplicate type
			if (strcmp(propStrShort,"site_name") == 0) isSite = 1; //WEBCRAWL specific problem, duplicate site_name
			if (strcmp(propStrShort,"email") == 0) isEmail = 1; //WEBCRAWL specific problem, duplicate email		
			if (strcmp(propStrShort,"country") == 0) isCountry = 1; //WEBCRAWL specific problem, duplicate site_name
			if (strcmp(propStrShort,"locality") == 0) isLocality = 1; //WEBCRAWL specific problem, duplicate email		

			GDKfree(propStrShort);
#else
			fprintf(fout,";%s", propStr);
#endif
			GDKfree(propStr);
		}
		fprintf(fout, "\n");
		fprintf(fouttb, "\n); \n \n");
		
		//List of support
		for (j = 0; j < sample.numProp; j++){
			if (freqCS.lstPropSupport[j] * 100 < freqCS.support * SAMPLE_FILTER_THRESHOLD) continue; 
			fprintf(fout,";%d", freqCS.lstPropSupport[j]);
		}
		fprintf(fout, "\n");
		
		fprintf(foutis, "echo \"");
		//All the instances 
		for (k = 0; k < sample.numInstances; k++){
#if USE_SHORT_NAMES
			str subjStrShort = NULL;
#endif
			takeOid(sample.lstSubjOid[k], &subjStr); 
#if USE_SHORT_NAMES
			getPropNameShort(&subjStrShort, subjStr);
			fprintf(fout,"<%s>", subjStrShort);
			fprintf(foutis,"<%s>", subjStrShort);
			GDKfree(subjStrShort);
#else
			fprintf(fout,"%s", subjStr);
#endif
			GDKfree(subjStr); 
			
			for (j = 0; j < sample.numProp; j++){
				if (freqCS.lstPropSupport[j] * 100 < freqCS.support * SAMPLE_FILTER_THRESHOLD) continue; 
				objOid = sample.lstObj[j][k];
				if (objOid == BUN_NONE){
					fprintf(fout,";NULL");
					fprintf(foutis,"|NULL");
				}
				else{
					objStr = NULL;
					getObjStr(mbat, mapi, objOid, &objStr, &objType);
					if (objType == URI || objType == BLANKNODE){
#if USE_SHORT_NAMES
						str objStrShort = NULL;
						getPropNameShort(&objStrShort, objStr);
						fprintf(fout,";<%s>", objStrShort);
						fprintf(foutis,"|<%s>", objStrShort);
						GDKfree(objStrShort);
#else
						fprintf(fout,";%s", objStr);
#endif
						GDKfree(objStr);
					} else {
						str betweenQuotes;
						getStringBetweenQuotes(&betweenQuotes, objStr);
						fprintf(fout,";%s", betweenQuotes);
						pch = strstr (betweenQuotes,"\\");
						if (pch != NULL) *pch = '\0';	//Remove \ characters from table name
						fprintf(foutis,"|%s", betweenQuotes);
						GDKfree(betweenQuotes);
					}
				}
			}
			fprintf(fout, "\n");
			fprintf(foutis, "\n");

		}

		fprintf(fout, "\n");
		fprintf(foutis, "\" > tmp.txt \n \n");

		if (sample.name != BUN_NONE){
			str canStrShort = NULL;
			takeOid(sample.name, &canStr);
			getPropNameShort(&canStrShort, canStr);

			if (strstr (canStrShort,".") != NULL || 
				strcmp(canStrShort,"") == 0 || 
				strstr(canStrShort,"-") != NULL	){	// WEBCRAWL specific problem with Table name a.jpg, b.png....
				fprintf(foutis, "echo \"COPY %d RECORDS INTO tbSample%d FROM 'ABSOLUTEPATH/tmp.txt'     USING DELIMITERS '|', '\\n'; \" > tmpload.sql \n", sample.numInstances, i);
			}
			else if (strcmp(canStrShort,"page") == 0){
				fprintf(foutis, "echo \"COPY %d RECORDS INTO %s%d FROM 'ABSOLUTEPATH/tmp.txt'     USING DELIMITERS '|', '\\n'; \" > tmpload.sql \n", sample.numInstances, canStrShort, i);
			}
			else{

				pch = strstr (canStrShort,"(");
				if (pch != NULL) *pch = '\0';	//Remove (...) characters from table name
				fprintf(foutis, "echo \"COPY %d RECORDS INTO %s FROM 'ABSOLUTEPATH/tmp.txt'     USING DELIMITERS '|', '\\n'; \" > tmpload.sql \n", sample.numInstances, canStrShort);
			}
			fprintf(foutis, "mclient < tmpload.sql \n");
			GDKfree(canStrShort);
			GDKfree(canStr);
		}
		else{
			fprintf(foutis, "echo \"COPY %d RECORDS INTO tbSample%d FROM 'ABSOLUTEPATH/tmp.txt'     USING DELIMITERS '|', '\\n'; \" > tmpload.sql \n", sample.numInstances, i);
			fprintf(foutis, "mclient < tmpload.sql \n");
		}

			
	}

	fclose(fout);
	fclose(fouttb); 
	fclose(foutis); 
	
	TKNZRclose(&ret);
	return MAL_SUCCEED;
}

static 
str RDFExtractSampleData(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi,  
		oid *subjCSMap, int* csTblIdxMapping, int maxNumPwithDup, CSSample *csSample, BAT *tblCandBat, int numSampleTbl){

	BUN	 	p, q; 
	oid 		*sbt = 0, *obt, *pbt;
	oid 		curS; 		/* current Subject oid */
	//oid 		CSoid = 0; 	/* Characteristic set oid */
	int 		numP;	/* Number of properties for current S */
	oid* 		buffO; 
	oid*		buffP;
	oid		curP; 
	int		tblIdx; 
	BUN		sampleIdx = BUN_NONE; 
	int		totalInstance = 0; 
	int		maxNumInstance = NUM_SAMPLE_INSTANCE * numSampleTbl;

	(void) csSample; 

	buffO = (oid *) malloc(sizeof(oid) * (maxNumPwithDup + 1)); 
	buffP = (oid *) malloc(sizeof(oid) * (maxNumPwithDup + 1));

	numP = 0;
	curS = 0; 
	curP = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				tblIdx = csTblIdxMapping[subjCSMap[curS]];
				if (tblIdx != -1){
				
					sampleIdx = BUNfnd(BATmirror(tblCandBat),(ptr) &tblIdx);
					if (sampleIdx != BUN_NONE) {
						assert(!(numP > csSample[sampleIdx].numProp));
						if (csSample[sampleIdx].numInstances < NUM_SAMPLE_INSTANCE){	
							addSampleInstance(*sbt, buffO, buffP, numP, sampleIdx, csSample);
							totalInstance++;
						}
					}
				}
			}
			curS = *sbt; 
			numP = 0;
			curP = 0; 
		}
				
		if (totalInstance == maxNumInstance) break;

		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
	
		pbt = (oid *) BUNtloc(pi, p);

		if (curP != *pbt){
			buffO[numP] = *obt; 
			buffP[numP] = *pbt;
			numP++; 
			curP = *pbt; 
		}


	}
	
	/* Check for the last CS */

	free (buffO); 
	free (buffP); 

	*ret = 1; 

	return MAL_SUCCEED; 
}

/* Create a new data structure to store relationships including merged CS */
static
CSrel* generateCsRelBetweenMergeFreqSet(CSrel *csrelFreqSet, CSset *freqCSset){
	int 	i,j;
	int	numFreqCS = freqCSset->numOrigFreqCS; 
	int 	from, to;
	CSrel 	rel;
	CSrel*  csRelMergeFreqSet;
	
	csRelMergeFreqSet = initCSrelset(freqCSset->numCSadded);

	for (i = 0; i < numFreqCS; ++i) {
		if (csrelFreqSet[i].numRef == 0) continue; // ignore CS without relations
		rel = csrelFreqSet[i];
		// update the 'from' value
		from = i;
		while (freqCSset->items[from].parentFreqIdx != -1) {
			from = freqCSset->items[from].parentFreqIdx;
		}
		assert(freqCSset->items[from].parentFreqIdx == -1);

		for (j = 0; j < rel.numRef; ++j) {
			// update the 'to' value
			to = rel.lstRefFreqIdx[j];
			while (freqCSset->items[to].parentFreqIdx != -1) {
				to = freqCSset->items[to].parentFreqIdx;
			}
			assert(freqCSset->items[to].parentFreqIdx == -1);
			// add relation to new data structure
			addReltoCSRelWithFreq(from, to, rel.lstPropId[j], rel.lstCnt[j], rel.lstBlankCnt[j], &csRelMergeFreqSet[from]);
		}
	}
	return csRelMergeFreqSet;
}


/* Refine the relationship between mergeCS in order to create FK relationship between tables */

static
CSrel* getFKBetweenTableSet(CSrel *csrelFreqSet, CSset *freqCSset, CSPropTypes* csPropTypes, int* mfreqIdxTblIdxMapping, int numTables){
	int 	i,j;
	int 	from, to;
	int	toFreqId; 
	CSrel 	rel;
	CSrel*  refinedCsRel;
	int	propIdx; 	//Index of prop in list of props for each FreqCS
	int 	numRel = freqCSset->numCSadded; 
	int	numOneToMany = 0;
	int	numManyToMany = 0;

	refinedCsRel = initCSrelset(numTables);

	for (i = 0; i < numRel; ++i) {
		if (csrelFreqSet[i].numRef == 0) continue; // ignore CS without relations
		assert(freqCSset->items[i].parentFreqIdx == -1);
		if (!isCSTable(freqCSset->items[i])) continue; 
		rel = csrelFreqSet[i];
		from = mfreqIdxTblIdxMapping[i];
		assert(from < numTables);
		assert(from != -1); 
		// update the 'from' value
		for (j = 0; j < rel.numRef; ++j) {
			toFreqId = rel.lstRefFreqIdx[j];
			assert(freqCSset->items[toFreqId].parentFreqIdx == -1);
			if (!isCSTable(freqCSset->items[toFreqId])) continue; 
			// add relation to new data structure

			//Compare with prop coverage from csproptype	
			if (rel.lstCnt[j]  < freqCSset->items[toFreqId].support * MIN_FK_FREQUENCY)	continue; 

			to = mfreqIdxTblIdxMapping[toFreqId]; 
			assert(to != -1); 
			
			//printf("Pass all basic conditions \n"); 

			//Compare with the property coverage from csPropTypes
			propIdx = 0; 
			while (csPropTypes[from].lstPropTypes[propIdx].prop != rel.lstPropId[j]){
				propIdx++;
			}
			assert(propIdx < freqCSset->items[i].numProp);
			

			//Filtering: For big size table, if large number of prop's instances need to refer to a certain table
			// else, all instances of that prop must refer to the certain table
			if (freqCSset->items[i].coverage > MINIMUM_TABLE_SIZE){
				if (csPropTypes[from].lstPropTypes[propIdx].propCover * MIN_FK_PROPCOVERAGE > rel.lstCnt[j]) continue; 
				else
					csPropTypes[from].lstPropTypes[propIdx].isDirtyFKProp = 1;
			}
			else{
				if (csPropTypes[from].lstPropTypes[propIdx].propCover != rel.lstCnt[j]) continue; 
				else
					csPropTypes[from].lstPropTypes[propIdx].isDirtyFKProp = 0;
			}
			
			assert(to < numTables);
			if (rel.lstCnt[j] > freqCSset->items[toFreqId].support){
				//printf("ONE to MANY relatioship \n");	
				numOneToMany++;
			}
			if (csPropTypes[from].lstPropTypes[propIdx].isMVProp){
				//printf("MANY to MANY relatioship \n"); 
				numManyToMany++;
			}

			addReltoCSRelWithFreq(from, to, rel.lstPropId[j], rel.lstCnt[j], rel.lstBlankCnt[j], &refinedCsRel[from]);

			//Add rel info to csPropTypes
			csPropTypes[from].lstPropTypes[propIdx].isFKProp = 1; 
			csPropTypes[from].lstPropTypes[propIdx].refTblId = to;
			csPropTypes[from].lstPropTypes[propIdx].refTblSupport = freqCSset->items[toFreqId].support;
			csPropTypes[from].lstPropTypes[propIdx].numReferring = rel.lstCnt[j];


		}
	}
	printf("FK relationship: Possible number of One-to-Many FK: %d \n", numOneToMany);
	printf("FK relationship: Possible number of Many-to-Many FK: %d \n", numManyToMany);

	return refinedCsRel;
}

static
CSrel* generateCsRelToMergeFreqSet(CSrel *csrelFreqSet, CSset *freqCSset){
	int 	i,j;
	int	numFreqCS = freqCSset->numOrigFreqCS; 
	int 	from, to;
	CSrel 	rel;
	CSrel*  csRelMergeFreqSet;
	
	csRelMergeFreqSet = initCSrelset(freqCSset->numCSadded);

	for (i = 0; i < numFreqCS; ++i) {
		if (csrelFreqSet[i].numRef == 0) continue; // ignore CS without relations
		rel = csrelFreqSet[i];
		// update the 'from' value
		from = i;
		/*
		while (freqCSset->items[from].parentFreqIdx != -1) {
			from = freqCSset->items[from].parentFreqIdx;
		}
		assert(freqCSset->items[from].parentFreqIdx == -1);
		*/

		for (j = 0; j < rel.numRef; ++j) {
			// update the 'to' value
			to = rel.lstRefFreqIdx[j];
			while (freqCSset->items[to].parentFreqIdx != -1) {
				to = freqCSset->items[to].parentFreqIdx;
			}
			assert(freqCSset->items[to].parentFreqIdx == -1);
			// add relation to new data structure
			addReltoCSRelWithFreq(from, to, rel.lstPropId[j], rel.lstCnt[j], rel.lstBlankCnt[j], &csRelMergeFreqSet[from]);
		}
	}
	return csRelMergeFreqSet;
}

static
str printCSRel(CSset *freqCSset, CSrel *csRelMergeFreqSet, int freqThreshold){
	FILE 	*fout2,*fout2filter;
	char 	filename2[100];
	char 	tmpStr[20];
	str 	propStr;
	int		i,j, k;
	int		freq;
	int	*mfreqIdxTblIdxMapping;
	char*   schema = "rdf";
	int	ret; 

	strcpy(filename2, "csRelationshipBetweenMergeFreqCS");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename2, tmpStr);
	strcat(filename2, ".txt");

	fout2 = fopen(filename2,"wt");
	strcat(filename2, ".filter");
	fout2filter = fopen(filename2,"wt");

	k = 0; 
        mfreqIdxTblIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded);
	initIntArray(mfreqIdxTblIdxMapping , freqCSset->numCSadded, -1);

	for (i = 0; i < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){	// Only use the maximum or merge CS 
			mfreqIdxTblIdxMapping[i] = k; 
			k++; 
		}
	}
	

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "printCSrel",
				"could not open the tokenizer\n");
	}

	fprintf(fout2filter, "TblIdx: (Frequency) --> TblIdx (Property) (#of References) (#of blanknodes),...");	
	for (i = 0; i < freqCSset->numCSadded; i++){
		if (csRelMergeFreqSet[i].numRef != 0){	//Only print CS with FK
			fprintf(fout2, "Relationship "BUNFMT": ", freqCSset->items[csRelMergeFreqSet[i].origFreqIdx].csId);
			fprintf(fout2filter, "Relationship "BUNFMT": ", freqCSset->items[csRelMergeFreqSet[i].origFreqIdx].csId);
			freq = freqCSset->items[csRelMergeFreqSet[i].origFreqIdx].support;
			fprintf(fout2, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", freqCSset->items[csRelMergeFreqSet[i].origFreqIdx].csId, freq, 1);
			/*fprintf(fout2filter, "CS " BUNFMT " (Freq: %d, isFreq: %d) --> ", freqCSset->items[csRelMergeFreqSet[i].origFreqIdx].csId, freq, 1);*/
			fprintf(fout2filter, "Tbl %d (Freq: %d) --> ", mfreqIdxTblIdxMapping[csRelMergeFreqSet[i].origFreqIdx], freq);
			for (j = 0; j < csRelMergeFreqSet[i].numRef; j++){
				#if SHOWPROPERTYNAME
				takeOid(csRelMergeFreqSet[i].lstPropId[j], &propStr);
				fprintf(fout2, BUNFMT "(P:" BUNFMT " - %s) (%d)(Blank:%d) ", freqCSset->items[csRelMergeFreqSet[i].lstRefFreqIdx[j]].csId,csRelMergeFreqSet[i].lstPropId[j], propStr, csRelMergeFreqSet[i].lstCnt[j], csRelMergeFreqSet[i].lstBlankCnt[j]);
				GDKfree(propStr);
				#else
				fprintf(fout2, BUNFMT "(P:" BUNFMT ") (%d)(Blank:%d) ", freqCSset->items[csRelMergeFreqSet[i].lstRefFreqIdx[j]].csId,csRelMergeFreqSet[i].lstPropId[j], csRelMergeFreqSet[i].lstCnt[j], csRelMergeFreqSet[i].lstBlankCnt[j]);
				#endif

				if (freq < csRelMergeFreqSet[i].lstCnt[j]*100){
					/*fprintf(fout2filter, BUNFMT "(P:" BUNFMT ") (%d)(Blank:%d) ", freqCSset->items[csRelMergeFreqSet[i].lstRefFreqIdx[j]].csId,csRelMergeFreqSet[i].lstPropId[j], csRelMergeFreqSet[i].lstCnt[j], csRelMergeFreqSet[i].lstBlankCnt[j]);*/
					fprintf(fout2filter, "Tbl %d (P:" BUNFMT ") (%d)(Blank:%d) ", mfreqIdxTblIdxMapping[csRelMergeFreqSet[i].lstRefFreqIdx[j]],csRelMergeFreqSet[i].lstPropId[j], csRelMergeFreqSet[i].lstCnt[j], csRelMergeFreqSet[i].lstBlankCnt[j]);
					if (freq == csRelMergeFreqSet[i].lstCnt[j]){
						fprintf(fout2filter, " (FKRel) ");
					}
				}
			}
			fprintf(fout2, "\n");
			fprintf(fout2filter, "\n");
		}
	}

	TKNZRclose(&ret);
	fclose(fout2);
	fclose(fout2filter);
	free(mfreqIdxTblIdxMapping);

	return MAL_SUCCEED; 
}


static
str printFKs(CSrel *csRelFinalFKs, int freqThreshold, int numTables,  CSPropTypes* csPropTypes){
	FILE 	*fout;
	char 	filename[100];
	char 	tmpStr[20];
	str 	propStr;
	int	i,j;
	char*   schema = "rdf";
	int	ret; 
	int	propIdx; 	//Index of prop in each table

	strcpy(filename, "FKRelationship");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt");

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "printFKs",
				"could not open the tokenizer\n");
	}

	for (i = 0; i < numTables; i++){
		if (csRelFinalFKs[i].numRef != 0){	//Only print CS with FK
			fprintf(fout, "FK "BUNFMT ": ", csRelFinalFKs[i].origFreqIdx);
			for (j = 0; j < csRelFinalFKs[i].numRef; j++){

				propIdx = 0; 
				while (csPropTypes[i].lstPropTypes[propIdx].prop !=  csRelFinalFKs[i].lstPropId[j]){
					propIdx++;
				}
				assert(propIdx < csPropTypes[i].numProp);

				#if SHOWPROPERTYNAME
				takeOid(csRelFinalFKs[i].lstPropId[j], &propStr);
				fprintf(fout, BUNFMT "(P:" BUNFMT " - %s) (%d | %d)(Blank:%d) ", csRelFinalFKs[i].lstRefFreqIdx[j], csRelFinalFKs[i].lstPropId[j], propStr, csRelFinalFKs[i].lstCnt[j], csPropTypes[i].lstPropTypes[propIdx].propCover, csRelFinalFKs[i].lstBlankCnt[j]);
				GDKfree(propStr);
				#else
				fprintf(fout, BUNFMT "(P:" BUNFMT ") (%d | %d)(Blank:%d) ", csRelFinalFKs[i].lstRefFreqIdx[j],csRelFinalFKs[i].lstPropId[j], csRelFinalFKs[i].lstCnt[j], csPropTypes[i].lstPropTypes[propIdx].propCover, csRelFinalFKs[i].lstBlankCnt[j]);
				#endif
				//Indicate FK with dirty data
				if ( csPropTypes[i].lstPropTypes[propIdx].propCover >  csRelFinalFKs[i].lstCnt[j]) fprintf(fout, "[DIRTY]"); 

			}
			fprintf(fout, "\n");
		}
	}

	TKNZRclose(&ret);
	fclose(fout);

	return MAL_SUCCEED; 
}


static 
void printFKMultiplicityFromCSPropTypes(CSPropTypes* csPropTypes, int numMergedCS, CSset *freqCSset, int freqThreshold){
	char filename[100]; 
	char tmpStr[50]; 
	FILE *fout; 
	int i, j; 
	
	int	numMtoM = 0; 	//Many To Many
	int	numOtoM = 0; 	//One to Many
	int	numOtoO = 0; 
	int	freqCSId; 

	printf("Collect the statistic for FK multiplicity ... ");
	strcpy(filename, "FKMultiplicity");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	/* Print cspropTypes */

	fprintf(fout, "FromTbl	PropId	Prop	ToTbl	isMultiProp	PropCoverage	PropSupport	RefferingTblSupport	PropSupportRatio	RefferedTblSupport	NumReffering	NumReferred	Ratio\n");
	for (i = 0; i < numMergedCS; i++){
		for(j = 0; j < csPropTypes[i].numProp; j++){
			if (csPropTypes[i].lstPropTypes[j].isFKProp){
				if (csPropTypes[i].lstPropTypes[j].numDisRefValues == 0) continue; // These columns may be put into PSO, thus no FK
				
				freqCSId = csPropTypes[i].freqCSId; 
				fprintf(fout, "%d	%d	"BUNFMT"	%d	%d	%d	%d	%d	%f	%d	%d	%d	%f\n",
						i , j, csPropTypes[i].lstPropTypes[j].prop, csPropTypes[i].lstPropTypes[j].refTblId, 
						csPropTypes[i].lstPropTypes[j].isMVProp, csPropTypes[i].lstPropTypes[j].propCover, 
						csPropTypes[i].lstPropTypes[j].propFreq, freqCSset->items[freqCSId].support,
						(float)csPropTypes[i].lstPropTypes[j].propFreq / freqCSset->items[freqCSId].support,
						csPropTypes[i].lstPropTypes[j].refTblSupport, csPropTypes[i].lstPropTypes[j].numReferring,
						csPropTypes[i].lstPropTypes[j].numDisRefValues, 
						(float)csPropTypes[i].lstPropTypes[j].numReferring / csPropTypes[i].lstPropTypes[j].numDisRefValues
						);
				if (csPropTypes[i].lstPropTypes[j].numReferring == csPropTypes[i].lstPropTypes[j].numDisRefValues) numOtoO++;
				if  (csPropTypes[i].lstPropTypes[j].isMVProp)	numMtoM++;
				if (csPropTypes[i].lstPropTypes[j].numReferring > csPropTypes[i].lstPropTypes[j].numDisRefValues) numOtoM++;	
			}
		}

	}
	
	printf("Done!\n"); 
	
	printf("There are %d One to Many FKs\n", numOtoM);
	printf("There are %d Many to Many FKs\n", numMtoM);
	printf("There are %d One to One FKs\n", numOtoO);

	fclose(fout); 

}

// for storing ontology data
oid	**ontattributes = NULL;
int	ontattributesCount = 0;
oid	**ontmetadata = NULL;
int	ontmetadataCount = 0;

/* Extract CS from SPO triples table */
str
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold, void *_freqCSset, oid **subjCSMap, oid *maxCSoid, int *maxNumPwithDup, CSlabel** labels, CSrel **csRelMergeFreqSet){

	BAT 		*sbat = NULL, *pbat = NULL, *obat = NULL, *mbat = NULL; 
	BATiter 	si, pi, oi; 	/*iterator for BAT of s,p,o columns in spo table */

	CSBats		*csBats; 

	BUN		*maxSoid; 	
	int		maxNumProp = 0;
	CSrel   	*csrelSet;

	#if	NEEDSUBCS
	SubCSSet 	*csSubCSSet; 
	oid		*subjSubCSMap;  /* Store the corresponding CS sub Id for each subject */
	char		*subjdefaultMap = NULL; /* Specify whether this subject contains default value or not. This array may be large */
	#endif

	int 		*refCount; 	/* Count the number of references to each CS */

	int*		csIdFreqIdxMap; /* Map a CSId to a freqIdx. Should be removed in the future .... */
	oid		mergecsId = 0; 

	oid		*mergeCSFreqCSMap; 
	CSset		*freqCSset; 
	clock_t 	curT;
	clock_t		tmpLastT; 
	OntoUsageNode	*ontoUsageTree = NULL;
	int		curNumMergeCS = 0; 
	int 		tmpNumRel = 0;
	CSrel		*tmpCSrelToMergeCS = NULL; 
	float		*curIRScores = NULL; 

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
	initArray(*subjCSMap, (*maxSoid) + 1, BUN_NONE);


	
	
	tmpLastT = clock();

	*maxNumPwithDup	 = 0;
	//Phase 1: Assign an ID for each CS
	#if STOREFULLCS
	RDFassignCSId(ret, sbat, si, pi, oi, freqCSset, freqThreshold, csBats, *subjCSMap, maxCSoid, &maxNumProp, maxNumPwithDup);
	#else
	RDFassignCSId(ret, sbat, si, pi, freqCSset, freqThreshold, csBats, *subjCSMap, maxCSoid, &maxNumProp, maxNumPwithDup);
	#endif
		
	curT = clock(); 
	printf (" ----- Exploring all CSs took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
		
	printf("Max CS oid: " BUNFMT "\n", *maxCSoid);
	printf("Max Number of P per CS (with/without duplication): %d / %d \n", maxNumProp, *maxNumPwithDup);
	printf("Number of freqCSs found by frequency: %d \n", freqCSset->numCSadded);

	tmpLastT = curT; 		
	
	/* Phase 2: Get the references count for each CS. Add frequent one to freqCSset */

	csIdFreqIdxMap = (int *) malloc (sizeof(int) * (*maxCSoid + 1));
	initcsIdFreqIdxMap(csIdFreqIdxMap, *maxCSoid + 1, -1, freqCSset);

	refCount = (int *) malloc(sizeof(int) * (*maxCSoid + 1));
	initIntArray(refCount, (*maxCSoid + 1), 0); 
	RDFgetRefCounts(ret, sbat, si, pi,oi, *subjCSMap, maxNumProp, *maxSoid, refCount);

	addHighRefCSsToFreqCS(csBats->pOffsetBat, csBats->freqBat, csBats->coverageBat, csBats->fullPBat, refCount, freqCSset, csIdFreqIdxMap, *maxCSoid + 1, HIGH_REFER_THRESHOLD * (*freqThreshold)); 

	free(refCount);
	curT = clock();
	printf (" ----- Counting references and adding highly referred CS's took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	printf("Number of freqCSs after considering # references: %d \n", freqCSset->numCSadded);
	tmpLastT = curT;
	
	//Phase 3: Check the relationship	

	csrelSet = initCSrelset(freqCSset->numCSadded);
	
	#if NEEDSUBCS
	subjSubCSMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	subjdefaultMap = (char *) malloc (sizeof(char) * ((*maxSoid) + 1));

	initCharArray(subjdefaultMap,(*maxSoid) + 1, 0); 

	csSubCSSet = initCS_SubCSSets(*maxCSoid +1); 

	RDFrelationships(ret, sbat, si, pi, oi, *subjCSMap, subjSubCSMap, csSubCSSet, csrelSet, *maxSoid, *maxNumPwithDup, csIdFreqIdxMap);
	#else
	RDFrelationships(ret, sbat, si, pi, oi, *subjCSMap, csrelSet, *maxSoid, *maxNumPwithDup, csIdFreqIdxMap);
	#endif

	curT = clock(); 
	printf (" ----- Exploring subCSs and FKs took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		


	printCSrelSet(csrelSet, freqCSset, freqCSset->numCSadded, *freqThreshold);  

	#if NEEDSUBCS
	setdefaultSubCSs(csSubCSSet,*maxCSoid + 1, sbat, subjSubCSMap, *subjCSMap, subjdefaultMap);
	printSubCSInformation(csSubCSSet, csBats->freqBat, *maxCSoid + 1, 1, *freqThreshold); 
	#endif

	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	//createTreeForCSset(freqCSset); 	// DOESN'T HELP --> REMOVE
	
	/*get the statistic */
	//getTopFreqCSs(csMap,*freqThreshold);

	// Create label per freqCS

	printf("Using ontologies with %d ontattributesCount and %d ontmetadataCount \n",ontattributesCount,ontmetadataCount);
	
	(*labels) = createLabels(freqCSset, csrelSet, freqCSset->numCSadded, sbat, si, pi, oi, *subjCSMap, csIdFreqIdxMap, ontattributes, ontattributesCount, ontmetadata, ontmetadataCount, &ontoUsageTree);
	
	curT = clock(); 
	printf("Done labeling!!! Took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT;
	
	printFreqCSSet(freqCSset, csBats->freqBat, mbat, 1, *freqThreshold, *labels); 

	/* Get the number of indirect refs in order to detect dimension table */
	refCount = (int *) malloc(sizeof(int) * (freqCSset->numCSadded));
	curIRScores = (float *) malloc(sizeof(float) * (freqCSset->numCSadded));
	
	initIntArray(refCount, freqCSset->numCSadded, 0); 

	getOrigRefCount(csrelSet, freqCSset, freqCSset->numCSadded, refCount);  
	getIRNums(csrelSet, freqCSset, freqCSset->numCSadded, refCount, curIRScores, NUM_ITERATION_FOR_IR);  
	updateFreqCStype(freqCSset, freqCSset->numCSadded, curIRScores, refCount);

	free(refCount); 
	free(curIRScores);

	curT = clock(); 
	printf("Get number of indirect referrences to detect dimension tables !!! Took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT;
	/*------------------------------------*/
	
	curNumMergeCS = countNumberMergeCS(freqCSset);
	printf("Before using rules: Number of freqCS is: %d \n",curNumMergeCS);


	/* ---------- S1, S2 ------- */
	mergecsId = *maxCSoid + 1; 

	mergeMaxFreqCSByS1(freqCSset, labels, &mergecsId, ontmetadata, ontmetadataCount); /*S1: Merge all freqCS's sharing top-3 candidates */
	
	curNumMergeCS = countNumberMergeCS(freqCSset);

	curT = clock(); 
	printf("Merging with S1 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	printf("Number of added CS after S1: %d \n", freqCSset->numCSadded);
	tmpLastT = curT;
	
	/* ---------- S4 ------- */
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	/*S4: Merge two CS's having the subset-superset relationship */
	mergeCSbyS4(freqCSset, labels, mergeCSFreqCSMap,curNumMergeCS, ontmetadata, ontmetadataCount); 

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf("Merging with S4 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	printf("Number of added CS after S4: %d \n", freqCSset->numCSadded);
	tmpLastT = curT; 		
	
	/* ---------- S6 ------- */
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);
	
	tmpCSrelToMergeCS = generateCsRelToMergeFreqSet(csrelSet, freqCSset);
	tmpNumRel = freqCSset->numCSadded; 

	/* S6: Merged CS referred from the same CS via the same property */
	if (1) mergeMaxFreqCSByS6(tmpCSrelToMergeCS, freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS,  &mergecsId, ontmetadata, ontmetadataCount);
	//printf("DISABLE S6 (For Testing) \n"); 

	freeCSrelSet(tmpCSrelToMergeCS,tmpNumRel);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf("Merging with S6 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	tmpLastT = curT; 		
	
	/* S3, S5 */
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	mergeCSByS3S5(freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS, &mergecsId, ontoUsageTree, ontmetadata, ontmetadataCount);
	free(mergeCSFreqCSMap);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf ("Merging with S3, S5 took %f. (Number of mergeCS: %d) \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS);	
	tmpLastT = curT; 		

	updateParentIdxAll(freqCSset); 

	
	//Finally, re-create mergeFreqSet
	
	
	*csRelMergeFreqSet = generateCsRelBetweenMergeFreqSet(csrelSet, freqCSset);
	printCSRel(freqCSset, *csRelMergeFreqSet, *freqThreshold);
	
	curT = clock(); 
	printf ("Get the final relationships between mergeCS took %f. \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC);	
	tmpLastT = curT; 		

	printmergeCSSet(freqCSset, *freqThreshold);
	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csBats->pOffsetBat, csBats->freqBat, csBats->coverageBat, csBats->fullPBat, 1, *freqThreshold);



	BBPunfix(sbat->batCacheid); 
	BBPunfix(pbat->batCacheid); 
	BBPunfix(obat->batCacheid);
	BBPunfix(mbat->batCacheid);

	freeOntoUsageTree(ontoUsageTree);
	
	#if NEEDSUBCS
	free (subjSubCSMap);
	freeCS_SubCSMapSet(csSubCSSet, *maxCSoid + 1); 
	#endif

	free(csIdFreqIdxMap); 
	freeCSrelSet(csrelSet, freqCSset->numOrigFreqCS); 

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
BAT* getOriginalUriOBat(BAT *obat){
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
		objType = getObjType(*obt); 

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
 * In case of using type-specific cs table, we use one more bit at the 
 * position sizeof(BUN)*8 - NBITS_FOR_CSID - 1 for specifying whether 
 * a subject has the default data types for its properties or not. 
 * Thus, the way to calculate the table idx and base idx is changed
 * */
static 
void getTblIdxFromS(oid Soid, int *tbidx, oid *baseSoid){
	
	*tbidx = (int) ((Soid >> (sizeof(BUN)*8 - NBITS_FOR_CSID))  &  ((1 << (NBITS_FOR_CSID-1)) - 1)) ;
	
	*baseSoid = Soid - ((oid) (*tbidx) << (sizeof(BUN)*8 - NBITS_FOR_CSID));

	*tbidx = *tbidx - 1; 

	//return freqCSid; 
}

/* This function should be the same as getTblIdxFromS */
static 
void getTblIdxFromO(oid Ooid, int *tbidx){
	
	*tbidx = (int) ((Ooid >> (sizeof(BUN)*8 - NBITS_FOR_CSID))  &  ((1 << (NBITS_FOR_CSID-1)) - 1)) ;
	
	*tbidx = *tbidx - 1; 

	//return freqCSid; 
}

static
str getOrigPbt(oid *pbt, oid *origPbt, BAT *lmap, BAT *rmap){
	BUN ppos; 
	oid *tmp; 
	ppos = BUNfnd(BATmirror(rmap),pbt);
	if (ppos == BUN_NONE){
		throw(RDF, "rdf.RDFdistTriplesToCSs", "This modified prop must be in rmap");
	}
	tmp = (oid *) Tloc(lmap, ppos);
	if (*tmp == BUN_NONE){
		throw(RDF, "rdf.RDFdistTriplesToCSs", "The original prop value must be in lmap");
	}

	*origPbt = *tmp; 		

	return MAL_SUCCEED; 
}

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

static
void initCStables(CStableStat* cstablestat, CSset* freqCSset, CSPropTypes *csPropTypes, int numTables, CSlabel *labels, int *mTblIdxFreqIdxMapping){

	int 		i,j, k; 
	int		tmpNumDefaultCol; 
	int		tmpNumExCol; 		/*For columns of non-default types*/
	char* 		mapObjBATtypes;
	int		colIdx, colExIdx, t; 
	int		mvColIdx; 

	mapObjBATtypes = (char*) malloc(sizeof(char) * (MULTIVALUES + 1)); 
	mapObjBATtypes[URI] = TYPE_oid; 
	mapObjBATtypes[DATETIME] = TYPE_str;
	mapObjBATtypes[INTEGER] = TYPE_int; 
	mapObjBATtypes[FLOAT] = TYPE_flt; 
	mapObjBATtypes[STRING] = TYPE_str; 
	mapObjBATtypes[BLANKNODE] = TYPE_oid;
	mapObjBATtypes[MULTIVALUES] = TYPE_oid;
	
	printf("Start initCStables \n"); 
	// allocate memory space for cstablestat
	cstablestat->numTables = numTables; 
	cstablestat->lstbatid = (bat**) malloc(sizeof (bat*) * numTables); 
	cstablestat->numPropPerTable = (int*) malloc(sizeof (int) * numTables); 

	cstablestat->pbat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	cstablestat->sbat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	cstablestat->obat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	BATseqbase(cstablestat->pbat, 0);
	BATseqbase(cstablestat->sbat, 0);
	BATseqbase(cstablestat->obat, 0);

	cstablestat->lastInsertedS = (oid**) malloc(sizeof(oid*) * numTables);
	cstablestat->lstcstable = (CStable*) malloc(sizeof(CStable) * numTables); 

	#if CSTYPE_TABLE == 1
	cstablestat->lastInsertedSEx = (oid**) malloc(sizeof(oid*) * numTables);
	cstablestat->lstcstableEx = (CStableEx*) malloc(sizeof(CStableEx) * numTables);
	#endif
	
	for (i = 0; i < numTables; i++){
		tmpNumDefaultCol = csPropTypes[i].numProp -  csPropTypes[i].numInfreqProp; 
		cstablestat->numPropPerTable[i] = tmpNumDefaultCol; 
		cstablestat->lstbatid[i] = (bat*) malloc (sizeof(bat) * tmpNumDefaultCol);  
		cstablestat->lastInsertedS[i] = (oid*) malloc(sizeof(oid) * tmpNumDefaultCol); 
		cstablestat->lstcstable[i].numCol = tmpNumDefaultCol;
		cstablestat->lstcstable[i].colBats = (BAT**)malloc(sizeof(BAT*) * tmpNumDefaultCol); 
		cstablestat->lstcstable[i].lstMVTables = (CSMVtableEx *) malloc(sizeof(CSMVtableEx) * tmpNumDefaultCol); // TODO: Only allocate memory for multi-valued columns
		cstablestat->lstcstable[i].lstProp = (oid*)malloc(sizeof(oid) * tmpNumDefaultCol);
		cstablestat->lstcstable[i].colTypes = (ObjectType *)malloc(sizeof(ObjectType) * tmpNumDefaultCol);
		cstablestat->lstcstable[i].tblname = labels[mTblIdxFreqIdxMapping[i]].name;
		#if CSTYPE_TABLE == 1
		tmpNumExCol = csPropTypes[i].numNonDefTypes; 
		cstablestat->lastInsertedSEx[i] = (oid*) malloc(sizeof(oid) * tmpNumExCol); 
		cstablestat->lstcstableEx[i].numCol = tmpNumExCol;
		cstablestat->lstcstableEx[i].colBats = (BAT**)malloc(sizeof(BAT*) * tmpNumExCol); 
		cstablestat->lstcstableEx[i].colTypes = (ObjectType*)malloc(sizeof(ObjectType) * tmpNumExCol); 
		cstablestat->lstcstableEx[i].mainTblColIdx  = (int*)malloc(sizeof(int) * tmpNumExCol); 
		cstablestat->lstcstableEx[i].tblname = labels[mTblIdxFreqIdxMapping[i]].name;
		#endif
		
		colIdx = -1; 
		colExIdx = 0; 
		for(j = 0; j < csPropTypes[i].numProp; j++){
			#if	REMOVE_INFREQ_PROP
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1)	continue;  //Infrequent prop
			#endif

			colIdx++;	
			cstablestat->lstcstable[i].lstProp[colIdx] = freqCSset->items[csPropTypes[i].freqCSId].lstProp[j];

			if (csPropTypes[i].lstPropTypes[j].isMVProp == 0){
				//cstablestat->lstcstable[i].colBats[colIdx] = BATnew(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], smallbatsz);
				cstablestat->lstcstable[i].colBats[colIdx] = BATnew(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], freqCSset->items[csPropTypes[i].freqCSId].support + 1);
				cstablestat->lstcstable[i].lstMVTables[colIdx].numCol = 0; 	//There is no MV Tbl for this prop
				//TODO: use exact size for each BAT
			}
			else{
				//cstablestat->lstcstable[i].colBats[colIdx] = BATnew(TYPE_void, TYPE_oid, smallbatsz);
				cstablestat->lstcstable[i].colBats[colIdx] = BATnew(TYPE_void, TYPE_oid, freqCSset->items[csPropTypes[i].freqCSId].support + 1);
				BATseqbase(cstablestat->lstcstable[i].colBats[colIdx], 0);	
				cstablestat->lstcstable[i].lstMVTables[colIdx].numCol = csPropTypes[i].lstPropTypes[j].numMvTypes;
				if (cstablestat->lstcstable[i].lstMVTables[colIdx].numCol != 0){
					cstablestat->lstcstable[i].lstMVTables[colIdx].colTypes = (ObjectType *)malloc(sizeof(ObjectType)* cstablestat->lstcstable[i].lstMVTables[colIdx].numCol);
					cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats = (BAT **)malloc(sizeof(BAT*) * cstablestat->lstcstable[i].lstMVTables[colIdx].numCol);
			
					mvColIdx = 0;	//Go through all types
					cstablestat->lstcstable[i].lstMVTables[colIdx].colTypes[0] = csPropTypes[i].lstPropTypes[j].defaultType; //Default type for this MV col
					//Init the first col (default type) in MV Table
					//cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[0] = BATnew(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], smallbatsz);
					cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[0] = BATnew(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], csPropTypes[i].lstPropTypes[j].propCover + 1);
					for (k = 0; k < MULTIVALUES; k++){
						if (k != csPropTypes[i].lstPropTypes[j].defaultType && csPropTypes[i].lstPropTypes[j].TableTypes[k] == MVTBL){
							mvColIdx++;
							cstablestat->lstcstable[i].lstMVTables[colIdx].colTypes[mvColIdx] = k;
							//cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[mvColIdx] = BATnew(TYPE_void, mapObjBATtypes[k], smallbatsz);
							cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[mvColIdx] = BATnew(TYPE_void, mapObjBATtypes[k],csPropTypes[i].lstPropTypes[j].propCover + 1);
						}	
					}

					//Add a bat for storing FK to the main table
					//cstablestat->lstcstable[i].lstMVTables[colIdx].keyBat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
					cstablestat->lstcstable[i].lstMVTables[colIdx].keyBat = BATnew(TYPE_void, TYPE_oid, csPropTypes[i].lstPropTypes[j].propCover + 1);
				}

				//BATseqbase(cstablestat->lstcstable[i].mvExBats[colIdx], 0);
			}
			

			//For ex-type columns
			#if CSTYPE_TABLE == 1
			for (t = 0; t < csPropTypes[i].lstPropTypes[j].numType; t++){
				if ( csPropTypes[i].lstPropTypes[j].TableTypes[t] == TYPETBL){
					//cstablestat->lstcstableEx[i].colBats[colExIdx] = BATnew(TYPE_void, mapObjBATtypes[t], smallbatsz);
					cstablestat->lstcstableEx[i].colBats[colExIdx] = BATnew(TYPE_void, mapObjBATtypes[t], freqCSset->items[csPropTypes[i].freqCSId].support + 1);
					//Set mainTblColIdx for ex-table
					cstablestat->lstcstableEx[i].colTypes[colExIdx] = t; 
					cstablestat->lstcstableEx[i].mainTblColIdx[colExIdx] = colIdx; 
					colExIdx++;

				}
			}

			#endif
		}
		
		assert(colExIdx == csPropTypes[i].numNonDefTypes);


	}

	free(mapObjBATtypes);
	printf("Finish initCStables \n"); 
}



static
void initCSTableIdxMapping(CSset* freqCSset, int* csTblIdxMapping, int* mfreqIdxTblIdxMapping, int* mTblIdxFreqIdxMapping, int *numTables){

int 		i, k; 
CS 		cs;
	int		tmpParentidx; 

	k = 0; 
	for (i = 0; i < freqCSset->numCSadded; i++){
		if (isCSTable(freqCSset->items[i])){	// Only use the not-removed maximum or merge CS  
			mfreqIdxTblIdxMapping[i] = k; 
			mTblIdxFreqIdxMapping[k] = i; 
			k++; 
		}
	}
	
	*numTables = k; 

	// Mapping the csid directly to the index of the table ==> csTblIndxMapping
	
	for (i = 0; i < freqCSset->numOrigFreqCS; i++){
		cs = (CS)freqCSset->items[i];
		tmpParentidx = cs.parentFreqIdx;
		
		if (tmpParentidx == -1){	// maximumCS 
			csTblIdxMapping[cs.csId] = mfreqIdxTblIdxMapping[i];
		}
		else{	// A normal CS or a maxCS that have a mergeCS as its parent
			if (freqCSset->items[tmpParentidx].parentFreqIdx == -1){
				csTblIdxMapping[cs.csId] = mfreqIdxTblIdxMapping[tmpParentidx]; 
			}	
			else{
				csTblIdxMapping[cs.csId] = mfreqIdxTblIdxMapping[freqCSset->items[tmpParentidx].parentFreqIdx];
			}
		}

	}


	//return cstablestat; 

}

void freeCStableStat(CStableStat* cstablestat){
	int i,j, k; 

	for (i = 0; i < cstablestat->numTables; i++){
		free(cstablestat->lstbatid[i]); 
		free(cstablestat->lastInsertedS[i]); 
		#if CSTYPE_TABLE == 1
		free(cstablestat->lastInsertedSEx[i]); 	
		#endif
		for (j = 0; j < cstablestat->numPropPerTable[i];j++){
			BBPunfix(cstablestat->lstcstable[i].colBats[j]->batCacheid); 
			/*
			if (cstablestat->lstcstable[i].mvBats[j] != NULL)
				BBPunfix(cstablestat->lstcstable[i].mvBats[j]->batCacheid); 

			if (cstablestat->lstcstable[i].mvExBats[j] != NULL)
				BBPunfix(cstablestat->lstcstable[i].mvExBats[j]->batCacheid); 
			*/

			if (cstablestat->lstcstable[i].lstMVTables[j].numCol != 0){
				for (k = 0; k < cstablestat->lstcstable[i].lstMVTables[j].numCol; k++){
					BBPunfix(cstablestat->lstcstable[i].lstMVTables[j].mvBats[k]->batCacheid);
				}
				free(cstablestat->lstcstable[i].lstMVTables[j].mvBats);
				free(cstablestat->lstcstable[i].lstMVTables[j].colTypes);
				BBPunfix(cstablestat->lstcstable[i].lstMVTables[j].keyBat->batCacheid); 
			}
		}

		#if CSTYPE_TABLE == 1
		for (j = 0; j < cstablestat->lstcstableEx[i].numCol;j++){
			BBPunfix(cstablestat->lstcstableEx[i].colBats[j]->batCacheid); 
		}
		#endif
		free(cstablestat->lstcstable[i].colBats);
		//free(cstablestat->lstcstable[i].mvBats);
		//free(cstablestat->lstcstable[i].mvExBats);
		free(cstablestat->lstcstable[i].lstMVTables);
		free(cstablestat->lstcstable[i].lstProp);
		free(cstablestat->lstcstable[i].colTypes);
		#if CSTYPE_TABLE == 1
		free(cstablestat->lstcstableEx[i].colBats);
		free(cstablestat->lstcstableEx[i].colTypes);
		free(cstablestat->lstcstableEx[i].mainTblColIdx);
		#endif
	}
	BBPunfix(cstablestat->pbat->batCacheid); 
	BBPunfix(cstablestat->sbat->batCacheid); 
	BBPunfix(cstablestat->obat->batCacheid); 
	free(cstablestat->lstbatid); 
	free(cstablestat->lastInsertedS); 
	free(cstablestat->lstcstable); 
	#if CSTYPE_TABLE == 1
	free(cstablestat->lastInsertedSEx); 
	free(cstablestat->lstcstableEx);
	#endif
	free(cstablestat->numPropPerTable);
	free(cstablestat); 
}

/*
static str
creatPBats(BAT** setofBats, Postinglist ptl, int HeadType, int TailType){
	int 	i; 
	int	numbat; 

	numbat = ptl.numAdded; 

	for (i = 0; i < numbat; i++){ 
		setofBats[ptl.lstIdx[i]] = BATnew(HeadType, TailType, smallbatsz);	
		// only create BAT for few 
	}

	return MAL_SUCCEED; 
}
*/
/*
static str
savePBats(BAT** setofBats, Postinglist ptl, CStableStat* cstablestat){
	int 	i; 
	int	numbat; 

	numbat = ptl.numAdded; 

	for (i = 0; i < numbat; i++){ 
		//store to cstablestat
		cstablestat->lstbatid[ptl.lstIdx[i]][ptl.lstInvertIdx[i]] = setofBats[ptl.lstIdx[i]]->batCacheid; 

		//removec completely
		BBPreclaim(setofBats[ptl.lstIdx[i]]) ;	
	}

	return MAL_SUCCEED; 
}
*/

static
void updateTblIdxPropIdxMap(int* tblIdxPropColumIdxMapping, int* lstCSIdx,int* lstInvertIdx,int numTblperPos){
	int i; 
	for (i = 0; i < numTblperPos; i++){
		tblIdxPropColumIdxMapping[lstCSIdx[i]] = lstInvertIdx[i];
	}

}

static 
str fillMissingvalues(BAT* curBat, int from, int to){
	int k; 
	//Insert nil values to the last column if it does not have the same
	//size as the table
	//printf("Fill from  %d to %d \n", from, to);
	if (curBat != NULL){
		for(k = from -1; k < to; k++){
			//BUNappend(curBat, ATOMnilptr(curBat->ttype), TRUE);
			if (BUNfastins(curBat, ATOMnilptr(TYPE_void), ATOMnilptr(curBat->ttype))== NULL){
				throw(RDF, "fillMissingvalues", "[Debug] Problem in inserting value");
			}
		}
	}

	return MAL_SUCCEED; 
}

static 
str fillMissingvaluesAll(CStableStat* cstablestat, CSPropTypes *csPropTypes, int lasttblIdx, int lastColIdx, int lastPropIdx, oid* lastSubjId){
	BAT     *tmpBat = NULL;
	int i; 
	int tmpColExIdx; 

	//printf("Fill for Table %d and prop %d (lastSubjId = " BUNFMT" \n", lasttblIdx, lastColIdx, lastSubjId[lasttblIdx]);

	tmpBat = cstablestat->lstcstable[lasttblIdx].colBats[lastColIdx];	
	if (fillMissingvalues(tmpBat, (int)BATcount(tmpBat), (int)lastSubjId[lasttblIdx]) != MAL_SUCCEED){
		throw(RDF, "fillMissingvaluesAll", "[Debug 1] Problem in filling missing values");
	} 

	for (i = 0; i < (MULTIVALUES + 1); i++){
		if (csPropTypes[lasttblIdx].lstPropTypes[lastPropIdx].TableTypes[i] == TYPETBL){
			tmpColExIdx = csPropTypes[lasttblIdx].lstPropTypes[lastPropIdx].colIdxes[i]; 
			tmpBat = cstablestat->lstcstableEx[lasttblIdx].colBats[tmpColExIdx];
			//printf("Fill excol %d \n", tmpColExIdx);
			if (fillMissingvalues(tmpBat, (int)BATcount(tmpBat), (int)lastSubjId[lasttblIdx]) != MAL_SUCCEED){
				throw(RDF, "fillMissingvaluesAll", "[Debug 2] Problem in filling missing values");
			}
		}
		
	}

	return MAL_SUCCEED; 
}


// colIdx: The column to be appenned
// First append nils for all missing subject from "from" to "to - 1"
static 
str fillMissingValueByNils(CStableStat* cstablestat, CSPropTypes *csPropTypes, int tblIdx, int colIdx, int propIdx, int colIdxEx, char tblType,int from, int to){
	BAT     *tmpBat = NULL;
	int i; 
	int tmpColExIdx; 
	int k; 

	//printf("Fill nils for Table %d  (type: %d)and prop %d from %d to %d \n", tblIdx, tblType, colIdx, from, to);

	tmpBat = cstablestat->lstcstable[tblIdx].colBats[colIdx];	
	//Fill all missing values from From to To
	for(k = from; k < to; k++){
		//printf("Append null to main table: Col: %d \n", colIdx);
		BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
	}

	//"Append null to not to-be-inserted col in main table: Col: %d \n", colIdx
	// MOVE OUT as we want to consider whether we can use the casted value instead of NULL value
	/*
	if (tblType != MAINTBL){
		BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
	}
	*/
	for (i = 0; i < (MULTIVALUES + 1); i++){
		if (csPropTypes[tblIdx].lstPropTypes[propIdx].TableTypes[i] == TYPETBL){
			tmpColExIdx = csPropTypes[tblIdx].lstPropTypes[propIdx].colIdxes[i]; 
			tmpBat = cstablestat->lstcstableEx[tblIdx].colBats[tmpColExIdx];
			//Fill all missing values from From to To
			for(k = from; k < to; k++){
				//printf("Append null to ex table: Col: %d \n", tmpColExIdx);
				//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
				if (BUNfastins(tmpBat, ATOMnilptr(TYPE_void), ATOMnilptr(tmpBat->ttype)) == NULL){
					throw(RDF, "fillMissingvaluesByNils", "[Debug1] Problem in inserting value");
				}
			}

			if (tblType == MAINTBL){
				//printf("Append null to not to-be-inserted col in ex table: Col: %d  (# colIdxEx = %d) \n", tmpColExIdx, colIdxEx);
				//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
				if (BUNfastins(tmpBat, ATOMnilptr(TYPE_void), ATOMnilptr(tmpBat->ttype)) == NULL){
					throw(RDF, "fillMissingvaluesByNils", "[Debug2] Problem in inserting value");
				}
			}
			else if (tmpColExIdx != colIdxEx){
				//printf("Append null to not to-be-inserted col in ex table: Col: %d (WHILE tblType = %d,  colIdxEx = %d) \n", tmpColExIdx, tblType, colIdxEx);
				//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
				if (BUNfastins(tmpBat, ATOMnilptr(TYPE_void), ATOMnilptr(tmpBat->ttype)) == NULL){
					throw(RDF, "fillMissingvaluesByNils", "[Debug3] Problem in inserting value");
				}
			}
		}
		
	}

	return MAL_SUCCEED; 
}


static
void getRealValue(ValPtr returnValue, oid objOid, ObjectType objType, BATiter mapi, BAT *mapbat){
	str 	objStr; 
	str	datetimeStr; 
	str	tmpStr; 
	BUN	bun; 	
	BUN	maxObjectURIOid =  ((oid)1 << (sizeof(BUN)*8 - NBITS_FOR_CSID - 1)) - 1; //Base on getTblIdxFromS
	float	realFloat; 
	int	realInt; 
	oid	realUri;

	//printf("objOid = " BUNFMT " \n",objOid);
	if (objType == URI || objType == BLANKNODE){
		objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
		
		if (objOid < maxObjectURIOid){
			//takeOid(objOid, &objStr); 		//TODO: Do we need to get URI string???
			//printf("From tokenizer URI object value: "BUNFMT " (str: %s) \n", objOid, objStr);
		}
		//else, this object value refers to a subject oid
		//IDEA: Modify the function for calculating new subject Id:
		//==> subjectID = TBLID ... tmpSoid .... 	      
	}
	else{
		objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */ 
		bun = BUNfirst(mapbat);
		objStr = (str) BUNtail(mapi, bun + objOid); 
		//printf("From mapbat BATcount= "BUNFMT" at position " BUNFMT ": %s \n", BATcount(mapbat),  bun + objOid,objStr);
	}
		

	switch (objType)
	{
		case STRING:
			//printf("A String object value: %s \n",objStr);
			tmpStr = GDKmalloc(sizeof(char) * strlen(objStr) + 1); 
			memcpy(tmpStr, objStr, sizeof(char) * strlen(objStr) + 1);
			VALset(returnValue, TYPE_str, tmpStr);
			break; 
		case DATETIME:
			//printf("A Datetime object value: %s \n",objStr);
			datetimeStr = getDateTimeFromRDFString(objStr);
			VALset(returnValue, TYPE_str, datetimeStr);
			break; 
		case INTEGER:
			//printf("Full object value: %s \n",objStr);
			realInt = getIntFromRDFString(objStr);
			VALset(returnValue,TYPE_int, &realInt);
			break; 
		case FLOAT:
			//printf("Full object value: %s \n",objStr);
			realFloat = getFloatFromRDFString(objStr);
			VALset(returnValue,TYPE_flt, &realFloat);
			break; 
		default: //URI or BLANK NODE		
			realUri = objOid;
			VALset(returnValue,TYPE_oid, &realUri);
	}

}


//Macro for inserting to PSO
#define insToPSO(pb, sb, ob, pbt, sbt, obt)	\
	do{					\
			if (BUNfastins(pb, ATOMnilptr(TYPE_void), pbt) == NULL){		\
				throw(RDF, "insToPSO","[Debug] Problem in inserting to pbat"); 	\
			}									\
			if (BUNfastins(sb, ATOMnilptr(TYPE_void), sbt) == NULL){		\
				throw(RDF, "insToPSO","[Debug] Problem in inserting to sbat"); 	\
			}									\
			if (BUNfastins(ob, ATOMnilptr(TYPE_void), obt) == NULL){		\
				throw(RDF, "insToPSO","[Debug] Problem in inserting to obat"); 	\
			}									\
	}while (0)


str RDFdistTriplesToCSs(int *ret, bat *sbatid, bat *pbatid, bat *obatid,  bat *mbatid, bat *lmapbatid, bat *rmapbatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId){
	
	BAT *sbat = NULL, *pbat = NULL, *obat = NULL, *mbat = NULL, *lmap = NULL, *rmap = NULL; 
	BATiter si,pi,oi, mi; 
	BUN p,q; 
	oid *pbt, *sbt, *obt;
	oid 	maxOrigPbt; 
	oid	origPbt; 
	oid lastP, lastS; 
	int	tblIdx = -1; 
	int	tmpOidTblIdx = -1; 
	oid	tmpSoid = BUN_NONE; 
	BUN	ppos; 
	int*	tmpTblIdxPropIdxMap;	//For each property, this maps the table Idx (in the posting list
					// of that property to the position of that property in the
					// list of that table's properties
	Postinglist tmpPtl; 
	int	tmpPropIdx = -1; 	// The index of property in the property list in a CS. It is not the same as the column Idx as some infrequent props can be removed
	int	tmpColIdx = -1; 
	int	tmpColExIdx = -1; 
	int	tmpMVColIdx = -1; 
	int	lasttblIdx = -1; 
	int	lastColIdx = -1; 
	int	lastPropIdx = -1; 
	char	isSetLasttblIdx = 0;
	ObjectType	objType, defaultType; 
	char	tmpTableType = 0;

	int	i,j, k; 
	BAT	*curBat = NULL;
	BAT	*tmpBat = NULL; 
	BAT     *tmpmvBat = NULL;       // Multi-values BAT
	//BAT	*tmpmvExBat = NULL; 
	int	tmplastInsertedS = -1; 
	int     numMultiValues = 0;
	oid	tmpmvValue; 
	oid	tmpmvKey = BUN_NONE; 
	char	istmpMVProp = 0; 
	char*   schema = "rdf";
	//void* 	realObjValue = NULL;
	ValRecord	vrRealObjValue;
	ValRecord	vrCastedObjValue; 
	#if	DETECT_PKCOL
	BAT	*tmpHashBat = NULL; 
	char	isCheckDone = 0; 
	BUN	tmpObjBun = BUN_NONE; 
	int	numPKcols = 0; 
	char	isPossiblePK = 0; 
	#endif
	#if	COUNT_DISTINCT_REFERRED_S
	BAT     *tmpFKHashBat = NULL;
	int	initHashBatgz = 0; 
	BUN	tmpFKRefBun = BUN_NONE; 
	char	isFKCol = 0; 
	#endif

	maxOrigPbt = ((oid)1 << (sizeof(BUN)*8 - NBITS_FOR_CSID)) - 1; 
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "RDFdistTriplesToCSs",
				"could not open the tokenizer\n");
	}

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	if ((mbat = BATdescriptor(*mbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		BBPreleaseref(obat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	if ((lmap = BATdescriptor(*lmapbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		BBPreleaseref(obat->batCacheid);
		BBPreleaseref(mbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	
	if ((rmap = BATdescriptor(*rmapbatid)) == NULL) {
		BBPreleaseref(sbat->batCacheid);
		BBPreleaseref(pbat->batCacheid);
		BBPreleaseref(obat->batCacheid);
		BBPreleaseref(mbat->batCacheid);
		BBPreleaseref(lmap->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat);
	mi = bat_iterator(mbat);
	
	tmpTblIdxPropIdxMap = (int*)malloc(sizeof(int) * cstablestat->numTables);
	initIntArray(tmpTblIdxPropIdxMap, cstablestat->numTables, -1); 

	tmplastInsertedS = -1; 
	

	lastP = BUN_NONE; 
	lastS = BUN_NONE; 
	
	printf("Reorganize the triple store by using %d CS tables \n", cstablestat->numTables);

	//setofBats = (BAT**)malloc(sizeof(BAT*) * cstablestat->numTables); 
	isSetLasttblIdx = 0; 

	BATloop(pbat, p, q){
		if (p % 1048576 == 0) printf(".");
		pbt = (oid *) BUNtloc(pi, p);
		sbt = (oid *) BUNtloc(si, p);
		obt = (oid *) BUNtloc(oi, p);
		
		//BATprint(pbat);
		//BATprint(sbat); 
		//BATprint(obat); 
		
		//printf(BUNFMT ": " BUNFMT "  |  " BUNFMT " | " BUNFMT "\n", p, *pbt, *sbt, *obt); 
		getTblIdxFromS(*sbt, &tblIdx, &tmpSoid);	
		//printf("  --> Tbl: %d  tmpSoid: " BUNFMT " | Last SubjId " BUNFMT "\n", tblIdx,tmpSoid, lastSubjId[tblIdx]);


		if (tblIdx == -1){	// This is for irregular triples, put them to pso table
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			//printf(" ==> To PSO \n");
			isFKCol = 0;
			continue; 
		}

		if (*pbt != lastP){
			if (*pbt > maxOrigPbt){	//This pbt has been changed according to the modification of Soid
				if (getOrigPbt(pbt, &origPbt, lmap, rmap) != MAL_SUCCEED){
					throw(RDF, "rdf.RDFdistTriplesToCSs","Problem in getting the orignal pbt ");
				} 	
				//printf("Pbt = " BUNFMT " ==> orignal pbt = " BUNFMT "\n", *pbt, origPbt); 
			}
			else {
				origPbt = *pbt;
			}

	
			//Get number of BATs for this p
			ppos = BUNfnd(BATmirror(propStat->pBat), &origPbt);
			if (ppos == BUN_NONE){
				throw(RDF, "rdf.RDFdistTriplesToCSs", "This prop must be in propStat bat");
			}

			tmpPtl =  propStat->plCSidx[ppos];
			updateTblIdxPropIdxMap(tmpTblIdxPropIdxMap, 
					tmpPtl.lstIdx, tmpPtl.lstInvertIdx,tmpPtl.numAdded);
			
			lastP = *pbt; 
			//lastS = *sbt; 
			lastS = BUN_NONE; 
			numMultiValues = 0;
			tmplastInsertedS = -1;

		}

		objType = getObjType(*obt); 
		assert (objType != BLANKNODE);


		tmpPropIdx = tmpTblIdxPropIdxMap[tblIdx]; 
		//printf(" PropIdx = %d \n", tmpPropIdx);
		tmpColIdx = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].defColIdx; 
		if (tmpColIdx == -1){ 	// This col is removed as an infrequent prop
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			continue; 
		}

		if (csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isDirtyFKProp){	//Check whether this URI have a reference 		
			if (objType != URI){ //Must be a dirty one --> put to pso
				//printf("Dirty FK at tbl %d | propId " BUNFMT " \n", tblIdx, *pbt);
				insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
				continue; 
			}
			else{ //  
				getTblIdxFromO(*obt,&tmpOidTblIdx);
				if (tmpOidTblIdx != csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblId){
					//printf("Dirty FK at tbl %d | propId " BUNFMT " \n", tblIdx, *pbt);
					insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
					continue; 
				}
			}
		}

		//printf(" Tbl: %d   |   Col: %d \n", tblIdx, tmpColIdx);
		#if COUNT_DISTINCT_REFERRED_S
		isFKCol = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isFKProp; 
		#endif
		
		istmpMVProp = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isMVProp; 
		defaultType = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].defaultType; 
		#if     DETECT_PKCOL
			isPossiblePK = 1;
			#if ONLY_URI_PK
			if (defaultType != URI) isPossiblePK = 0; 
			#endif
		#endif
		if (isSetLasttblIdx == 0){
			lastColIdx = tmpColIdx;
			lastPropIdx = tmpPropIdx; 
			lasttblIdx = tblIdx;
			cstablestat->lastInsertedS[tblIdx][tmpColIdx] = BUN_NONE;
			#if     DETECT_PKCOL
			if (isPossiblePK){
				tmpHashBat = BATnew(TYPE_void, TYPE_oid, lastSubjId[tblIdx] + 1);
				
				if (tmpHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpHashBat");
				}	
				(void)BATprepareHash(BATmirror(tmpHashBat));
				if (!(tmpHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for Bat");
				}

				if (BUNappend(tmpHashBat,obt, TRUE) == NULL){		//Insert the first value
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpHashBat");
				}
				isCheckDone = 0; 
				numPKcols++;
			}
			#endif
			#if COUNT_DISTINCT_REFERRED_S
			if (isFKCol){
				initHashBatgz = (csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport > smallHashBatsz)?smallHashBatsz:csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport;
				tmpFKHashBat = BATnew(TYPE_void, TYPE_oid, initHashBatgz + 1);

				if (tmpFKHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpFKHashBat");
				}	
				(void)BATprepareHash(BATmirror(tmpFKHashBat));
				if (!(tmpFKHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
				}
				if (BUNappend(tmpFKHashBat,obt, TRUE) == NULL){		//The first value
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpFKHashBat");
				}

			}
			#endif
			isSetLasttblIdx = 1; 
		}

		/* New column. Finish with lastTblIdx and lastColIdx. Note: This lastColIdx is
		 * the position of the prop in a final CS. Not the exact colIdx in MAINTBL or TYPETBL
		 * */
		if (tmpColIdx != lastColIdx || lasttblIdx != tblIdx){ 
			//Insert missing values for all columns of this property in this table

			if (fillMissingvaluesAll(cstablestat, csPropTypes, lasttblIdx, lastColIdx, lastPropIdx, lastSubjId) != MAL_SUCCEED){
				throw(RDF, "rdf.RDFdistTriplesToCSs", "Problem in filling missing values all");		
			}
				
			#if COUNT_DISTINCT_REFERRED_S
			if (csPropTypes[lasttblIdx].lstPropTypes[lastPropIdx].isFKProp ) {
				//printf("Update refcount for FK Col at: Table %d  Prop %d (Orig Ref size: %d) --> " BUNFMT "\n", lasttblIdx, lastPropIdx, csPropTypes[lasttblIdx].lstPropTypes[lastPropIdx].refTblSupport, BATcount(tmpFKHashBat)); 
				csPropTypes[lasttblIdx].lstPropTypes[lastPropIdx].numDisRefValues = BATcount(tmpFKHashBat);
				if (tmpFKHashBat != NULL){
					BBPreclaim(tmpFKHashBat);
					tmpFKHashBat = NULL; 
				}
			}
			if (isFKCol){
				initHashBatgz = (csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport > smallHashBatsz)?smallHashBatsz:csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport;
				tmpFKHashBat = BATnew(TYPE_void, TYPE_oid, initHashBatgz + 1);

				if (tmpFKHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpFKHashBat");
				}	
				(void)BATprepareHash(BATmirror(tmpFKHashBat));
				if (!(tmpFKHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
				}

				if (BUNappend(tmpFKHashBat,obt, TRUE) == NULL){		//The first value
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpFKHashBat");
				}
			}
			#endif

			lastColIdx = tmpColIdx; 
			lastPropIdx = tmpPropIdx; 
			lasttblIdx = tblIdx;
			tmplastInsertedS = -1;
			cstablestat->lastInsertedS[tblIdx][tmpColIdx] = BUN_NONE;

			#if     DETECT_PKCOL	
			if (isPossiblePK){
				if (tmpHashBat != NULL){
					BBPreclaim(tmpHashBat); 
					tmpHashBat = NULL; 
				}
				tmpHashBat = BATnew(TYPE_void, TYPE_oid, lastSubjId[tblIdx] + 1);

				if (tmpHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpHashBat");
				}	
				(void)BATprepareHash(BATmirror(tmpHashBat));
				if (!(tmpHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for Bat");
				}

				csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isPKProp = 1;  /* Assume that the object values are all unique*/

				if (BUNappend(tmpHashBat,obt, TRUE) == NULL){		//Insert the first value
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpHashBat");
				}
				isCheckDone = 0;
				numPKcols++;
			}
			#endif

			
		}
		else{

			#if     DETECT_PKCOL
			if (isCheckDone == 0 && isPossiblePK){
				tmpObjBun = BUNfnd(BATmirror(tmpHashBat),(ptr) obt);
				if (tmpObjBun == BUN_NONE){
					if (BUNappend(tmpHashBat,obt, TRUE) == NULL){		//Insert the first value
						throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpHashBat");
					}
				}
				else{
					isCheckDone = 1; 
					csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isPKProp = 0; 
					numPKcols--;
					//printf("Found duplicated value at " BUNFMT "  |  " BUNFMT " | " BUNFMT "\n", *pbt, *sbt, *obt);
				}
			}

			#endif
			#if COUNT_DISTINCT_REFERRED_S
			if (isFKCol){
				tmpFKRefBun = BUNfnd(BATmirror(tmpFKHashBat),(ptr) obt);
				if (tmpFKRefBun == BUN_NONE){

				       if (tmpFKHashBat->T->hash && BATcount(tmpFKHashBat) > 4 * tmpFKHashBat->T->hash->mask) {
						HASHdestroy(tmpFKHashBat);
						BAThash(BATmirror(tmpFKHashBat), 2*BATcount(tmpFKHashBat));

						if (!(tmpFKHashBat->T->hash)){
							throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
						}
					}
					if (BUNappend(tmpFKHashBat,obt, TRUE) == NULL){		
						throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpFKHashBat");
					}
				}
			}
			#endif
		}


		if (istmpMVProp == 1){	// This is a multi-valued prop
			//printf("Multi values prop \n"); 
			if (*sbt != lastS){ 	
				numMultiValues = 0;
				lastS = *sbt; 
			}

			assert(objType != MULTIVALUES); 	//TODO: Remove this
			tmpMVColIdx = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].colIdxes[(int)objType];
			tmpBat = cstablestat->lstcstable[tblIdx].colBats[tmpColIdx];
			getRealValue(&vrRealObjValue, *obt, objType, mi, mbat);
				
			for (i = 0; i < cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].numCol; i++){
				tmpmvBat = cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].mvBats[i];
				//BATprint(tmpmvBat);
				if (i == tmpMVColIdx){	
					// TODO: If i != 0, try to cast to default value		
					if (BUNfastins(tmpmvBat, ATOMnilptr(TYPE_void), VALget(&vrRealObjValue)) == NULL){
						throw(RDF, "rdf.RDFdistTriplesToCSs", " Error in Bunfastins ");
					} 
				}
				else{
					if (i == 0){	//The deafult type column
						//Check whether we can cast the value to the default type value
						if (rdfcast(objType, defaultType, &vrRealObjValue, &vrCastedObjValue) == 1){
							if (BUNfastins(tmpmvBat,ATOMnilptr(TYPE_void),VALget(&vrCastedObjValue)) == NULL){ 
								throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins ");
							} 	
							VALclear(&vrCastedObjValue);
						}
						else{
							if (BUNfastins(tmpmvBat,ATOMnilptr(TYPE_void),ATOMnilptr(tmpmvBat->ttype)) == NULL){
								throw(RDF, "rdf.RDFdistTriplesToCSs", "Error in Bunfastins ");
							} 
						}
					}
					else{
						if (BUNfastins(tmpmvBat,ATOMnilptr(TYPE_void),ATOMnilptr(tmpmvBat->ttype)) == NULL){ 
							throw(RDF, "rdf.RDFdistTriplesToCSs", "Error in Bunfastins ");
						}
					 
					}
				}
			
			}

			VALclear(&vrRealObjValue);

			if (numMultiValues == 0){	
				//In search the position of the first value 
				//to the correcponding column in the MAINTBL
				//First: Insert all missing value
				if ((int)tmpSoid > (tmplastInsertedS + 1)){
					fillMissingvalues(tmpBat, tmplastInsertedS + 1, (int)tmpSoid-1);
				}
				
				//BATprint(tmpmvBat);
				tmpmvValue = (oid)(BUNlast(tmpmvBat) - 1);
				//printf("Insert the refered oid " BUNFMT "for MV prop \n", tmpmvValue);
				if (BUNfastins(tmpBat, ATOMnilptr(TYPE_void), &tmpmvValue) == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins error");
				}
				//BATprint(tmpBat);
				
				//Insert this "key" to the key column of mv table.
				tmpmvKey = tmpmvValue; 
				if (BUNfastins(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].keyBat,ATOMnilptr(TYPE_void),&tmpmvKey) == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins error");		
				} 
				tmplastInsertedS = (int)tmpSoid; 
				
				lastColIdx = tmpColIdx; 
				lastPropIdx = tmpPropIdx; 
				lasttblIdx = tblIdx;
				
				numMultiValues++;
			}
			else{
				//Repeat referred "key" in the key column of mvtable
				if (BUNfastins(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].keyBat,ATOMnilptr(TYPE_void),&tmpmvKey) == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins error");		
				} 
			}
			
			continue; 
		}
		else{	
			//If there exist multi-valued prop, but handle them as single-valued prop.
			//Only first object value is stored. Other object values are 
			if (*sbt != lastS){
				lastS = *sbt; 
			}
			else{	// This is an extra object value
				insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
				//printf(" Extra object value ==> To PSO \n");
				continue; 
			}
		}


		tmpTableType = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].TableTypes[(int)objType]; 

		//printf("  objType: %d  TblType: %d \n", (int)objType,(int)tmpTableType);
		if (tmpTableType == PSOTBL){			//For infrequent type ---> go to PSO
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			//printf(" ==> To PSO \n");
			continue; 
		}

		if (tmpTableType == MAINTBL){
			curBat = cstablestat->lstcstable[tblIdx].colBats[tmpColIdx];
			//printf(" tmpColIdx = %d \n",tmpColIdx);
		}
		else{	//tmpTableType == TYPETBL
			tmpColExIdx = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].colIdxes[(int)objType];
			curBat = cstablestat->lstcstableEx[tblIdx].colBats[tmpColExIdx];
			//printf(" tmpColExIdx = %d \n",tmpColExIdx);
		}


		tmplastInsertedS = (cstablestat->lastInsertedS[tblIdx][tmpColIdx] == BUN_NONE)?(-1):(int)(cstablestat->lastInsertedS[tblIdx][tmpColIdx]);

		//If S is not continuous meaning that some S's have missing values for this property. Fill nils for them.
		if (fillMissingValueByNils(cstablestat, csPropTypes, tblIdx, tmpColIdx, tmpPropIdx, tmpColExIdx, tmpTableType, tmplastInsertedS + 1, (int)tmpSoid)!= MAL_SUCCEED){
			throw(RDF, "rdf.RDFdistTriplesToCSs", "Problem in filling missing values by Nils error");			
		}
		
		getRealValue(&vrRealObjValue, *obt, objType, mi, mbat);

		if (tmpTableType != MAINTBL){	//Check whether it can be casted to the default type
			tmpBat = cstablestat->lstcstable[tblIdx].colBats[tmpColIdx];
			if (rdfcast(objType, defaultType, &vrRealObjValue, &vrCastedObjValue) == 1){
				//printf("Casted a value (type: %d) to tables %d col %d (type: %d)  \n", objType, tblIdx,tmpColIdx,defaultType);
				if (BUNfastins(tmpBat, ATOMnilptr(TYPE_void), VALget(&vrCastedObjValue)) == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins error");		
				} 
	
				VALclear(&vrCastedObjValue);
			}
			else{
				if (BUNfastins(tmpBat, ATOMnilptr(TYPE_void),ATOMnilptr(tmpBat->ttype)) == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins error");		
				}
			}

		}
		
		if (BUNfastins(curBat, ATOMnilptr(TYPE_void), VALget(&vrRealObjValue)) == NULL){
			throw(RDF, "rdf.RDFdistTriplesToCSs", "Bunfastins error");		
		} 
		
		VALclear(&vrRealObjValue);
		
		//printf(BUNFMT": Table %d | column %d  for prop " BUNFMT " | sub " BUNFMT " | obj " BUNFMT "\n",p, tblIdx, 
		//					tmpColIdx, *pbt, tmpSoid, *obt); 
					
		//Update last inserted S
		cstablestat->lastInsertedS[tblIdx][tmpColIdx] = tmpSoid;

	}
	
	#if DETECT_PKCOL 
	if (tmpHashBat != NULL){
		BBPreclaim(tmpHashBat); 
		tmpHashBat = NULL; 
	}
	printf("Number of possible PK cols is: %d \n", numPKcols); 
	#endif

	#if COUNT_DISTINCT_REFERRED_S
	if (isFKCol){
		//Update FK referred count for the last csProp
		printf("LAST update ref count for FK Col at: Table %d  Prop %d (Orig Ref size: %d) \n", tblIdx, tmpPropIdx, csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport); 
		csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].numDisRefValues = BATcount(tmpFKHashBat);
		if (tmpFKHashBat != NULL){
			BBPreclaim(tmpFKHashBat);
			tmpFKHashBat = NULL; 
		}
	}
	#endif

	//HAVE TO GO THROUGH ALL BATS
	if (fillMissingvaluesAll(cstablestat, csPropTypes, lasttblIdx, lastColIdx, lastPropIdx, lastSubjId) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFdistTriplesToCSs", "Problem in filling missing values all");			
	}

	
	// Keep the batCacheId
	for (i = 0; i < cstablestat->numTables; i++){
		//printf("----- Table %d ------ \n",i );
		for (j = 0; j < cstablestat->numPropPerTable[i];j++){
			//printf("Column %d \n", j);
			cstablestat->lstbatid[i][j] = cstablestat->lstcstable[i].colBats[j]->batCacheid; 
			//BATprint(cstablestat->lstcstable[i].colBats[j]);
			if (csPropTypes[i].lstPropTypes[j].isMVProp){
				//printf("MV Columns: \n");
				for (k = 0; k < cstablestat->lstcstable[i].lstMVTables[j].numCol; k++){
					//BATprint(cstablestat->lstcstable[i].lstMVTables[j].mvBats[k]);
				}

			}

		}
	}
	

	*ret = 1; 

	printf(" ... Done \n");
	
	BBPunfix(sbat->batCacheid);
	BBPunfix(pbat->batCacheid);
	BBPunfix(obat->batCacheid);
	BBPunfix(mbat->batCacheid);
	BBPunfix(lmap->batCacheid);
	BBPunfix(rmap->batCacheid);

	free(tmpTblIdxPropIdxMap); 

	TKNZRclose(ret);

	return MAL_SUCCEED; 
}

str
RDFreorganize(int *ret, CStableStat *cstablestat, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold, int *mode){

	CSset		*freqCSset; 	/* Set of frequent CSs */
	oid		*subjCSMap = NULL;  	/* Store the corresponding CS Id for each subject */
	oid 		maxCSoid = 0; 
	BAT		*sbat = NULL, *obat = NULL, *pbat = NULL, *mbat = NULL;
	BATiter		si,pi,oi; 
	BUN		p,q; 
	BAT		*sNewBat, *lmap, *rmap, *oNewBat, *origobat, *pNewBat; 
	BUN		newId; 
	oid		*sbt; 
	oid		*lastSubjId; 	/* Store the last subject Id in each freqCS */
	//oid		*lastSubjIdEx; 	/* Store the last subject Id (of not-default type) in each freqCS */
	int		tblIdx; 
	oid		lastS;
	oid		l,r; 
	bat		oNewBatid, pNewBatid; 
	int		*csTblIdxMapping;	/* Store the mapping from a CS id to an index of a maxCS or mergeCS in freqCSset. */
	int		*mfreqIdxTblIdxMapping;  /* Store the mapping from the idx of a max/merge freqCS to the table Idx */
	int		*mTblIdxFreqIdxMapping;  /* Invert of mfreqIdxTblIdxMapping */
	int		numTables = 0; 
	PropStat	*propStat; 
	int		numdistinctMCS = 0; 
	int		maxNumPwithDup = 0;
	//CStableStat	*cstablestat;
	CSPropTypes	*csPropTypes; 
	CSlabel		*labels;
	CSrel		*csRelMergeFreqSet = NULL;
	CSrel		*csRelFinalFKs = NULL;   	//Store foreign key relationships 

	//int 		curNumMergeCS;
	//oid		*mergeCSFreqCSMap;
	int		numSampleTbl = 0;  
	
	clock_t 	curT;
	clock_t		tmpLastT; 
	
	str		returnStr; 

	tmpLastT = clock();
	freqCSset = initCSset();

	if (RDFextractCSwithTypes(ret, sbatid, pbatid, obatid, mapbatid, freqThreshold, freqCSset,&subjCSMap, &maxCSoid, &maxNumPwithDup, &labels, &csRelMergeFreqSet) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in extracting CSs");
	}
	

	curT = clock(); 
	printf (" Total schema extraction process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	printf("Start re-organizing triple store for " BUNFMT " CSs \n", maxCSoid + 1);

	csTblIdxMapping = (int *) malloc (sizeof (int) * (maxCSoid + 1)); 
	initIntArray(csTblIdxMapping, (maxCSoid + 1), -1);

	mfreqIdxTblIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded); 
	initIntArray(mfreqIdxTblIdxMapping , freqCSset->numCSadded, -1);

	mTblIdxFreqIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded);  // TODO: little bit reduntdant space
	initIntArray(mTblIdxFreqIdxMapping , freqCSset->numCSadded, -1);

	//Mapping from from CSId to TableIdx 
	printf("Init CS tableIdxMapping \n");
	initCSTableIdxMapping(freqCSset, csTblIdxMapping, mfreqIdxTblIdxMapping, mTblIdxFreqIdxMapping, &numTables);


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

	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat); 

	/* Get possible types of each property in a table (i.e., mergedCS) */
	csPropTypes = (CSPropTypes*)GDKmalloc(sizeof(CSPropTypes) * numTables); 
	initCSPropTypes(csPropTypes, freqCSset, numTables);
	
	printf("Extract CSPropTypes \n");
	RDFExtractCSPropTypes(ret, sbat, si, pi, oi, subjCSMap, csTblIdxMapping, csPropTypes, maxNumPwithDup);
	genCSPropTypesColIdx(csPropTypes, numTables, freqCSset);
	printCSPropTypes(csPropTypes, numTables, freqCSset, *freqThreshold);
	//Collecting the statistic
	printf("Get table statistics by CSPropTypes \n");
	getTableStatisticViaCSPropTypes(csPropTypes, numTables, freqCSset, *freqThreshold);
	
	#if COLORINGPROP
	/* Update list of support for properties in freqCSset */
	updatePropSupport(csPropTypes, numTables, freqCSset);
	#endif

	curT = clock(); 
	printf (" Preparing process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	// print labels
	printf("Start exporting labels \n"); 
	
	#if EXPORT_LABEL
	exportLabels(labels, freqCSset, csRelMergeFreqSet, *freqThreshold, mTblIdxFreqIdxMapping, mfreqIdxTblIdxMapping, numTables);
	#endif


	curT = clock(); 
	printf (" Export label process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		
	

	csRelFinalFKs = getFKBetweenTableSet(csRelMergeFreqSet, freqCSset, csPropTypes,mfreqIdxTblIdxMapping,numTables);
	printFKs(csRelFinalFKs, *freqThreshold, numTables, csPropTypes); 

	// Init CStableStat
	initCStables(cstablestat, freqCSset, csPropTypes, numTables, labels, mTblIdxFreqIdxMapping);
	
	// Summarize the statistics
	getStatisticFinalCSs(freqCSset, sbat, *freqThreshold, numTables, mTblIdxFreqIdxMapping, csPropTypes);

	/* Extract sample data for the evaluation */
	{	

	BAT	*outputBat;
	CSSample *csSample; 

	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	//Generate evaluating tables

	numSampleTbl = (NUM_SAMPLETABLE > (numTables/2))?(numTables/2):NUM_SAMPLETABLE;

	printf("Select list of sample tables \n");
	outputBat = generateTablesForEvaluating(freqCSset, numSampleTbl, mTblIdxFreqIdxMapping, numTables);
	assert (BATcount(outputBat) == (oid) numSampleTbl);
	csSample = (CSSample*)malloc(sizeof(CSSample) * numSampleTbl);
	printf("Select sample instances for %d tables \n", numSampleTbl);
	initSampleData(csSample, outputBat, freqCSset, mTblIdxFreqIdxMapping, labels);
	RDFExtractSampleData(ret, sbat, si, pi, oi, subjCSMap, csTblIdxMapping, maxNumPwithDup, csSample, outputBat, numSampleTbl);
	printsubsetFromCSset(freqCSset, outputBat, numSampleTbl, mTblIdxFreqIdxMapping, labels);
	printSampleData(csSample, freqCSset, mbat, numSampleTbl);
	freeSampleData(csSample, numSampleTbl);
	BBPreclaim(outputBat);
	BBPunfix(mbat->batCacheid);
	}

	if (*mode == EXPLOREONLY){
		printf("Only explore the schema information \n");
		freeLabels(labels, freqCSset);
		freeCSrelSet(csRelMergeFreqSet,freqCSset->numCSadded);
		freeCSset(freqCSset); 
		free(subjCSMap);
		free(csTblIdxMapping);
		free(mfreqIdxTblIdxMapping);
		free(mTblIdxFreqIdxMapping);
		freeCSPropTypes(csPropTypes,numTables);
		printf("Finish & Exit exploring step! \n"); 
		
		return MAL_SUCCEED;
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
	
	lastSubjId = (oid *) malloc (sizeof(oid) * cstablestat->numTables); 
	initArray(lastSubjId, cstablestat->numTables, -1); 
	
	printf("Re-assigning Subject oids ... ");
	lastS = -1; 
	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);
		tblIdx = csTblIdxMapping[subjCSMap[*sbt]];
	
		if (tblIdx != -1){

			if (lastS != *sbt){	//new subject
				lastS = *sbt; 
				
				newId = lastSubjId[tblIdx] + 1;
				newId |= (BUN)(tblIdx + 1) << (sizeof(BUN)*8 - NBITS_FOR_CSID);
				lastSubjId[tblIdx]++;
				
				// subject belongs to CS0 will  have 
				// the value for 'CS identifying field' of this subject: 1

				l = *sbt; 
				r = newId; 

				lmap = BUNappend(lmap, &l, TRUE);
				rmap = BUNappend(rmap, &r, TRUE);

			}

		}
		else{	// Does not belong to a freqCS. Use original subject Id
			newId = *sbt; 
		}

		sNewBat = BUNappend(sNewBat, &newId, TRUE);	
		//printf("Tbl: %d  || Convert s: " BUNFMT " to " BUNFMT " \n", tblIdx, *sbt, newId); 
		
	}


	//BATprint(VIEWcreate(BATmirror(lmap),rmap)); 
	
	origobat = getOriginalUriOBat(obat); 	//Return obat without type-specific information for URI & BLANKNODE
						//This is to get the same oid as the subject oid for a same URI, BLANKNODE
						//--> There will be no BLANKNODE indication in this obat object oid. 
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

	printf("Done! \n");
	printf("Sort triple table according to P, S, O order ... ");
	if (triplesubsort(&pNewBat, &sNewBat, &oNewBat) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in sorting PSO");	
	}	
	printf("Done  \n");

	//BATprint(pNewBat);
	//BATprint(sNewBat);

	propStat = getPropStatisticsByTable(numTables, mTblIdxFreqIdxMapping, freqCSset,  &numdistinctMCS); 
	
	//printPropStat(propStat,0); 
	
	curT = clock(); 
	printf (" Prepare and create sub-sorted PSO took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		
	returnStr = RDFdistTriplesToCSs(ret, &sNewBat->batCacheid, &pNewBat->batCacheid, &oNewBat->batCacheid, mapbatid, 
			&lmap->batCacheid, &rmap->batCacheid, propStat, cstablestat, csPropTypes, lastSubjId);
	printf("Return value from RDFdistTriplesToCSs is %s \n", returnStr);
	if (returnStr != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in distributing triples to BATs using CSs");		
	}
		
	curT = clock(); 
	printf ("RDFdistTriplesToCSs process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		
	
	printFKMultiplicityFromCSPropTypes(csPropTypes, numTables, freqCSset, *freqThreshold);
		
	freeCSrelSet(csRelMergeFreqSet,freqCSset->numCSadded);
	freeCSrelSet(csRelFinalFKs, numTables); 
	freeCSPropTypes(csPropTypes,numTables);
	freeLabels(labels, freqCSset);
	freeCSset(freqCSset); 
	free(subjCSMap); 
	free(csTblIdxMapping);
	free(mfreqIdxTblIdxMapping);
	free(mTblIdxFreqIdxMapping);
	free(lastSubjId);
	//free(lastSubjIdEx); 
	freePropStat(propStat);
	//freeCStableStat(cstablestat); 
	//

	BBPreclaim(lmap);
	BBPreclaim(rmap); 
	BBPunfix(sbat->batCacheid);
	BBPreclaim(sNewBat);
	BBPunfix(obat->batCacheid); 
	BBPreclaim(origobat);
	BBPreclaim(oNewBat); 
	BBPunfix(pbat->batCacheid); 
	BBPreclaim(pNewBat); 

	return MAL_SUCCEED; 
}
