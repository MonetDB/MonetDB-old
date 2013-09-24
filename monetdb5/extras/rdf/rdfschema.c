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
#include "rdfschema.h"
#include "rdflabels.h"
#include "rdfretrieval.h"
#include "algebra.h"
#include <gdk.h>
#include <hashmap/hashmap.h>
#include "tokenizer.h"
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

static char
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
CSPropTypes* initCSPropTypes(CSset* freqCSset, int numMergedCS){
	int numFreqCS = freqCSset->numCSadded;
	int i, j, k ;
	int id; 

	CSPropTypes* csPropTypes = (CSPropTypes*)GDKmalloc(sizeof(CSPropTypes) * numMergedCS); 
	
	id = 0; 
	for (i = 0; i < numFreqCS; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){   // Only use the maximum or merge CS		
			csPropTypes[id].freqCSId = i; 
			csPropTypes[id].numProp = freqCSset->items[i].numProp;
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
				csPropTypes[id].lstPropTypes[j].lstTypes = (char*)GDKmalloc(sizeof(char) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].lstFreq = (int*)GDKmalloc(sizeof(int) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].colIdxes = (int*)GDKmalloc(sizeof(int) * csPropTypes[id].lstPropTypes[j].numType);
				csPropTypes[id].lstPropTypes[j].TableTypes = (char*)GDKmalloc(sizeof(char) * csPropTypes[id].lstPropTypes[j].numType);

				for (k = 0; k < csPropTypes[id].lstPropTypes[j].numType; k++){
					csPropTypes[id].lstPropTypes[j].lstFreq[k] = 0; 
					csPropTypes[id].lstPropTypes[j].TableTypes[k] = 0; 
					csPropTypes[id].lstPropTypes[j].colIdxes[k] = -1; 
				}

			}

			id++;
		}
	}

	assert(id == numMergedCS);

	return csPropTypes;
}

static 
void genCSPropTypesColIdx(CSPropTypes* csPropTypes, int numMergedCS, CSset* freqCSset){
	int i, j, k; 
	int tmpMaxFreq;  
	int defaultIdx;	 /* Index of the default type for a property */
	int curTypeColIdx = 0;

	(void) freqCSset;

	for (i = 0; i < numMergedCS; i++){
		curTypeColIdx = 0; 
		for(j = 0; j < csPropTypes[i].numProp; j++){
			tmpMaxFreq = csPropTypes[i].lstPropTypes[j].lstFreq[0];
			defaultIdx = 0; 
			for (k = 0; k < csPropTypes[i].lstPropTypes[j].numType; k++){
				if (csPropTypes[i].lstPropTypes[j].lstFreq[k] > tmpMaxFreq){
					tmpMaxFreq =  csPropTypes[i].lstPropTypes[j].lstFreq[k];
					defaultIdx = k; 	
				}
				if (csPropTypes[i].lstPropTypes[j].lstFreq[k] < csPropTypes[i].lstPropTypes[j].propFreq * 0.1){
					//non-frequent type goes to PSO
					csPropTypes[i].lstPropTypes[j].TableTypes[k] = PSOTBL; 
				}
				else
					csPropTypes[i].lstPropTypes[j].TableTypes[k] =TYPETBL;
			}
			/* One type is set to be the default type (in the main table) */
			csPropTypes[i].lstPropTypes[j].TableTypes[defaultIdx] = MAINTBL; 
			csPropTypes[i].lstPropTypes[j].colIdxes[defaultIdx] = j;
			csPropTypes[i].lstPropTypes[j].defaultType = defaultIdx; 

			/* Count the number of column needed */
			for (k = 0; k < csPropTypes[i].lstPropTypes[j].numType; k++){
				if (csPropTypes[i].lstPropTypes[j].TableTypes[k] == TYPETBL){
					csPropTypes[i].lstPropTypes[j].colIdxes[k] = curTypeColIdx; 
					curTypeColIdx++;
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
	double	threshold = 1.1; 
	double  tmpRatio; 

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
			tmpRatio = (double) (csPropTypes[i].lstPropTypes[j].propCover / (csPropTypes[i].lstPropTypes[j].numSingleType + csPropTypes[i].lstPropTypes[j].numMVType));

			if ((csPropTypes[i].lstPropTypes[j].numMVType > 0) && (tmpRatio > threshold)){
				tmpIsMVCSFilter = 1; 
				numMVColsFilter++;
			}

			fprintf(fout, "  P " BUNFMT "(%d | freq: %d | cov:%d | Null: %d | Single: %d | Multi: %d) \n", 
					csPropTypes[i].lstPropTypes[j].prop, csPropTypes[i].lstPropTypes[j].defaultType,
					csPropTypes[i].lstPropTypes[j].propFreq, csPropTypes[i].lstPropTypes[j].propCover,
					csPropTypes[i].lstPropTypes[j].numNull, csPropTypes[i].lstPropTypes[j].numSingleType, csPropTypes[i].lstPropTypes[j].numMVType);
			fprintf(fout, "         ");
			for (k = 0; k < csPropTypes[i].lstPropTypes[j].numType; k++){
				fprintf(fout, " Type %d (%d)  | ", k, csPropTypes[i].lstPropTypes[j].lstFreq[k]);
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
/*
 * Add types of properties 
 * Note that the property list is sorted by prop's oids
 * E.g., buffP = {3, 5, 7}
 * csPropTypes[tbIdx] contains properties {1,3,4,5,7} with types for each property and frequency of each <property, type>
 * */
static 
void addPropTypes(char *buffTypes, oid* buffP, int numP, int* buffCover, int csId, int* csTblIdxMapping, CSPropTypes* csPropTypes){
	int i,j; 
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
	for (i = 0; i < csSet->numCSadded; i ++){
		if (csSet->items[i].parentFreqIdx == -1){
			num++;	
		}
	}

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
	mergecs->type = MERGECS; 
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
	BAT 	*tmpBat; 
	int 	*first; 
	int 	last; 

	lstDistinctFreqId = (int*) malloc(sizeof(int) * num); /* A bit redundant */
	
	tmpBat = BATnew(TYPE_void, TYPE_int, num);

	for (i = 0; i < num; i++){
		tmpBat = BUNappend(tmpBat, &lstMergeCSFreqId[i], TRUE);
	}

	/* Sort the array of the freqIdx list in order to remove duplication */
	
	//TODO: Ask whether there is a sorting function available for an array
	//TODO: Ask why it is not possible by using memcpy
	
	//memcpy(Tloc(tmpBat, BUNfirst(tmpBat)), lstMergeCSFreqId, sizeof(int) * num); 
	//memcpy(Hloc(tmpBat, BUNfirst(tmpBat)), hSeq, sizeof(oid) * num); 
	//BATsetcount(tmpBat, (BUN) (tmpBat->batCount + num));

	BATorder(BATmirror(tmpBat));

	first = (int*)Tloc(tmpBat, BUNfirst(tmpBat));
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

	BBPreclaim(tmpBat);

	return lstDistinctFreqId; 

}
/*
Multi-way merging for list of freqCS
*/
static 
void mergeMultiCS(CSset *freqCSset, int *lstFreqId, int num, oid *mergecsId){
	
	int 	i, j, tmpIdx; 
	int 	*lstMergeCSFreqId, *lstDistinctFreqId; 
	int 	numDistinct = 0; 
	int	mergeNumConsistsOf = 0;
	int	tmpFreqIdx; 
	int 	tmpConsistFreqIdx; 
	CS	*newmergeCS; 
	char 	isExistingMergeCS = 0;
	int	mergecsFreqIdx = -1;
	int	*_tmp; 
	oid	*_tmp2; 
	oid	*tmpPropList; 
	int	totalSupport = 0;
	int	totalCoverage = 0; 
	int 	numCombinedP = 0; 
	int	tmpParentIdx;	



	/* Get the list of merge FreqIdx */
	lstMergeCSFreqId = (int*) malloc(sizeof(int) * num); 
	
	//printf("Number of input %d \n",num);
	//printf("List of input freqIdx: \n");
	//for (i = 0; i < num; i++) printf("  %d",lstFreqId[i]);
	//printf("\n");

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
	//printf("\n");

	if (isExistingMergeCS == 0) mergecsFreqIdx = freqCSset->numCSadded; 

	lstDistinctFreqId = getDistinctList(lstMergeCSFreqId,num, &numDistinct);

	//printf("Number of distinct values %d \n", numDistinct);	
	// printf("List of distinct input parent freqIdx: \n");
	//for (i = 0; i < numDistinct; i++) printf("  %d",lstDistinctFreqId[i]);
	//printf("\n"); 
	
	if (numDistinct < 2){
		free(lstMergeCSFreqId);
		free(lstDistinctFreqId);
		return;
	}
	
	//if (isExistingMergeCS) printf("They will be merged into an existing  %d \n",mergecsFreqIdx);
	//else  printf("They will be merged into a new %d \n",mergecsFreqIdx);


	/* Create or not create a new CS */
	if (isExistingMergeCS){
		newmergeCS = (CS*) &(freqCSset->items[mergecsFreqIdx]);
	}
	else{
		newmergeCS = (CS*) malloc (sizeof (CS));
		newmergeCS->support = 0;
		newmergeCS->coverage = 0; 
	}


	

	/* Calculate number of consistsOf in the merged CS 

	 and  Update support and coverage: Total of all suppors */
	//printf("Distinct: \n");
	for (i = 0; i < numDistinct; i++){
		tmpFreqIdx = lstDistinctFreqId[i]; 
		//printf("CS%d (%d)  ", tmpFreqIdx, freqCSset->items[tmpFreqIdx].numConsistsOf);
		mergeNumConsistsOf += freqCSset->items[tmpFreqIdx].numConsistsOf; 
	}
	
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
	tmpIdx = 0;
	for (i = 0; i < numDistinct; i++){
		tmpFreqIdx = lstDistinctFreqId[i];
		for (j = 0; j < freqCSset->items[tmpFreqIdx].numConsistsOf; j++){
			tmpConsistFreqIdx =  freqCSset->items[tmpFreqIdx].lstConsistsOf[j];
			newmergeCS->lstConsistsOf[tmpIdx] = tmpConsistFreqIdx; 
			//Reset the parentFreqIdx
			freqCSset->items[tmpConsistFreqIdx].parentFreqIdx = mergecsFreqIdx;
			tmpIdx++;
		}
		freqCSset->items[tmpFreqIdx].parentFreqIdx = mergecsFreqIdx;
		//Update support
		totalSupport += freqCSset->items[tmpFreqIdx].support; 
		totalCoverage += freqCSset->items[tmpFreqIdx].coverage;
	}
	assert(tmpIdx == mergeNumConsistsOf);
	newmergeCS->numConsistsOf = mergeNumConsistsOf;

	/*Reset parentIdx */
	newmergeCS->parentFreqIdx = -1;
	newmergeCS->type = MERGECS;

	newmergeCS->support = totalSupport;
	newmergeCS->coverage = totalCoverage; 

	/*Merge the list of prop list */
	//printf("Merge list of prop from %d cs .... ", numDistinct);
	tmpPropList = mergeMultiPropList(freqCSset, lstDistinctFreqId, numDistinct, &numCombinedP);
	//printf("Done (numCombinedP = %d) \n",numCombinedP);
	//printf("Cobined P has %d props \n", numCombinedP); 
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
	free(lstDistinctFreqId);
}

#endif /* USE_MULTIWAY_MERGING */

static 
str printFreqCSSet(CSset *freqCSset, BAT *freqBat, BAT *mapbat, char isWriteTofile, int freqThreshold, CSlabel* labels){

	int 	i; 
	int 	j; 
	int 	*freq; 
	FILE 	*fout, *fout2; 
	char 	filename[100], filename2[100];
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
		strcpy(filename2, "max");
		strcat(filename2, filename);  
		strcat(filename, ".txt");
		strcat(filename2, ".txt");

		fout = fopen(filename,"wt"); 
		fout2 = fopen(filename2, "wt");

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
				}

				// Filter max freq cs set
				if (cs.type == MAXCS){
					fprintf(fout2,"CS " BUNFMT " (Freq: %d) | Subject: %s  | Parent " BUNFMT " \n", cs.csId, cs.support, subStr, cs.csId);
				}
					
				GDKfree(subStr);
			}
			else{
				fprintf(fout,"CS " BUNFMT " (Freq: %d) | Subject: NOTAVAI  | FreqParentIdx %d \n", cs.csId, *freq, cs.parentFreqIdx);

				if (cs.type == MAXCS){
					fprintf(fout2,"CS " BUNFMT " (Freq: %d) | Subject: NOTAVAI  | Parent " BUNFMT " \n", cs.csId, cs.support, cs.csId);
				}
					
			}
			#endif	

			for (j = 0; j < cs.numProp; j++){
				takeOid(cs.lstProp[j], &propStr);
				//fprintf(fout, "  P:" BUNFMT " --> ", cs.lstProp[j]);	
				fprintf(fout, "  P(" BUNFMT "):%s --> ", cs.lstProp[j],propStr);	
				if (cs.type == MAXCS){
					fprintf(fout2, "  P:%s --> ", propStr);
				}

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
					if (cs.type == MAXCS){
						fprintf(fout2, "  O: %s \n", objStr);
					}

					if (objType == URI || objType == BLANKNODE){
						GDKfree(objStr);
					}
				}
				#endif


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
				fprintf(fout,"          %s\n", propStr);
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
void mergeCSbyS4(CSset *freqCSset, CSlabel* labels, oid *mergeCSFreqCSMap, int curNumMergeCS){

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
		#if USE_LABEL_FINDING_MAXCS
		isLabelComparable = 0;
		if (labels[i].name != BUN_NONE) isLabelComparable = 1; // no "DUMMY"
		#endif

		for (j = (i+1); j < numMergeCS; j++){
			freqId2 = mergeCSFreqCSMap[j];
			isDiffLabel = 0; 
			#if USE_LABEL_FINDING_MAXCS
			if (isLabelComparable == 0 || strcmp(labels[freqId1].name, labels[freqId2].name) != 0) {
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
						break; 
					}
				}
				else if (numP2 < numP1 && (numP1-numP2)< MAX_SUB_SUPER_NUMPROP_DIF){
					if (isSubset(freqCSset->items[freqId1].lstProp, freqCSset->items[freqId2].lstProp,  
							numP1,numP2) == 1) { 
						/* CSj is a subset of CSi */
						freqCSset->items[freqId2].parentFreqIdx = freqId1; 
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
PropStat* getPropStatisticsByTable(CSset* freqCSset, int* mfreqIdxTblIdxMapping, int *numdistinctMCS){

	int i, j, k; 
	CS cs;

	PropStat* propStat; 
	
	propStat = initPropStat(); 

	k = 0; 

	for (i = 0; i < freqCSset->numCSadded; i++){

		if (freqCSset->items[i].parentFreqIdx == -1){	// Only use the maximum or merge CS 
			cs = (CS)freqCSset->items[i];
			k++; 
			for (j = 0; j < cs.numProp; j++){
				addaProp(propStat, cs.lstProp[j], mfreqIdxTblIdxMapping[i], j);
			}

			if (cs.numProp > propStat->maxNumPPerCS)
				propStat->maxNumPPerCS = cs.numProp;
		}
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
		if (freq < csRel.lstCnt[i] * 100){			
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
				candidate = labels[i].candidates[j];
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
				candidate = labels[i].candidates[j];
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
void mergeMaxFreqCSByS1(CSset *freqCSset, CSlabel* labels, oid *mergecsId){
	int 		i; 

	#if !USE_MULTIWAY_MERGING
	int		j,k;
	int 		freqId1, freqId2;
	CS     		*mergecs;
	oid		existMergecsId = BUN_NONE; 
	CS		*cs1, *cs2;
	CS		*existmergecs, *mergecs1, *mergecs2; 
	#endif
	LabelStat	*labelStat = NULL; 

	labelStat = initLabelStat(); 
	buildLabelStat(labelStat, labels, freqCSset, TOPK);
	printf("Num FreqCSadded before using S1 = %d \n", freqCSset->numCSadded);

	for (i = 0; i < labelStat->numLabeladded; i++){
		if (labelStat->lstCount[i] > 1){
			/*TODO: Multi-way merge */
			#if USE_MULTIWAY_MERGING	
			mergeMultiCS(freqCSset,  labelStat->freqIdList[i], labelStat->lstCount[i], mergecsId); 
			#else
			freqId1 = labelStat->freqIdList[i][0];
			cs1 = (CS*) &(freqCSset->items[freqId1]);
			for (j = 1; j < labelStat->lstCount[i]; j++){
				freqId2 = labelStat->freqIdList[i][j];
				cs2 = (CS*) &(freqCSset->items[freqId2]);
				//Check whether these CS's belong to any mergeCS
				if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
					mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, *mergecsId);
					//addmergeCStoSet(mergecsSet, *mergecs);
					cs1->parentFreqIdx = freqCSset->numCSadded;
					cs2->parentFreqIdx = freqCSset->numCSadded;
					addCStoSet(freqCSset,*mergecs);
					free(mergecs);

					mergecsId[0]++;
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
			#endif /* USE_MULTIWAY_MERGING */
		}
	}

	freeLabelStat(labelStat);
}

static
void mergeMaxFreqCSByS6(CSrel *csrelMergeFreqSet, CSset *freqCSset, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId){
	int 		i; 
	int 		freqId;
	//int 		relId; 
	//CS*		cs1;
	CSrelSum 	*csRelSum; 
	int		maxNumRefPerCS = 0; 
	int 		j, k; 
	#if 		!USE_MULTIWAY_MERGING
	int 		freqId1, freqId2;
	int 		m; 
	CS     		*mergecs;
	oid		existMergecsId = BUN_NONE; 
	CS		*cs1, *cs2;
	CS		*existmergecs, *mergecs1, *mergecs2; 
	#endif	

	char 		filename[100];
	FILE		*fout; 
	int		maxNumPropInMergeCS =0;
	//int 		numCombinedP = 0; 
	
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
			fprintf(fout, "csRelSum " BUNFMT ": ",csRelSum->origFreqIdx);
			for (j = 0; j < csRelSum->numProp; j++){
				if ( csRelSum->numPropRef[j] > 1){
					fprintf(fout, "  P " BUNFMT " -->",csRelSum->lstPropId[j]);
					for (k = 0; k < csRelSum->numPropRef[j]; k++){
						fprintf(fout, " %d | ", csRelSum->freqIdList[j][k]);
					}	
					/* Merge each refCS into the first CS. 
					 * TODO: The Multi-way merging should be better
					 * */ 
					//mergeMultiPropList(freqCSset, csRelSum->freqIdList[j],csRelSum->numPropRef[j] , &numCombinedP);
					#if USE_MULTIWAY_MERGING	
					mergeMultiCS(freqCSset, csRelSum->freqIdList[j],csRelSum->numPropRef[j], mergecsId); 
					#else
					freqId1 = csRelSum->freqIdList[j][0];
					cs1 = (CS*) &(freqCSset->items[freqId1]);
					for (k = 1; k < csRelSum->numPropRef[j]; k++){
						freqId2 = csRelSum->freqIdList[j][k];
						cs2 = (CS*) &(freqCSset->items[freqId2]);
						//Check whether these CS's belong to any mergeCS
						if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
							mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, *mergecsId);
							//addmergeCStoSet(mergecsSet, *mergecs);
							cs1->parentFreqIdx = freqCSset->numCSadded;
							cs2->parentFreqIdx = freqCSset->numCSadded;
							//printf("Merge into %d \n", freqCSset->numCSadded);
							addCStoSet(freqCSset,*mergecs);
							free(mergecs);

							mergecsId[0]++;
						}
						else if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx != -1){
							existMergecsId = cs2->parentFreqIdx;
							existmergecs = (CS*) &(freqCSset->items[existMergecsId]);
							mergeACStoExistingmergeCS(*cs1,freqId1, existmergecs);
							cs1->parentFreqIdx = existMergecsId; 
							//printf("Merge into "BUNFMT" \n", existMergecsId);
							
						}
						
						else if (cs1->parentFreqIdx != -1 && cs2->parentFreqIdx == -1){
							existMergecsId = cs1->parentFreqIdx;
							existmergecs = (CS*)&(freqCSset->items[existMergecsId]);
							mergeACStoExistingmergeCS(*cs2,freqId2, existmergecs);
							cs2->parentFreqIdx = existMergecsId; 
							//printf("Merge into "BUNFMT" \n", existMergecsId);
						}
						else if (cs1->parentFreqIdx != cs2->parentFreqIdx){
							mergecs1 = (CS*)&(freqCSset->items[cs1->parentFreqIdx]);
							mergecs2 = (CS*)&(freqCSset->items[cs2->parentFreqIdx]);
							
							mergeTwomergeCS(mergecs1, mergecs2, cs1->parentFreqIdx);
							//printf("Merge into %d \n", cs1->parentFreqIdx);
							//Re-map for all maxCS in mergecs2
							for (m = 0; m < mergecs2->numConsistsOf; m++){
								freqCSset->items[mergecs2->lstConsistsOf[m]].parentFreqIdx = cs1->parentFreqIdx;
							}
						}

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
char isSemanticSimilar(int freqId1, int freqId2, CSlabel* labels, OntoUsageNode *tree, int numOrigFreqCS){	/*Rule S1 S2 S3*/
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
			return 1;
		}

	}


	return 0;
}

static
void mergeCSByS3S5(CSset *freqCSset, CSlabel* labels, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId,OntoUsageNode *ontoUsageTree){
	int 		i, j, k; 
	int 		freqId1, freqId2; 
	float 		simscore = 0.0; 
	CS     		*mergecs;
	oid		existMergecsId = BUN_NONE; 
	int 		numCombineP = 0; 
	CS		*cs1, *cs2;
	CS		*existmergecs, *mergecs1, *mergecs2; 

	PropStat	*propStat; 	/* Store statistics about properties */
	char		isLabelComparable = 0; 
	char		isSameLabel = 0; 

	

	
	(void) labels;
	(void) isLabelComparable;


	propStat = initPropStat();
	getPropStatisticsFromMergeCSs(propStat, curNumMergeCS, mergeCSFreqCSMap, freqCSset); /*TODO: Get PropStat from MaxCSs or From mergedCS only*/

	for (i = 0; i < curNumMergeCS; i++){		/*TODO: Only go through the list of mergedCS. */
		freqId1 = mergeCSFreqCSMap[i];
		//printf("Label of %d CS is %s \n", freqId1, labels[freqId1].name);
		isLabelComparable = 0; 
		if (labels[freqId1].name != BUN_NONE) isLabelComparable = 1; // no "DUMMY"

		cs1 = (CS*) &(freqCSset->items[freqId1]);
	 	for (j = (i+1); j < curNumMergeCS; j++){
			freqId2 = mergeCSFreqCSMap[j];
			cs2 = (CS*) &(freqCSset->items[freqId2]);
			isSameLabel = 0; 

			#if	USE_LABEL_FOR_MERGING
			if (isLabelComparable == 1 && isSemanticSimilar(freqId1, freqId2, labels, ontoUsageTree,freqCSset->numOrigFreqCS) == 1){
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
				//Check whether these CS's belong to any mergeCS
				if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
					mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, *mergecsId);
					//addmergeCStoSet(mergecsSet, *mergecs);
					cs1->parentFreqIdx = freqCSset->numCSadded;
					cs2->parentFreqIdx = freqCSset->numCSadded;
					addCStoSet(freqCSset,*mergecs);
					free(mergecs);

					mergecsId[0]++;


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
	oid*		buffP;
	oid		curP; 

	buffTypes = (char *) malloc(sizeof(char) * (maxNumPwithDup + 1)); 
	buffP = (oid *) malloc(sizeof(oid) * (maxNumPwithDup + 1));
	buffCoverage = (int *)malloc(sizeof(int) * (maxNumPwithDup + 1));

	numPwithDup = 0;
	curS = 0; 
	curP = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				addPropTypes(buffTypes, buffP, numPwithDup, buffCoverage, subjCSMap[curS], csTblIdxMapping, csPropTypes);
			}
			curS = *sbt; 
			numPwithDup = 0;
			curP = 0; 
		}
				
		obt = (oid *) BUNtloc(oi, p); 
		/* Check type of object */
		objType = (char) ((*obt) >> (sizeof(BUN)*8 - 4))  &  7 ;	/* Get two bits 63th, 62nd from object oid */
	
		pbt = (oid *) BUNtloc(pi, p);

		if (curP == *pbt){
			#if USE_MULTIPLICITY == 1	
			// Update the object type for this P as MULTIVALUES	
			buffTypes[numPwithDup-1] = MULTIVALUES; 
			buffCoverage[numPwithDup-1]++;
			#else
			buffTypes[numPwithDup] = objType;
			numPwithDup++;
			#endif
		}
		else{			
			buffTypes[numPwithDup] = objType; 
			buffP[numPwithDup] = *pbt;
			buffCoverage[numPwithDup] = 1; 
			numPwithDup++; 
			curP = *pbt; 
		}


	}
	
	/* Check for the last CS */
	addPropTypes(buffTypes, buffP, numPwithDup, buffCoverage, subjCSMap[curS], csTblIdxMapping, csPropTypes);

	free (buffTypes); 
	free (buffP); 
	free (buffCoverage);

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

static
void printCSRel(CSset *freqCSset, CSrel *csRelMergeFreqSet, int freqThreshold){
	FILE 	*fout2,*fout2filter;
	char 	filename2[100];
	char 	tmpStr[20];
	str 	propStr;
	int		i,j, k;
	int		freq;
	int	*mfreqIdxTblIdxMapping;

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

	fclose(fout2);
	fclose(fout2filter);
	free(mfreqIdxTblIdxMapping);

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
	addHighRefCSsToFreqCS(csBats->pOffsetBat, csBats->freqBat, csBats->coverageBat, csBats->fullPBat, refCount, freqCSset, csIdFreqIdxMap, *maxCSoid + 1, 2* (*freqThreshold)); 
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

	/* ---------- S1, S2 ------- */
	mergecsId = *maxCSoid + 1; 

	mergeMaxFreqCSByS1(freqCSset, *labels, &mergecsId); /*S1: Merge all freqCS's sharing top-3 candidates */
	
	curNumMergeCS = countNumberMergeCS(freqCSset);

	curT = clock(); 
	printf("Merging with S1 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	tmpLastT = curT;
	
	/* ---------- S4 ------- */
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	/*S4: Merge two CS's having the subset-superset relationship */
	mergeCSbyS4(freqCSset, *labels, mergeCSFreqCSMap,curNumMergeCS); 

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf("Merging with S4 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	printf("Number of added CS after S4: %d \n", freqCSset->numCSadded);
	tmpLastT = curT; 		
	
	/* ---------- S6 ------- */
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);
	
	*csRelMergeFreqSet = generateCsRelBetweenMergeFreqSet(csrelSet, freqCSset);

	/* S6: Merged CS referred from the same CS via the same property */
	mergeMaxFreqCSByS6(*csRelMergeFreqSet, freqCSset, mergeCSFreqCSMap, curNumMergeCS,  &mergecsId);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf("Merging with S6 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	tmpLastT = curT; 		
	
	/* S3, S5 */
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	mergeCSByS3S5(freqCSset, *labels, mergeCSFreqCSMap, curNumMergeCS, &mergecsId, ontoUsageTree);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf ("Merging with S3, S5 took %f. (Number of mergeCS: %d) \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS);	
	tmpLastT = curT; 		

	updateParentIdxAll(freqCSset); 
	//Finally, re-create mergeFreqSet
	
	freeCSrelSet(*csRelMergeFreqSet,freqCSset->numOrigFreqCS);
	*csRelMergeFreqSet = generateCsRelBetweenMergeFreqSet(csrelSet, freqCSset);
	printCSRel(freqCSset, *csRelMergeFreqSet, *freqThreshold);

	printmergeCSSet(freqCSset, *freqThreshold);
	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csBats->pOffsetBat, csBats->freqBat, csBats->coverageBat, csBats->fullPBat, 1, *freqThreshold);
	getStatisticMaxCSs(freqCSset, 1, *freqThreshold);



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
void initCStables(CStableStat* cstablestat, CSset* freqCSset, CSPropTypes *csPropTypes, int numTables){

	int 		i,j; 
	int		tmpNumDefaultCol; 
	int		tmpNumExCol; 		/*For columns of non-default types*/
	char* 		mapObjBATtypes;
	int		colExIdx, t; 

	mapObjBATtypes = (char*) malloc(sizeof(char) * (MULTIVALUES + 1)); 
	mapObjBATtypes[URI] = TYPE_oid; 
	mapObjBATtypes[DATETIME] = TYPE_str;
	mapObjBATtypes[INTEGER] = TYPE_int; 
	mapObjBATtypes[FLOAT] = TYPE_flt; 
	mapObjBATtypes[STRING] = TYPE_str; 
	mapObjBATtypes[BLANKNODE] = TYPE_oid;
	mapObjBATtypes[MULTIVALUES] = TYPE_oid;


	// allocate memory space for cstablestat
	cstablestat->numTables = numTables; 
	cstablestat->lstbatid = (bat**) malloc(sizeof (bat*) * numTables); 
	cstablestat->numPropPerTable = (int*) malloc(sizeof (int) * numTables); 

	cstablestat->pbat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	cstablestat->sbat = BATnew(TYPE_void, TYPE_oid, smallbatsz);
	cstablestat->obat = BATnew(TYPE_void, TYPE_oid, smallbatsz);

	cstablestat->lastInsertedS = (oid**) malloc(sizeof(oid*) * numTables);
	cstablestat->lstcstable = (CStable*) malloc(sizeof(CStable) * numTables); 

	#if CSTYPE_TABLE == 1
	cstablestat->lastInsertedSEx = (oid**) malloc(sizeof(oid*) * numTables);
	cstablestat->lstcstableEx = (CStableEx*) malloc(sizeof(CStableEx) * numTables);
	#endif

	for (i = 0; i < numTables; i++){
		tmpNumDefaultCol = csPropTypes[i].numProp; 
		cstablestat->numPropPerTable[i] = tmpNumDefaultCol; 
		cstablestat->lstbatid[i] = (bat*) malloc (sizeof(bat) * tmpNumDefaultCol);  
		cstablestat->lastInsertedS[i] = (oid*) malloc(sizeof(oid) * tmpNumDefaultCol); 
		cstablestat->lstcstable[i].numCol = tmpNumDefaultCol;
		cstablestat->lstcstable[i].colBats = (BAT**)malloc(sizeof(BAT*) * tmpNumDefaultCol); 
		cstablestat->lstcstable[i].mvBats = (BAT**)malloc(sizeof(BAT*) * tmpNumDefaultCol); 
		cstablestat->lstcstable[i].lstProp = (oid*)malloc(sizeof(oid) * tmpNumDefaultCol);
		cstablestat->lstcstable[i].colTypes = (ObjectType *)malloc(sizeof(ObjectType) * tmpNumDefaultCol);
		#if CSTYPE_TABLE == 1
		tmpNumExCol = csPropTypes[i].numNonDefTypes; 
		cstablestat->lastInsertedSEx[i] = (oid*) malloc(sizeof(oid) * tmpNumExCol); 
		cstablestat->lstcstableEx[i].numCol = tmpNumExCol;
		cstablestat->lstcstableEx[i].colBats = (BAT**)malloc(sizeof(BAT*) * tmpNumExCol); 
		#endif

		for(j = 0; j < tmpNumDefaultCol; j++){

			cstablestat->lstcstable[i].colBats[j] = BATnew(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], smallbatsz);
			cstablestat->lstcstable[i].mvBats[j] = BATnew(TYPE_void, TYPE_oid, smallbatsz);
			cstablestat->lstcstable[i].lstProp[j] = freqCSset->items[csPropTypes[i].freqCSId].lstProp[j];
			//TODO: use exact size for each BAT
		}

		#if CSTYPE_TABLE == 1
		colExIdx = 0; 
		for(j = 0; j < csPropTypes[i].numProp; j++){
			for (t = 0; t < csPropTypes[i].lstPropTypes[j].numType; t++){
				if ( csPropTypes[i].lstPropTypes[j].TableTypes[t] == TYPETBL){
					cstablestat->lstcstableEx[i].colBats[colExIdx] = BATnew(TYPE_void, mapObjBATtypes[t], smallbatsz);
					colExIdx++;
				}
			}
		}

		assert(colExIdx == csPropTypes[i].numNonDefTypes);

		#endif

	}

	free(mapObjBATtypes);
}


static
void initCSTableIdxMapping(CSset* freqCSset, int* csTblIdxMapping, int* mfreqIdxTblIdxMapping, int* mTblIdxFreqIdxMapping, int *numTables){

int 		i, k; 
CS 		cs;
	int		tmpParentidx; 

	k = 0; 
	for (i = 0; i < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx == -1){	// Only use the maximum or merge CS 
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
	int i,j; 

	for (i = 0; i < cstablestat->numTables; i++){
		free(cstablestat->lstbatid[i]); 
		free(cstablestat->lastInsertedS[i]); 
		#if CSTYPE_TABLE == 1
		free(cstablestat->lastInsertedSEx[i]); 	
		#endif
		for (j = 0; j < cstablestat->numPropPerTable[i];j++){
			BBPunfix(cstablestat->lstcstable[i].colBats[j]->batCacheid); 
			BBPunfix(cstablestat->lstcstable[i].mvBats[j]->batCacheid); 

		}

		#if CSTYPE_TABLE == 1
		for (j = 0; j < cstablestat->lstcstableEx[i].numCol;j++){
			BBPunfix(cstablestat->lstcstableEx[i].colBats[j]->batCacheid); 
		}
		#endif
		free(cstablestat->lstcstable[i].colBats);
		free(cstablestat->lstcstable[i].mvBats);
		free(cstablestat->lstcstable[i].lstProp);
		#if CSTYPE_TABLE == 1
		free(cstablestat->lstcstableEx[i].colBats);
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
void fillMissingvalues(BAT* curBat, oid from, oid to){
	oid k; 
	//Insert nil values to the last column if it does not have the same
	//size as the table
	printf("Fill from  " BUNFMT " to " BUNFMT " \n", from, to);
	if (curBat != NULL){
		for(k = from -1; k < to; k++){
			BUNappend(curBat, ATOMnilptr(curBat->ttype), TRUE);
		}
	}
}

static 
void fillMissingvaluesAll(CStableStat* cstablestat, CSPropTypes *csPropTypes, int lasttblIdx, int lastColIdx, oid* lastSubjId){
	BAT     *tmpBat = NULL;
	int i; 
	int tmpColExIdx; 

	printf("Fill for Table %d and prop %d \n", lasttblIdx, lastColIdx);

	tmpBat = cstablestat->lstcstable[lasttblIdx].colBats[lastColIdx];	
	fillMissingvalues(tmpBat, BATcount(tmpBat), lastSubjId[lasttblIdx]); 
	for (i = 0; i < (MULTIVALUES + 1); i++){
		if (csPropTypes[lasttblIdx].lstPropTypes[lastColIdx].TableTypes[i] == TYPETBL){
			tmpColExIdx = csPropTypes[lasttblIdx].lstPropTypes[lastColIdx].colIdxes[i]; 
			tmpBat = cstablestat->lstcstableEx[lasttblIdx].colBats[tmpColExIdx];
			fillMissingvalues(tmpBat, BATcount(tmpBat), lastSubjId[lasttblIdx]);
		}
		
	}
}

str RDFdistTriplesToCSs(int *ret, bat *sbatid, bat *pbatid, bat *obatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId){
	
	BAT *sbat = NULL, *pbat = NULL, *obat = NULL; 
	BATiter si,pi,oi; 
	BUN p,q; 
	oid *pbt, *sbt, *obt;
	oid lastP, lastS; 
	int	tblIdx = -1; 
	oid	tmpSoid = BUN_NONE; 
	BUN	ppos; 
	int*	tmpTblIdxPropIdxMap;	//For each property, this maps the table Idx (in the posting list
					// of that property to the position of that property in the
					// list of that table's properties
	Postinglist tmpPtl; 
	int	tmpColIdx = -1; 
	int	tmpColExIdx = -1; 
	int	lasttblIdx = -1; 
	int	lastColIdx = -1; 
	char	objType; 
	char	tmpTableType = 0;

	int	i,j; 
	BAT	*curBat = NULL;
	BAT	*tmpBat = NULL; 
	BAT     *tmpmvBat = NULL;       // Multi-values BAT
	oid	*tmplastInsertedS; 
	int     numMutiValues = 0;
	oid	*lastDupValue; 
	oid	tmpmvValue; 


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

	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat);
	
	tmpTblIdxPropIdxMap = (int*)malloc(sizeof(int) * cstablestat->numTables);
	initIntArray(tmpTblIdxPropIdxMap, cstablestat->numTables, -1); 

	tmplastInsertedS = (oid*)malloc(sizeof(oid) * (MULTIVALUES + 1));
	initArray(tmplastInsertedS, (MULTIVALUES + 1), 0); 
	

	lastP = BUN_NONE; 
	lastS = BUN_NONE; 
	
	printf("Reorganize the triple store by using %d CS tables \n", cstablestat->numTables);

	//setofBats = (BAT**)malloc(sizeof(BAT*) * cstablestat->numTables); 

	BATloop(pbat, p, q){
		pbt = (oid *) BUNtloc(pi, p);
		sbt = (oid *) BUNtloc(si, p);
		obt = (oid *) BUNtloc(oi, p);
		
		printf(BUNFMT ": " BUNFMT "  |  " BUNFMT " | " BUNFMT , p, *pbt, *sbt, *obt); 
		getTblIdxFromS(*sbt, &tblIdx, &tmpSoid);	
		printf("  --> Tbl: %d  tmpSoid: " BUNFMT, tblIdx,tmpSoid);


		if (tblIdx == -1){	// This is for irregular triples, put them to pso table
			BUNappend(cstablestat->pbat,pbt , TRUE);
			BUNappend(cstablestat->sbat,sbt , TRUE);
			BUNappend(cstablestat->obat,obt , TRUE);
			printf(" ==> To PSO \n");

			continue; 
		}

		if (*pbt != lastP){
			//Get number of BATs for this p
			ppos = BUNfnd(BATmirror(propStat->pBat),pbt);
			if (ppos == BUN_NONE)
				throw(RDF, "rdf.RDFdistTriplesToCSs", "This prop must be in propStat bat");

			tmpPtl =  propStat->plCSidx[ppos];
			updateTblIdxPropIdxMap(tmpTblIdxPropIdxMap, 
					tmpPtl.lstIdx, tmpPtl.lstInvertIdx,tmpPtl.numAdded);
			//init set of BATs containing this property
			//
			//if (creatPBats(setofBats, propStat->plCSidx[ppos], TYPE_void, TYPE_oid) != MAL_SUCCEED){
			//	throw(RDF, "rdf.RDFdistTriplesToCSs", "Problem in creating set of bats for a P");
			//}
			
			lastP = *pbt; 
			lastS = *sbt; 
			numMutiValues = 0;

		}
		else{
			if (*sbt == lastS){ 	//multi-values prop
				printf("Multi values prop \n"); 
				//printf("Multivalue at table %d col %d \n", tblIdx,tmpColIdx);
				if (numMutiValues == 0){ 	// The first duplication 
					// Insert the last value from curBat to mvBat, then update this value to null
					// Add a value to MULVALUE column in TableEx
					// pointing to the offset of mul
					lastDupValue = (oid *)Tloc(curBat, BUNlast(curBat) -1);
					tmpmvValue = *lastDupValue; 
					BUNappend(tmpmvBat, &tmpmvValue, TRUE); 

					*lastDupValue = oid_nil; 	
					//*lastDupValue = BUNlast(tmpmvBat) - 1; 
					//*lastDupValue |= (BUN)MULTIVALUES << (sizeof(BUN)*8 - 4);
					
					// Add the current object to mvBat
					BUNappend(tmpmvBat, obt, TRUE);			
					
					// For the MULTIVALUE column in TableEx
					tmpColExIdx = csPropTypes[tblIdx].lstPropTypes[tmpColIdx].colIdxes[MULTIVALUES];
					tmpBat = cstablestat->lstcstableEx[tblIdx].colBats[tmpColExIdx];
					if (tmpSoid > (tmplastInsertedS[MULTIVALUES] + 1)){
						fillMissingvalues(tmpBat, tmplastInsertedS[MULTIVALUES] + 1, tmpSoid-1);
					}
					tmpmvValue = BUNlast(tmpmvBat) - 1;
					BUNappend(tmpBat,&tmpmvValue, TRUE);

					numMutiValues++;

				}
				else{
					// Add the current object to mvBat
					BUNappend(tmpmvBat, obt, TRUE);			
					numMutiValues++;
				}
				continue; 				
			}
			else{
				lastS = *sbt; 	
				numMutiValues = 0;

			}
		}

		objType = getObjType(*obt); 

		tmpColIdx = tmpTblIdxPropIdxMap[tblIdx]; 

		tmpTableType = csPropTypes[tblIdx].lstPropTypes[tmpColIdx].TableTypes[(int)objType]; 

		printf("  objType: %d  TblType: %d", (int)objType,(int)tmpTableType);
		if (tmpTableType == PSOTBL){			//For infrequent type ---> go to PSO
			BUNappend(cstablestat->pbat,pbt , TRUE);
			BUNappend(cstablestat->sbat,sbt , TRUE);
			BUNappend(cstablestat->obat,obt , TRUE);
			printf(" ==> To PSO \n");
			continue; 
		}

		if (p == 0){
			lastColIdx = tmpColIdx;
			lasttblIdx = tblIdx;
		}

		/* New column. Finish with lastTblIdx and lastColIdx */
		if (tmpColIdx != lastColIdx || lasttblIdx != tblIdx){ 
			//Insert missing values for all columns of this property in this table

			fillMissingvaluesAll(cstablestat, csPropTypes, lasttblIdx, lastColIdx, lastSubjId);

			lastColIdx = tmpColIdx; 
			lasttblIdx = tblIdx;
			initArray(tmplastInsertedS, (MULTIVALUES + 1), 0);
			
		}
		
		if (tmpTableType == MAINTBL){
			curBat = cstablestat->lstcstable[tblIdx].colBats[tmpColIdx];
			tmplastInsertedS[(int)objType] = cstablestat->lastInsertedS[tblIdx][tmpColIdx];
			printf(" tmpColIdx = %d \n",tmpColIdx);
		}
		else{	//tmpTableType == TYPETBL
			tmpColExIdx = csPropTypes[tblIdx].lstPropTypes[tmpColIdx].colIdxes[(int)objType];
			curBat = cstablestat->lstcstableEx[tblIdx].colBats[tmpColExIdx];
			tmplastInsertedS[(int)objType] = cstablestat->lastInsertedSEx[tblIdx][tmpColExIdx];
			printf(" tmpColIdx = %d \n",tmpColExIdx);
		}

		tmpmvBat = cstablestat->lstcstable[tblIdx].mvBats[tmpColIdx];

		//TODO: Check last subjectId for this prop. If the subjectId is not continuous, insert NIL
		if (tmpSoid > (tmplastInsertedS[(int)objType] + 1)){
			printf("Fill begin from tmplastInsertedS[%d] = "BUNFMT" to " BUNFMT "\n",  (int)objType, tmplastInsertedS[(int)objType],tmpSoid-1);
			fillMissingvalues(curBat, tmplastInsertedS[(int)objType] + 1, tmpSoid-1);
		}

		BUNappend(curBat, obt, TRUE); 

		//printf(BUNFMT": Table %d | column %d  for prop " BUNFMT " | sub " BUNFMT " | obj " BUNFMT "\n",p, tblIdx, 
		//					tmpColIdx, *pbt, tmpSoid, *obt); 
					
		//Update last inserted S
		if (tmpTableType == MAINTBL){
			cstablestat->lastInsertedS[tblIdx][tmpColIdx] = tmpSoid;
		}
		else{		//tmpTableType == TYPETBL
			cstablestat->lastInsertedSEx[tblIdx][tmpColExIdx] = tmpSoid;
		}
	}

	//HAVE TO GO THROUGH ALL BATS
	fillMissingvaluesAll(cstablestat, csPropTypes, lasttblIdx, lastColIdx, lastSubjId);

	// Keep the batCacheId
	for (i = 0; i < cstablestat->numTables; i++){
		for (j = 0; j < cstablestat->numPropPerTable[i];j++){
			cstablestat->lstbatid[i][j] = cstablestat->lstcstable[i].colBats[j]->batCacheid; 
		}
	}

	*ret = 1; 

	printf(" ... Done \n");
	
	BBPunfix(sbat->batCacheid);
	BBPunfix(pbat->batCacheid);
	BBPunfix(obat->batCacheid);
	free(tmpTblIdxPropIdxMap); 

	return MAL_SUCCEED; 
}

str
RDFreorganize(int *ret, CStableStat *cstablestat, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold, int *mode){

	CSset		*freqCSset; 	/* Set of frequent CSs */
	oid		*subjCSMap = NULL;  	/* Store the corresponding CS Id for each subject */
	oid 		maxCSoid = 0; 
	BAT		*sbat = NULL, *obat = NULL, *pbat = NULL;
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
	CSlabel		*labels, *labels2;
	CSrel		*csRelMergeFreqSet = NULL;

	freqCSset = initCSset();

	if (RDFextractCSwithTypes(ret, sbatid, pbatid, obatid, mapbatid, freqThreshold, freqCSset,&subjCSMap, &maxCSoid, &maxNumPwithDup, &labels, &csRelMergeFreqSet) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in extracting CSs");
	}
	
	printf("Start re-organizing triple store for " BUNFMT " CSs \n", maxCSoid);

	csTblIdxMapping = (int *) malloc (sizeof (int) * (maxCSoid + 1)); 
	initIntArray(csTblIdxMapping, (maxCSoid + 1), -1);

	mfreqIdxTblIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded); 
	initIntArray(mfreqIdxTblIdxMapping , freqCSset->numCSadded, -1);

	mTblIdxFreqIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded);  // A little bit reduntdant space
	initIntArray(mTblIdxFreqIdxMapping , freqCSset->numCSadded, -1);

	//Mapping from from CSId to TableIdx 
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
	csPropTypes = initCSPropTypes(freqCSset, numTables);
	RDFExtractCSPropTypes(ret, sbat, si, pi, oi, subjCSMap, csTblIdxMapping, csPropTypes, maxNumPwithDup);
	genCSPropTypesColIdx(csPropTypes, numTables, freqCSset);
	printCSPropTypes(csPropTypes, numTables, freqCSset, *freqThreshold);
	
	#if COLORINGPROP
	/* Update list of support for properties in freqCSset */
	updatePropSupport(csPropTypes, numTables, freqCSset);
	#endif

	// final labeling
	labels2 = createFinalLabels(labels, freqCSset, csRelMergeFreqSet, *freqThreshold);
	(void) labels2; // TODO use!

	// Init CStableStat
	initCStables(cstablestat, freqCSset, csPropTypes, numTables);

	if (*mode == EXPLOREONLY){
		printf("Only explore the schema information \n");
		freeFinalLabels(labels2, freqCSset);
		freeLabels(labels, freqCSset);
		freeCSrelSet(csRelMergeFreqSet,freqCSset->numCSadded);
		freeCSset(freqCSset); 
		free(subjCSMap);
		free(csTblIdxMapping);
		free(mfreqIdxTblIdxMapping);
		free(mTblIdxFreqIdxMapping);
		freeCSPropTypes(csPropTypes,numTables);

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

	}


	//BATprint(VIEWcreate(BATmirror(lmap),rmap)); 
	
	origobat = getOriginalOBat(obat); 

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

	propStat = getPropStatisticsByTable(freqCSset,mfreqIdxTblIdxMapping, &numdistinctMCS); 
	
	//printPropStat(propStat,0); 
	
	if (RDFdistTriplesToCSs(ret, &sNewBat->batCacheid, &pNewBat->batCacheid, &oNewBat->batCacheid, propStat, cstablestat, csPropTypes, lastSubjId) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in distributing triples to BATs using CSs");		
	}
		

	freeCSrelSet(csRelMergeFreqSet,freqCSset->numCSadded);
	freeCSPropTypes(csPropTypes,numTables);
	freeFinalLabels(labels2, freqCSset);
	freeLabels(labels, freqCSset);
	freeCSset(freqCSset); 
	free(subjCSMap); 
	free(csTblIdxMapping);
	
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
