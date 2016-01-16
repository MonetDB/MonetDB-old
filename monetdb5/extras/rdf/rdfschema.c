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
#include "rdfontologyload.h"
#include "rdfdump.h"
#include "rdfcommon.h"
#include <mtime.h>
#include <rdfgraph.h>
#include <rdfparams.h>
#include "bat5.h"

#define SHOWPROPERTYNAME 1


// for storing ontology data
oid	**ontattributes = NULL;
int	ontattributesCount = 0;
oid	**ontmetadata = NULL;
int	ontmetadataCount = 0;
BAT	*ontmetaBat = NULL;
OntClass  *ontclassSet = NULL; 
int	totalNumberOfTriples = 0;
int	acceptableTableSize = 0;

str
RDFSchemaExplore(int *ret, str *tbname, str *clname)
{
	printf("Explore from table %s with colum %s \n", *tbname, *clname);
	*ret = 1; 
	return MAL_SUCCEED;
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

#endif /* if NEEDSUBCS */



static void initcsIdFreqIdxMap(int* inputArr, int num, int defaultValue, CSset *freqCSset){
	int i; 
	for (i = 0; i < num; i++){
		inputArr[i] = defaultValue;
	}

	for (i = 0; i < freqCSset->numCSadded; i++){
		inputArr[freqCSset->items[i].csId] = i; 
	}

}


ObjectType
getObjType(oid objOid){
	ObjectType objType = (ObjectType) ((objOid >> (sizeof(BUN)*8 - 4))  &  7) ;

	return objType; 

}

str printTKNZStringFromOid(oid id){
	int ret; 
	char*   schema = "rdf";
	str propStr; 

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

	takeOid(id, &propStr);	
	printf("String for "BUNFMT": %s\n", id, propStr);
	
	GDKfree(propStr); 
	TKNZRclose(&ret);

	return MAL_SUCCEED; 
}

#if USE_ONTLABEL_FOR_NAME
static 
char isOntologyName(oid valueOid, BUN *ontClassPos){
	*ontClassPos = BUN_NONE; 
	*ontClassPos = BUNfnd(ontmetaBat, &valueOid);
	if (*ontClassPos == BUN_NONE) return 0; 
	else return 1; 
}
#endif

//Get the string for 
static
char getStringName(oid objOid, str *objStr, BATiter mapi, BAT *mapbat, char isTblName){
	
	ObjectType	objType = getObjType(objOid); 
	oid	realObjOid; 
	BUN	bun;
	int 	i = 0;
	char	hasOntologyLabel = 0; 

	#if USE_ONTLABEL_FOR_NAME
	if (isTblName){
		char 	isOntName = 0; 
		BUN	tmpontClassPos = BUN_NONE; 

		isOntName = isOntologyName(objOid, &tmpontClassPos);	

		if (isOntName == 1){
			//Check if label is availabel 
			if (ontclassSet[tmpontClassPos].label != NULL){	//Use this label
				*objStr =  GDKstrdup(ontclassSet[tmpontClassPos].label);
				hasOntologyLabel = 1; 
			} 
		}
	}
	#endif

	if (hasOntologyLabel == 0){
		if (objType == URI || objType == BLANKNODE){
			realObjOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
			takeOid(realObjOid, objStr); 
		}
		else if (objType == STRING){
			str tmpObjStr;
			str s;
			int len; 
			realObjOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */ 
			bun = BUNfirst(mapbat);
			tmpObjStr = (str) BUNtail(mapi, bun + realObjOid); 
			
			*objStr = GDKstrdup(tmpObjStr);
					
			if (isTblName){
				s = *objStr;
				len = strlen(s);
				//Replace all non-alphabet character by ___
				for (i = 0; i < len; i++)
				{	
					//printf("i = %d: %c \n",i, s[i]);
					if (!isalpha(*s)){
						*s = '_';
					}
					s++;
					
				}
			}

		}
		else{
			getStringFormatValueFromOid(objOid, objType, objStr);	
		}
	}
	
	return objType;
}


char isCSTable(CS item, oid name){
	(void) name; 
	if (item.parentFreqIdx != -1) return 0; 

	if (item.type == DIMENSIONCS) return 1; 

	#if REMOVE_SMALL_TABLE
	if (item.support > acceptableTableSize) return 1;

	//if (item.coverage < minTableSize) return 0;
	//More strict with table which does not have name
	//if ((name == BUN_NONE) && item.support < minTableSize) return 0; 
	if (item.support < minTableSize) return 0; 
	#endif	

	return 1; 
}

static 
void addCStoSet(CSset *csSet, CS item)
{
	void *_tmp; 
	if(csSet->numCSadded == csSet->numAllocation) 
	{ 
		csSet->numAllocation *= 2; 
		
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
			//csrel->numAllocation += INIT_NUM_CSREL; 
			csrel->numAllocation = csrel->numAllocation * 2;
			
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

#if NO_OUTPUTFILE == 0
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
#endif

static 
void getOrigRefCount(CSrel *csrelSet, CSset *freqCSset, int num,  int* refCount){

	int 	i, j; 
	int	freqId; 

	for (i = 0; i < num; i++){
		if (csrelSet[i].numRef != 0){	
			for (j = 0; j < csrelSet[i].numRef; j++){
				freqId = csrelSet[i].lstRefFreqIdx[j]; 
				#if FILTER_INFREQ_FK_FOR_IR
				if (csrelSet[i].lstCnt[j] < infreqTypeThreshold * freqCSset->items[freqId].support) continue; 
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
					if (csrelSet[i].lstCnt[j] < infreqTypeThreshold * freqCSset->items[freqId].support) continue; 
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
			printf("IR score[%d] is %f (support: %d)\n", i, curIRScores[i],freqCSset->items[i].support);
		}
		*/
	}

	free(lastIRScores);
}


static 
void updateFreqCStype(CSset *freqCSset, int num,  float *curIRScores, int *refCount){

	int 	i; 
	int	numDimensionCS = 0; 
	int	threshold = 0; 
	int 	ratio; 

	//ratio = pow(IR_DIMENSION_FACTOR, nIterIR);
	ratio = IR_DIMENSION_FACTOR;

	printf("List of dimension tables: \n");
	for (i = 0; i < num; i++){
		#if ONLY_SMALLTBL_DIMENSIONTBL
		if (freqCSset->items[i].support > minTableSize) continue; 
		#endif
		if (refCount[i] < freqCSset->items[i].support) continue; 
		threshold = freqCSset->items[i].support * ratio;
		if (curIRScores[i] < threshold) continue; 
		
		freqCSset->items[i].type = DIMENSIONCS;
		//printf("A dimension CS with IR score = %f \n", curIRScores[i]);
		printf(" %d  (Ratio %.2f)", i, (float) curIRScores[i]/freqCSset->items[i].support);
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

#if NO_OUTPUTFILE == 0 
static 
void printSubCSInformation(SubCSSet *subcsset, BAT* freqBat, int num, char isWriteTofile, int freqThreshold){

	int i; 
	int j; 
	int *freq; 
	int numSubCSFilter; 
	int totalNumSubCS = 0;

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
				totalNumSubCS += subcsset[i].numSubCS;
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

		printf("Avg. number of subCSs per CS: %f \n", (float) totalNumSubCS / num);
	}
}
#endif /*NO_OUTPUTFILE*/
#endif  /* NEEDSUBCS */






/*
 * Init property types for each CS in FreqCSset (after merging)
 * For each property, init with all possible types (MULTIVALUES + 1))
 * 
 * */
static 
void initCSPropTypes(CSPropTypes* csPropTypes, CSset* freqCSset, int numMergedCS, CSlabel *labels){
	int numFreqCS = freqCSset->numCSadded;
	int i, j, k ;
	int id; 
	
	id = 0; 
	for (i = 0; i < numFreqCS; i++){
		if ( isCSTable(freqCSset->items[i], labels[i].name)){   // Only use the maximum or merge CS		
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

#ifndef NDEBUG
	assert(id == numMergedCS);
#else
	(void) numMergedCS;		
#endif

	//return csPropTypes;
}

#if COUNT_NUMTYPES_PERPROP

static 
void initCSPropTypesForBasicFreqCS(CSPropTypes* csPropTypes, CSset* freqCSset, int numMergedCS){
	int numFreqCS = freqCSset->numCSadded;
	int i, j, k ;
	int id; 
	
	id = 0; 
	for (i = 0; i < numFreqCS; i++){
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

#ifndef NDEBUG
	assert(id == numMergedCS);
#else
	(void) numMergedCS;		
#endif

	//return csPropTypes;
}
#endif

static 
char isMultiValueCol(PropTypes pt){
	double tmpRatio;

	tmpRatio = ((double)pt.propCover / (pt.numSingleType + pt.numMVType));
	//printf("NumMVType = %d  | Ratio %f \n", pt.numMVType, tmpRatio);
	if ((pt.numMVType > 0) && (tmpRatio > (1 + infreqTypeThreshold))){
		return 1; 
	}
	else return 0; 
}

static
char isInfrequentProp(PropTypes pt, CS cs){
	if (pt.propFreq < cs.support * infreqPropThreshold) return 1; 
	else return 0;

}

#if NO_OUTPUTFILE == 0 
static
char isInfrequentSampleCol(CS freqCS, PropTypes pt){
	if (pt.propFreq * 100 <  freqCS.support * SAMPLE_FILTER_THRESHOLD) return 1;
	else return 0; 
}
#endif

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
				csPropTypes[i].lstPropTypes[j].defaultType = (ObjectType)defaultIdx;
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
					if (csPropTypes[i].lstPropTypes[j].lstFreq[k] < csPropTypes[i].lstPropTypes[j].propFreq * infreqTypeThreshold){
						//non-frequent type goes to PSO
						csPropTypes[i].lstPropTypes[j].TableTypes[k] = PSOTBL; 
					}
					else
						csPropTypes[i].lstPropTypes[j].TableTypes[k] =TYPETBL;
				}
				/* One type is set to be the default type (in the main table) */
				csPropTypes[i].lstPropTypes[j].TableTypes[defaultIdx] = MAINTBL; 
				csPropTypes[i].lstPropTypes[j].colIdxes[defaultIdx] = curDefaultColIdx;
				csPropTypes[i].lstPropTypes[j].defaultType = (ObjectType)defaultIdx; 
				
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

#if NO_OUTPUTFILE == 0
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
		fprintf(fout, "MergedCS %d (Freq: %d ) \n", i, freqCSset->items[csPropTypes[i].freqCSId].support);
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
					csPropTypes[i].lstPropTypes[j].prop, (int) (csPropTypes[i].lstPropTypes[j].defaultType),
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
#endif

#if NO_OUTPUTFILE == 0
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
#endif

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

	//printf("Max number of prop among %d merged CS is: %d \n", num, maxNumProp);

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
		#if EXTRAINFO_FROM_RDFTYPE
		if (csSet->items[i].typevalues != NULL) 
			free(csSet->items[i].typevalues); 
		#endif
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
		#if EXTRAINFO_FROM_RDFTYPE

		#endif
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
	#if STORE_PERFORMANCE_METRIC_INFO
	csSet->totalInRef = 0;
	#endif
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
	#if EXTRAINFO_FROM_RDFTYPE
	cs->typevalues = NULL; 
	cs->numTypeValues = 0;
	#endif

	#if STORE_PERFORMANCE_METRIC_INFO
	cs->numInRef = 0;
	cs->numFill = 0;
	#endif

	return cs; 
}


static 
CS* mergeTwoCSs(CS cs1, CS cs2, int freqIdx1, int freqIdx2, oid mergeCSId){
	
	int numCombineP = 0; 

	CS *mergecs = (CS*) malloc (sizeof (CS)); 
	if (cs1.type == DIMENSIONCS || cs2.type == DIMENSIONCS)
		 mergecs->type = DIMENSIONCS; 
	else
		mergecs->type = (char)MERGECS; 

	mergecs->numConsistsOf = 2; 
	mergecs->lstConsistsOf = (int*) malloc(sizeof(int) * 2);

	//mergecs->lstConsistsOf[0] = cs1->csId;  
	//mergecs->lstConsistsOf[1] = cs2->csId; 

	mergecs->lstConsistsOf[0] = freqIdx1;  
	mergecs->lstConsistsOf[1] = freqIdx2; 
	#if EXTRAINFO_FROM_RDFTYPE
	mergecs->typevalues = NULL; 
	mergecs->numTypeValues = 0;
	#endif
	
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
	
	#if STORE_PERFORMANCE_METRIC_INFO
	mergecs->numInRef = cs1.numInRef + cs2.numInRef;
	mergecs->numFill = cs1.numFill + cs2.numFill;
	#endif
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
	
	#if EXTRAINFO_FROM_RDFTYPE
	mergecs->typevalues = NULL; 
	mergecs->numTypeValues = 0;
	#endif

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
	
	#if STORE_PERFORMANCE_METRIC_INFO
	mergecs->numInRef += cs.numInRef;
	mergecs->numFill += cs.numFill;
	#endif

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

	#if STORE_PERFORMANCE_METRIC_INFO
	mergecs1->numInRef += mergecs2->numInRef;
	mergecs1->numFill += mergecs2->numFill;
	#endif

	#if EXTRAINFO_FROM_RDFTYPE
	mergecs1->typevalues = NULL; 
	mergecs1->numTypeValues = 0;
	#endif

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

#if NO_OUTPUTFILE == 0
static 
int getOntologyIndex(BAT *ontbat, oid prop){
	str 	propStr; 
	char 	*ontpart;		
	char 	*ptrEndOnt;
	int	ontlen; 
	BUN	bunOnt; 

	takeOid2(prop, &propStr);

	ptrEndOnt = NULL;
	ptrEndOnt = strrchr((str)propStr, '#');
	if (ptrEndOnt == NULL) ptrEndOnt = strrchr((str)propStr, '/');
	
	if (ptrEndOnt != NULL){
		ontlen = (int) (ptrEndOnt - (str)propStr);

		ontpart = substring((char*)propStr, 1, ontlen); 

		//Check whether ontpart appear in the ontBat
		bunOnt = BUNfnd(ontbat,(ptr) (str)ontpart);	
		if (bunOnt == BUN_NONE){
			//printf("Non-ontology string: %s \n",propStr);
			GDKfree(ontpart);
			GDKfree(propStr);
			return -1;
		}
		else{
			GDKfree(ontpart);
			GDKfree(propStr);
			return (int)bunOnt; 
		}
	}
	else{
		GDKfree(propStr);
		return -1; 
	}
}

static
int getNumOntology(oid* lstProp, int numProp, BAT *ontbat, int *buffOntologyNums, int numOnt){
	int i; 
	int idx; 
	int numOntology = 0; 
	//Reset buffOntologyNums
	for (i = 0; i < (numOnt+1); i++){
		buffOntologyNums[i] = 0; 
	}
	for (i = 0; i < numProp; i++){
		idx = getOntologyIndex(ontbat, lstProp[i]); 
		if (idx != -1)
			buffOntologyNums[idx]++;
		else
			buffOntologyNums[numOnt]++;
	}
	
	for (i = 0; i < (numOnt+1); i++){
		if (buffOntologyNums[i] != 0)  numOntology++;	
	}
	
	return numOntology;
}



static 
str printMergedFreqCSSet(CSset *freqCSset, BAT *mapbat, BAT *ontbat, char isWriteTofile, int freqThreshold, CSlabel* labels, int mergingstep){

	int 	i,j; 
	int 	mergeCSid, tmpParentFreqId; 
	int 	freq; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];

	int	*buffOntologyNums; 	//Number of instances in each ontology 
	int	numOnt; 		//Number of ontology
	int	tmpNumOnt = 0; 
	int	totalNumOntology = 0; 
	int	numFreqCSWithNonOntProp = 0;

#if SHOWPROPERTYNAME
	str 	propStr; 
	#if STOREFULLCS
	str	subStr; 
	str	objStr; 
	oid 	objOid; 
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

	numOnt = BATcount(ontbat); 
	buffOntologyNums = GDKmalloc(sizeof(int) * (numOnt+1));  //The last index stores number of non-ontology instances
	for (i = 0; i < (numOnt+1); i++){
		buffOntologyNums[i] = 0;
	}

	mergeCSid = -1;
	if (isWriteTofile == 0){
		for (i = 0; i < freqCSset->numCSadded; i++){
			CS cs = (CS)freqCSset->items[i];
			if (cs.parentFreqIdx != -1) continue;
			freq = cs.support; 

			printf("CS " BUNFMT " (Freq: %d) | Parent " BUNFMT " \n", cs.csId, freq, freqCSset->items[cs.parentFreqIdx].csId);
			for (j = 0; j < cs.numProp; j++){
				printf("  P:" BUNFMT " --> \n", cs.lstProp[j]);	
			}	
			printf("\n");
		}
	}
	else{
	
		strcpy(filename, "freqCSFullInfo");
		sprintf(tmpStr, "%d_%d", freqThreshold,mergingstep);
		strcat(filename, tmpStr);
		strcat(filename, ".txt");

		fout = fopen(filename,"wt"); 

		for (i = 0; i < freqCSset->numCSadded; i++){
			CS cs = (CS)freqCSset->items[i];
			if (cs.parentFreqIdx != -1) continue; 
			mergeCSid++;	
			freq = cs.support; 

			//Get ontology stat
			tmpNumOnt = getNumOntology(cs.lstProp,cs.numProp, ontbat, buffOntologyNums, numOnt); 
			totalNumOntology += tmpNumOnt;
			if (buffOntologyNums[numOnt] != 0) numFreqCSWithNonOntProp++;

			#if STOREFULLCS	
			if (i < freqCSset->numOrigFreqCS){
				if (cs.subject != BUN_NONE){
					takeOid(cs.subject, &subStr);
					if (labels[i].name == BUN_NONE) {
						fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) | Subject: %s  | FreqParentIdx %d \n", cs.csId, i, "DUMMY", freq, cs.coverage, subStr, cs.parentFreqIdx);
					} else {
						str labelStr;
						//takeOid(labels[i].name, &labelStr);
						getStringName(labels[i].name, &labelStr, mapi, mapbat, 1);
						fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) (NameFreq: %d --> %.2f percent) | Subject: %s  | FreqParentIdx %d \n", cs.csId, i, labelStr, freq, cs.coverage,labels[i].nameFreq, (float) labels[i].nameFreq/freq * 100, subStr, cs.parentFreqIdx);
						GDKfree(labelStr); 
					}

					GDKfree(subStr);
				}
				else{
					if (labels[i].name == BUN_NONE) {
						fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) | FreqParentIdx %d \n", cs.csId, i, "DUMMY", freq, cs.coverage,cs.parentFreqIdx);
					} else {
						str labelStr;
						//takeOid(labels[i].name, &labelStr);
						getStringName(labels[i].name, &labelStr, mapi, mapbat, 1);
						fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) (NameFreq: %d --> %.2f percent) | FreqParentIdx %d \n", cs.csId, i, labelStr, freq, cs.coverage,labels[i].nameFreq, (float) labels[i].nameFreq/freq * 100, cs.parentFreqIdx);
						GDKfree(labelStr);
					}
				}
			}
			else {

				if (labels[i].name == BUN_NONE) {
					fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) | Subject: <Not available>  | FreqParentIdx %d \n", cs.csId, i, "DUMMY", freq, cs.coverage,cs.parentFreqIdx);
				} else {
					str labelStr;
					//takeOid(labels[i].name, &labelStr);
					getStringName(labels[i].name, &labelStr, mapi, mapbat, 1);	
					fprintf(fout,"CS " BUNFMT " - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) | Subject: <Not available>  | FreqParentIdx %d \n", cs.csId, i, labelStr, freq, cs.coverage,cs.parentFreqIdx);
					GDKfree(labelStr); 
				}


				fprintf(fout, "MergeCS %d (Number of parent: %d | FreqId list: ) \n",mergeCSid, cs.numConsistsOf);
				for (j = 0; j < cs.numConsistsOf; j++){
					tmpParentFreqId = cs.lstConsistsOf[j];
					fprintf(fout, " %d [F:%d]",tmpParentFreqId, freqCSset->items[tmpParentFreqId].support);
					if (labels[tmpParentFreqId].name == BUN_NONE) fprintf(fout, "[DUMMY]  ");
					else{
						str labelStr = NULL;
						str labelShortStr = NULL; 
						//takeOid(labels[tmpParentFreqId].name, &labelStr);
						getStringName(labels[tmpParentFreqId].name, &labelStr, mapi, mapbat, 1);
						getPropNameShort(&labelShortStr,labelStr);
						fprintf(fout, "[%s]  ",labelShortStr);
						GDKfree(labelShortStr);
						GDKfree(labelStr);
					}

				}
				fprintf(fout, "\n");
			}
			#endif	
			
			fprintf(fout, "Number of ontologies: %d (Number non-ontology props: %d \n",tmpNumOnt, buffOntologyNums[numOnt]);

			for (j = 0; j < cs.numProp; j++){
				takeOid(cs.lstProp[j], &propStr);
				//fprintf(fout, "  P:" BUNFMT " --> ", cs.lstProp[j]);	
				fprintf(fout, "  P(" BUNFMT "):%s --> ", cs.lstProp[j],propStr);	

				GDKfree(propStr);
				
				#if STOREFULLCS
				// Get object value
				if (i >= freqCSset->numOrigFreqCS){
					fprintf(fout, " <No Object value>  \n");
					continue; 
				}
				if (cs.lstObj != NULL){
					objOid = cs.lstObj[j]; 
					getStringName(objOid, &objStr, mapi, mapbat, 0);
					fprintf(fout, "  O: %s \n", objStr);
					GDKfree(objStr);
				}
				else{
					fprintf(fout, " <No Object value>  \n");
				}
				#endif


			}	
			fprintf(fout, "\n");
		}

		fclose(fout);
	}
	
	GDKfree(buffOntologyNums);
	
	printf("Total number of ontologies: %d (%d mergeCS) --> Average: %f ontologies/freqCS \n",totalNumOntology, mergeCSid+1, (float)totalNumOntology/(mergeCSid+1));
	printf("Number of frequent CS's having non-ontology props: %d \n", numFreqCSWithNonOntProp);

#if SHOWPROPERTYNAME
	TKNZRclose(&ret);
#endif
	
	return MAL_SUCCEED;
}


//Do not remove infrequent prop form final table
static 
str printFinalTableWithPropSupport(CSPropTypes* csPropTypes, int numTables, CSset *freqCSset, bat *mapbatid, int freqThreshold, CSlabel* labels){

	int 	i,j; 
	int 	freq; 
	int	freqId; 
	FILE 	*fout; 
	char 	filename[100];
	char 	tmpStr[20];
	BAT	*mapbat = NULL; 
	BATiter mapi; 
	str 	propStr; 
	int	ret; 
	char*   schema = "rdf";
	CS 	cs;
	
	(void) mapi;
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	
	if ((mapbat = BATdescriptor(*mapbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	mapi = bat_iterator(mapbat); 
	
	strcpy(filename, "finalfreqCSFullInfoWithPropSupport");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 

	for (i = 0; i < numTables; i++){
		freqId = csPropTypes[i].freqCSId;
		cs = (CS)freqCSset->items[freqId];
		freq = cs.support; 

		if (labels[freqId].name == BUN_NONE) {
			fprintf(fout,"Table %d - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) \n", i, freqId, "DUMMY", freq, cs.coverage);
		} else {
			str labelStr;
			
			getStringName(labels[freqId].name, &labelStr, mapi, mapbat, 1);	
			fprintf(fout,"Table %d - FreqId %d - Name: %s  (Freq: %d | Coverage: %d) \n", i, freqId, labelStr, freq, cs.coverage);
			GDKfree(labelStr); 
		}


		for (j = 0; j < cs.numProp; j++){
			takeOid(cs.lstProp[j], &propStr);
			//fprintf(fout, "  P:" BUNFMT " --> ", cs.lstProp[j]);	
			fprintf(fout, "  P(" BUNFMT ") %s | PropFreq: %d ", cs.lstProp[j],propStr, csPropTypes[i].lstPropTypes[j].propFreq);	
			if (csPropTypes[i].lstPropTypes[j].propFreq < STRANGE_PROP_FREQUENCY){
				fprintf(fout, " [REALLY INFREQUENT PROP] ");
			}
			GDKfree(propStr);
			fprintf(fout, "\n");

		}	
		fprintf(fout, "\n");
	}

	fclose(fout);

	BBPunfix(mapbat->batCacheid);
	TKNZRclose(&ret);
	
	return MAL_SUCCEED;
}

#endif /*  NO_OUTPUTFILE == 0 */

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

#if NO_OUTPUTFILE == 0
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
#endif


#if NO_OUTPUTFILE == 0 
static 
str printsubsetFromCSset(CSset *freqCSset, BAT* subsetIdxBat, BAT *mbat, int num, int* mergeCSFreqCSMap, CSlabel *label, int sampleVersion){

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
	BATiter mapi;

	mapi = bat_iterator(mbat);
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	

	strcpy(filename, "selectedSubset");
	sprintf(tmpStr, "%d_v%d", num, sampleVersion);
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
				//takeOid(label[freqIdx].candidates[j], &canStr); 
				getStringName(label[freqIdx].candidates[j], &canStr, mapi, mbat, 1);
				
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
#endif

/*
 * Hashing function for a set of values
 * Rely on djb2 http://www.cse.yorku.ca/~oz/hash.html
 *
 */
static oid RDF_hash_oidlist(oid* key, int num, int numTypeValues, oid* rdftypeOntologyValues){
	//unsigned int hashCode = 5381u; 
	oid  hashCode = 5381u;
	int i; 

	for (i = 0; i < num; i++){
		hashCode = ((hashCode << 5) + hashCode) + key[i];
	}
	
	for (i = 0; i < numTypeValues; i++){		//If no type attribute is used, numTypeValues = 0
		hashCode = ((hashCode << 5) + hashCode) + rdftypeOntologyValues[i];
	}
	// return 0x7fffffff & hashCode 
	return hashCode;
}

static 
char checkCSduplication(CSBats *csBats, BUN cskey, oid* key, int numK, int numTypeValues, oid* rdftypeOntologyValues, oid *csId){
	oid *offset, *offsetT; 
	oid *offset2, *offsetT2; 
	int numP; 
	int numT; 	//number of type values
	int i; 
	BUN *existvalue, *existTvalue; 
	BUN pos; 
	char isDuplication = 0; 

	BATiter bi = bat_iterator(csBats->hsKeyBat);
			
	HASHloop(bi, csBats->hsKeyBat->T->hash, pos, (ptr) &cskey){
		//printf("  pos: " BUNFMT, pos);

		//Get number of properties
		offset = (oid *) Tloc(csBats->pOffsetBat, pos); 
		if ((pos + 1) < csBats->pOffsetBat->batCount){
			offset2 = (oid *)Tloc(csBats->pOffsetBat, pos + 1);
			numP = *offset2 - *offset;
		}
		else{
			numP = BUNlast(csBats->fullPBat) - *offset;
		}

		//Get number of type values
		offsetT = (oid *) Tloc(csBats->typeOffsetBat, pos); 
		if ((pos + 1) < csBats->typeOffsetBat->batCount){
			offsetT2 = (oid *)Tloc(csBats->typeOffsetBat, pos + 1);
			numT = *offsetT2 - *offsetT;
		}
		else{
			numT = BUNlast(csBats->fullTypeBat) - *offsetT;
		}

		// Check each value
		if (numK != numP || numT != numTypeValues) {
			continue; 
		}
		else{
			isDuplication = 1; 
			existvalue = (oid *)Tloc(csBats->fullPBat, *offset);	
			for (i = 0; i < numP; i++){
				if (key[i] != existvalue[i]) {
					isDuplication = 0;
					break; 
				}	
			}

			existTvalue = (oid *)Tloc(csBats->fullTypeBat, *offsetT);	
			for (i = 0; i < numT; i++){
				if (rdftypeOntologyValues[i] != existTvalue[i]) {
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

	testBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
		
	for (i = 0; i < 7; i++){
		csKey = key[i]; 
		bun = BUNfnd(testBat,(ptr) &key[i]);
		if (bun == BUN_NONE) {
			if (testBat->T->hash && BATcount(testBat) > 4 * testBat->T->hash->mask) {
				HASHdestroy(testBat);
				BAThash(testBat, 2*BATcount(testBat));
			}

			BUNappend(testBat, (ptr) &csKey, TRUE);
		
		}
		else{

			printf("Input: " BUNFMT, csKey);
			printf(" --> bun: " BUNFMT "\n", bun);



			BUNappend(testBat, (ptr) &csKey, TRUE);

		}
	}
	BATprint(testBat);

	BBPreclaim(testBat); 
}
*/

void addaProp(PropStat* propStat, oid prop, int csIdx, int invertIdx){
	BUN	bun; 
	BUN	p; 

	int* _tmp1; 
	float* _tmp2; 
	Postinglist* _tmp3;
	int* _tmp4; 
	
	p = prop; 
	bun = BUNfnd(propStat->pBat,(ptr) &prop);
	if (bun == BUN_NONE) {	/* New Prop */
	       if (propStat->pBat->T->hash && BATcount(propStat->pBat) > 4 * propStat->pBat->T->hash->mask) {
			HASHdestroy(propStat->pBat);
			BAThash(propStat->pBat, 2*BATcount(propStat->pBat));
		}

		BUNappend(propStat->pBat,&p, TRUE);
		
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
		propStat->plCSidx[propStat->numAdded].lstOnt = NULL; 	//lstOnt only used in labeling


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
void addNewCS(CSBats *csBats, PropStat* fullPropStat, BUN* csKey, oid* key, oid *csoid, int num, int numTriples, int numTypeValues, oid* rdftypeOntologyValues){
	int freq = 1; 
	int coverage = numTriples; 
	BUN	offset, typeOffset; 
	#if FULL_PROP_STAT
	int	i; 
	#endif
	
	if (csBats->hsKeyBat->T->hash && BATcount(csBats->hsKeyBat) > 4 * csBats->hsKeyBat->T->hash->mask) {
		HASHdestroy(csBats->hsKeyBat);
		BAThash(csBats->hsKeyBat, 2*BATcount(csBats->hsKeyBat));
	}

	BUNappend(csBats->hsKeyBat, csKey, TRUE);
		
	(*csoid)++;
		
	offset = BUNlast(csBats->fullPBat);
	/* Add list of p to fullPBat and pOffsetBat*/
	BUNappend(csBats->pOffsetBat, &offset , TRUE);
	appendArrayToBat(csBats->fullPBat, key, num);

	typeOffset = BUNlast(csBats->fullTypeBat);
	BUNappend(csBats->typeOffsetBat, &typeOffset , TRUE);
	appendArrayToBat(csBats->fullTypeBat, rdftypeOntologyValues, numTypeValues);

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
oid putaCStoHash(CSBats *csBats, oid* key, int num, int numTriples, int numTypeValues, oid* rdftypeOntologyValues, 
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset, oid subjectId, oid* buffObjs, PropStat *fullPropStat,
		BAT *ontbat, int *buffOntologyNums, int *totalNumOntology, int numOnt)
#else
static 
oid putaCStoHash(CSBats *csBats, oid* key, int num, int numTriples, int numTypeValues, oid* rdftypeOntologyValues,
		oid *csoid, char isStoreFreqCS, int freqThreshold, CSset *freqCSset, PropStat *fullPropStat,
		BAT *ontbat, int *buffOntologyNums, int *totalNumOntology, int numOnt)
#endif	
{
	BUN 	csKey; 
	int 	*freq; 
	oid	*coverage; 	//Total number of triples coverred by this CS
	CS	*freqCS; 
	BUN	bun; 
	oid	csId; 		/* Id of the characteristic set */
	char	isDuplicate = 0; 

	(void) ontbat;
	(void) buffOntologyNums;
	(void) totalNumOntology;
	(void) numOnt;

	csKey = RDF_hash_oidlist(key, num, numTypeValues, rdftypeOntologyValues);
	bun = BUNfnd(csBats->hsKeyBat,(ptr) &csKey);
	if (bun == BUN_NONE) {
		csId = *csoid; 
		addNewCS(csBats, fullPropStat, &csKey, key, csoid, num, numTriples, numTypeValues, rdftypeOntologyValues);
	
		#if NO_OUTPUTFILE == 0
		*totalNumOntology = (*totalNumOntology) + getNumOntology(key, num, ontbat, buffOntologyNums, numOnt);
		#endif
		//if (csId == 2){
		//	printf("Extra info for cs 73 is: ");
		//	printTKNZStringFromOid(rdftypeOntologyValues[0]);
		//}

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
		isDuplicate = checkCSduplication(csBats, csKey, key, num, numTypeValues, rdftypeOntologyValues, &csId);

		if (isDuplicate == 0) {
			//printf(" No duplication (new CS) \n");	
			// New CS
			csId = *csoid;
			addNewCS(csBats, fullPropStat, &csKey, key, csoid, num, numTriples, numTypeValues, rdftypeOntologyValues);
			
			#if NO_OUTPUTFILE == 0
			*totalNumOntology = (*totalNumOntology) + getNumOntology(key, num, ontbat, buffOntologyNums, numOnt);
			#endif
			
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
 * idf(t) = log(#totalNumOfCSs / #numberCSs_containing_t)
 * Note that, some function may use #numberCSs_containing_t + 1 as it can be division 
 * by 0 if the term does not appear in any document. However, in our case, 
 * every prop must appear in at least one CS
 * tf-idf(t,d,D) = tf(t,d) * idf(t,D)
 *
 * Note that: If we use normalize tf by dividing with maximum tf 
 * in each CS, we still get the value 1. 
 * */

static 
float tfidfComp(int numContainedCSs, int totalNumCSs){
	return log((float)totalNumCSs/(numContainedCSs)); 
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
float similarityScoreTFIDF(oid* arr1, oid* arr2, int m, int n, int *numCombineP, 
		TFIDFInfo *tfidfInfos, int mergeCSId1, int mergeCSId2, char *existDiscriminatingProp){
	
	int i = 0, j = 0;
	int numOverlap = 0; 
	float sumXY = 0.0;

	i = 0;
	j = 0;
	while( i < n && j < m )
	{
		if( arr1[j] < arr2[i] ){
			j++;

		}
		else if( arr1[j] == arr2[i] )
		{
			if (tfidfInfos[mergeCSId1].lsttfidfs[j] > MIN_TFIDF_PROP_S4) *existDiscriminatingProp = 1;

			sumXY += tfidfInfos[mergeCSId1].lsttfidfs[j] * tfidfInfos[mergeCSId1].lsttfidfs[j];
			j++;
			i++;
			numOverlap++;

		}
		else if( arr1[j] > arr2[i] )
			i++;
	}

	*numCombineP = m + n - numOverlap;
	
	if (sumXY == 0) return 0; 

	return  ((float) sumXY / (tfidfInfos[mergeCSId1].totalTFIDF * tfidfInfos[mergeCSId2].totalTFIDF));
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

#if USE_LABEL_FINDING_MAXCS
/*
 *  * Return 1 if there is semantic evidence against merging the two CS's, this is the case iff the two CS's have a hierarchy and their common ancestor is too generic (support above generalityThreshold).
 *   */
static
char isEvidenceAgainstMerging(int freqId1, int freqId2, CSlabel* labels, OntoUsageNode *tree) {
	int i, j;
	int level;
	OntoUsageNode *tmpNode;

	// Get common ancestor
	int hCount1 = labels[freqId1].hierarchyCount;
	int hCount2 = labels[freqId2].hierarchyCount;
	int minCount = (hCount1 > hCount2)?hCount2:hCount1;

	if (minCount == 0) {
		// at least one CS does not have a hierarchy --> no semantic information --> no semantic evidence against merging
		return 0;
	}

	// get level where the hierarchies differ
 	for (i = 0; i < minCount; i++){
		if (labels[freqId1].hierarchy[hCount1-1-i] != labels[freqId2].hierarchy[hCount2-1-i]) break;
	}

	if (i == 0) {
		// not even the top level of the hierarchy is the same --> there is semantic evidence against merging the two CS's
		return 1;
	} else if (i == minCount) {
		// same name --> no semantic evidence against merging
		return 0;
	}

	// get the common ancestor at level i
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

	if (tmpNode->percentage >= generalityThreshold) {
		// have common ancestor but it is too generic --> there is semantic evidence against merging the two CS's
 		return 1;
	} else {
		// common ancestor is specific --> no semantic evidence against merging
		return 0;
	}
}
#endif

/*
 * Get the maximum frequent CSs from a CSset
 * Here maximum frequent CS is a CS that there exist no other CS which contains that CS
 * */
static 
void mergeCSbyS3(CSset *freqCSset, CSlabel** labels, oid *mergeCSFreqCSMap, int curNumMergeCS, oid **ontmetadata, int ontmetadataCount, OntoUsageNode *tree){

	int 	numMergeCS = curNumMergeCS; 
	int 	i, j; 
	int 	numMaxCSs = 0;

	int 	tmpParentIdx; 
	int	freqId1, freqId2; 
	int	numP1, numP2; 
	CS	*mergecs1, *mergecs2; 

#if !USE_LABEL_FINDING_MAXCS
	(void) tree;
#endif

	printf("Retrieving maximum frequent CSs: \n");

	for (i = 0; i < numMergeCS; i++){
		freqId1 = mergeCSFreqCSMap[i]; 
		if (freqCSset->items[freqId1].parentFreqIdx != -1) continue;
		#if	NOT_MERGE_DIMENSIONCS
		if (freqCSset->items[freqId1].type == DIMENSIONCS) continue; 
		#endif

		for (j = (i+1); j < numMergeCS; j++){
			freqId2 = mergeCSFreqCSMap[j];
			#if	NOT_MERGE_DIMENSIONCS
			if (freqCSset->items[freqId2].type == DIMENSIONCS) continue; 
			#endif

			numP2 = freqCSset->items[freqId2].numProp;
			numP1 = freqCSset->items[freqId1].numProp;
			if (numP2 > numP1 && (numP2-numP1)< MAX_SUB_SUPER_NUMPROP_DIF){
				if (isSubset(freqCSset->items[freqId2].lstProp, freqCSset->items[freqId1].lstProp, numP2,numP1) == 1) { 
					/* CSj is a superset of CSi */
#if USE_LABEL_FINDING_MAXCS
					if (isEvidenceAgainstMerging(freqId1, freqId2, *labels, tree)) continue;
#endif
					freqCSset->items[freqId1].parentFreqIdx = freqId2; 
					updateLabel(S3, freqCSset, labels, 0, freqId2, freqId1, freqId2, BUN_NONE, 0, 0, 0, ontmetadata, ontmetadataCount, NULL, -1); // name, isType, isOntology, isFK are not used for case CS
					break; 
				}
			}
			else if (numP2 < numP1 && (numP1-numP2)< MAX_SUB_SUPER_NUMPROP_DIF){
				if (isSubset(freqCSset->items[freqId1].lstProp, freqCSset->items[freqId2].lstProp,  
						numP1,numP2) == 1) { 
					/* CSj is a subset of CSi */
#if USE_LABEL_FINDING_MAXCS
					if (isEvidenceAgainstMerging(freqId1, freqId2, *labels, tree)) continue;
#endif
					freqCSset->items[freqId2].parentFreqIdx = freqId1; 
					updateLabel(S3, freqCSset, labels, 0, freqId1, freqId1, freqId2, BUN_NONE, 0, 0, 0, ontmetadata, ontmetadataCount, NULL, -1); // name, isType, isOntology, isFK are not used for case CS
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
			#if STORE_PERFORMANCE_METRIC_INFO
			freqCSset->items[tmpParentIdx].numInRef += freqCSset->items[freqId1].numInRef;
			freqCSset->items[tmpParentIdx].numFill += freqCSset->items[freqId1].numFill;
			#endif

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

PropStat* initPropStat(void){

	PropStat *propStat = (PropStat *) malloc(sizeof(PropStat));
	propStat->pBat = BATnew(TYPE_void, TYPE_oid, INIT_PROP_NUM, TRANSIENT);

	BATseqbase(propStat->pBat, 0);
	
	if (propStat->pBat == NULL) {
		return NULL; 
	}

	(void)BAThash(propStat->pBat,0);
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
	{
	int ret; 
	char*   schema = "rdf";
	str propStr; 

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		printf("Fail in opening Tokenizer \n");
	}

	for (i = 0; i < (int)BATcount(propStat->pBat); i++){
		oid* propId;
		propId = (oid *) Tloc(propStat->pBat, i); 
		takeOid(*propId, &propStr);	
		printf("Prop %d || Id: " BUNFMT " (%s)|  freq: %d",i, *propId, propStr, propStat->freqs[i]);
		printf("   tfidf: %f \n",propStat->tfidfs[i] );
		GDKfree(propStr);
	}

	TKNZRclose(&ret);
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

	for (i = 0; i < propStat->numAdded; i++){
		propStat->tfidfs[i] = tfidfComp(propStat->freqs[i],numTables);
	}

	*numdistinctMCS = k; 

	return propStat; 
}

#if NO_OUTPUTFILE == 0
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
#endif

void freePropStat(PropStat *propStat){
	int i; 
	BBPreclaim(propStat->pBat); 
	free(propStat->freqs); 
	free(propStat->tfidfs); 
	for (i = 0; i < propStat->numAdded; i++){
		if (propStat->plCSidx[i].lstIdx) 
			free(propStat->plCSidx[i].lstIdx);
		if (propStat->plCSidx[i].lstInvertIdx)
			free(propStat->plCSidx[i].lstInvertIdx); 
		if (propStat->plCSidx[i].lstOnt)
			free(propStat->plCSidx[i].lstOnt);
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
void generatecsRelSum(CSrel csRel, int freqId, CSset* freqCSset, CSrelSum *csRelSum, PropStat *propStat){
	int i; 
	int propIdx; 
	int refIdx; 
	int freq; 
	int referredFreqId;
	int freqOfReferredCS; 
	oid p; 
	BUN bun = BUN_NONE;
	
	csRelSum->origFreqIdx = freqId;
	csRelSum->numProp = freqCSset->items[freqId].numProp;
	copyOidSet(csRelSum->lstPropId, freqCSset->items[freqId].lstProp, csRelSum->numProp);

	for (i = 0; i < csRelSum->numProp; i++){
		csRelSum->numPropRef[i] = 0;
	}

	for (i = 0; i < csRel.numRef; i++){
		freq = freqCSset->items[csRel.origFreqIdx].support; 
		referredFreqId = csRel.lstRefFreqIdx[i];
		freqOfReferredCS = freqCSset->items[referredFreqId].support;
		if (freq > MIN_FROMTABLE_SIZE_S5 && (((float)freq * infreqTypeThreshold) < csRel.lstCnt[i])   
		    && freqOfReferredCS < csRel.lstCnt[i] * MIN_TO_PERCETAGE_S5){			
			
			p = csRel.lstPropId[i]; 
			bun = BUNfnd(propStat->pBat,(ptr) &p);
			assert(bun != BUN_NONE);
			//printf("Prop " BUNFMT "Prop TFIDF score in S5 is %f \n",p, propStat->tfidfs[bun]); 
			if (propStat->tfidfs[bun] > MIN_TFIDF_PROP_S5){

				propIdx = 0;
				while (csRelSum->lstPropId[propIdx] != csRel.lstPropId[i])
					propIdx++;
			
				//Add to this prop
				refIdx = csRelSum->numPropRef[propIdx];
				csRelSum->freqIdList[propIdx][refIdx] = csRel.lstRefFreqIdx[i]; 
				csRelSum->numPropRef[propIdx]++;
				
				/*
				if (csRelSum->numPropRef[propIdx] >  1){
					int j;
					int toFreqId; 
					printf("Prop TFIDF score in S5 is %f \n",propStat->tfidfs[bun]); 
					for (j = 0; j < csRelSum->numPropRef[propIdx]; j++){
						toFreqId = csRelSum->freqIdList[propIdx][j];
						printf(" FreqCS %d (freq: %d | coverage: %d) ", toFreqId,freqCSset->items[toFreqId].support, freqCSset->items[toFreqId].coverage);
					}
					printf("Will be merged with S5: Refer from freqCS %d (freq:%d | cov: %d) with prop "BUNFMT" --> numRef = %d \n", freqId,freq, freqCSset->items[freqId].coverage, csRelSum->lstPropId[propIdx],csRel.lstCnt[i]);
				}
				*/
			}
		}
	}

}

static
LabelStat* initLabelStat(void){
	LabelStat *labelStat = (LabelStat*) malloc(sizeof(LabelStat)); 
	labelStat->labelBat = BATnew(TYPE_void, TYPE_oid, INIT_DISTINCT_LABEL, TRANSIENT);	
	if (labelStat->labelBat == NULL){
		return NULL; 
	}
	(void)BAThash(labelStat->labelBat,0);
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
	#if USE_NAME_INSTEADOF_CANDIDATE_IN_S1
		(void) candIdx;

		return labels[freqIdx].name; 
	#else
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
	#endif

}
#endif

#if DETECT_INCORRECT_TYPE_SUBJECT

#if USING_FINALTABLE
static
void buildLabelStatForTable(LabelStat *labelStat, int numTables, CStableStat* cstablestat){
	int 	i; 
	BUN 	bun; 
	int 	*_tmp; 
	int	numDummy = 0;
	oid	name; 
	int	tblIdx = -1;

	//Preparation
	for (i = 0; i  < numTables; i++){
		if ( cstablestat->lstcstable[i].tblname != BUN_NONE){
			name = cstablestat->lstcstable[i].tblname;
			bun = BUNfnd(labelStat->labelBat,(ptr) &name);
			if (bun == BUN_NONE) {
				//New string
				if (labelStat->labelBat->T->hash && BATcount(labelStat->labelBat) > 4 * labelStat->labelBat->T->hash->mask) {
					HASHdestroy(labelStat->labelBat);
					BAThash(labelStat->labelBat, 2*BATcount(labelStat->labelBat));
				}

				BUNappend(labelStat->labelBat, (ptr) &name, TRUE);
						
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
		else
			numDummy++;
	}
	
	//printf("Collect label stat for final table: Total number of distinct labels %d \n", labelStat->numLabeladded);
	//printf("Number of DUMMY freqCS: %d \n",numDummy);

	//Build list of freqId corresponding to each label
	labelStat->freqIdList = (int**) malloc(sizeof(int*) * labelStat->numLabeladded);
	for (i =0; i < labelStat->numLabeladded; i++){
		labelStat->freqIdList[i] = (int*)malloc(sizeof(int) * labelStat->lstCount[i]);
		//reset the lstCount
		labelStat->lstCount[i] = 0;
	}
	
	for (i = 0; i  < numTables; i++){
		name = cstablestat->lstcstable[i].tblname;
		if (name != BUN_NONE){
			bun = BUNfnd(labelStat->labelBat,(ptr) &name);
			if (bun == BUN_NONE) {
				fprintf(stderr, "[Error] All the name should be stored already!\n");
			}
			else{
				tblIdx = labelStat->lstCount[bun];
				labelStat->freqIdList[bun][tblIdx] = i; 
				labelStat->lstCount[bun]++;
			}
		}
	}

}

#else /* USING_FINALTABLE = 0*/

static
void buildLabelStatForFinalMergeCS(LabelStat *labelStat, CSset *freqCSset, CSlabel *labels){
	int 	i; 
	BUN 	bun; 
	int 	*_tmp; 
	int	numDummy = 0;
	oid	name; 
	int	freqIdx = -1;

	//Preparation
	for (i = 0; i  < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx != -1) continue; 

		if ( labels[i].name != BUN_NONE){
			name = labels[i].name;
			bun = BUNfnd(labelStat->labelBat,(ptr) &name);
			if (bun == BUN_NONE) {
				//New string
				if (labelStat->labelBat->T->hash && BATcount(labelStat->labelBat) > 4 * labelStat->labelBat->T->hash->mask) {
					HASHdestroy(labelStat->labelBat);
					BAThash(labelStat->labelBat, 2*BATcount(labelStat->labelBat));
				}

				BUNappend(labelStat->labelBat, (ptr) &name, TRUE);
						
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
		else
			numDummy++;
	}
	
	//printf("Collect label stat for final mergeCS: Total number of distinct labels %d \n", labelStat->numLabeladded);
	//printf("Number of DUMMY freqCS: %d \n",numDummy);

	//Build list of freqId corresponding to each label
	labelStat->freqIdList = (int**) malloc(sizeof(int*) * labelStat->numLabeladded);
	for (i =0; i < labelStat->numLabeladded; i++){
		labelStat->freqIdList[i] = (int*)malloc(sizeof(int) * labelStat->lstCount[i]);
		//reset the lstCount
		labelStat->lstCount[i] = 0;
	}
	

	for (i = 0; i  < freqCSset->numCSadded; i++){
		if (freqCSset->items[i].parentFreqIdx != -1) continue; 

		name = labels[i].name;
		if (name != BUN_NONE){
			bun = BUNfnd(labelStat->labelBat,(ptr) &name);
			if (bun == BUN_NONE) {
				fprintf(stderr, "[Error] All the name should be stored already!\n");
			}
			else{
				freqIdx = labelStat->lstCount[bun];
				labelStat->freqIdList[bun][freqIdx] = i; 
				labelStat->lstCount[bun]++;
			}
		}
	}

}
#endif

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
				bun = BUNfnd(labelStat->labelBat,(ptr) &candidate);
				if (bun == BUN_NONE) {
					/*New string*/
					if (labelStat->labelBat->T->hash && BATcount(labelStat->labelBat) > 4 * labelStat->labelBat->T->hash->mask) {
						HASHdestroy(labelStat->labelBat);
						BAThash(labelStat->labelBat, 2*BATcount(labelStat->labelBat));
					}

					BUNappend(labelStat->labelBat, (ptr) &candidate, TRUE);
							
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
				bun = BUNfnd(labelStat->labelBat,(ptr) &candidate);
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
char isSignificationPrecisionDrop(CS *cs1, CS *cs2){
	int newSupport = 0;
	int newFill = 0; 
	float fillRatio1, fillRatio2, minFillRatio; 
	float estimatedFillRatio = 0.0;
	int numCombineP = 0;
	
	newSupport = cs1->support + cs2->support;
	newFill = cs1->numFill + cs2->numFill; 
	
	getNumCombinedP(cs1->lstProp, cs2->lstProp, cs1->numProp, cs2->numProp, &numCombineP);

	fillRatio1 = (float) cs1->numFill / (float) (cs1->numProp * cs1->support); 
	fillRatio2 = (float) cs2->numFill / (float) (cs2->numProp * cs2->support); 
	minFillRatio = (fillRatio1 > fillRatio2) ? fillRatio2 : fillRatio1;

	estimatedFillRatio = (float) newFill / (float) (newSupport * numCombineP);

	if ((minFillRatio / estimatedFillRatio) > 5) return 1; 

	return 0;
}

static
char isNoCommonProp(CS *cs1, CS *cs2){
	int numCombineP = 0;
	
	getNumCombinedP(cs1->lstProp, cs2->lstProp, cs1->numProp, cs2->numProp, &numCombineP);
	
	if (numCombineP == (cs1->numProp + cs2->numProp)) return 1; 

	return 0;
}
static 
void doMerge(CSset *freqCSset, int ruleNum, int freqId1, int freqId2, oid *mergecsId, CSlabel** labels, oid** ontmetadata, int ontmetadataCount, oid name, int isType, int isOntology, int isFK){
	CS 	*mergecs; 
	int		existMergecsId; 
	CS		*existmergecs, *mergecs1, *mergecs2; 
	int	k; 
	CS 	*cs1, *cs2;

	cs1 = &(freqCSset->items[freqId1]);
	cs2 = &(freqCSset->items[freqId2]);
	

	if (0){
		if (isSignificationPrecisionDrop(cs1, cs2)){
			printf("Merging freqCS %d and %d may significantly drop precision\n", freqId1, freqId2);
			return;
		}
		if (isNoCommonProp(cs1, cs2)){
			printf("FreqCS %d and %d have no prop in common--> no merging\n", freqId1, freqId2);
			return;
		}
	}

	//Check whether these CS's belong to any mergeCS
	if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx == -1){	/* New merge */
		mergecs = mergeTwoCSs(*cs1,*cs2, freqId1,freqId2, *mergecsId);
		//addmergeCStoSet(mergecsSet, *mergecs);
		cs1->parentFreqIdx = freqCSset->numCSadded;
		cs2->parentFreqIdx = freqCSset->numCSadded;
		addCStoSet(freqCSset,*mergecs);
		updateLabel(ruleNum, freqCSset, labels, 1, freqCSset->numCSadded - 1, freqId1, freqId2, name, isType, isOntology, isFK, ontmetadata, ontmetadataCount, NULL, -1);
		free(mergecs);
		
		mergecsId[0]++;
	}
	else if (cs1->parentFreqIdx == -1 && cs2->parentFreqIdx != -1){
		existMergecsId = cs2->parentFreqIdx;
		existmergecs = &(freqCSset->items[existMergecsId]);
		mergeACStoExistingmergeCS(*cs1,freqId1, existmergecs);
		cs1->parentFreqIdx = existMergecsId; 
		updateLabel(ruleNum, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, name, isType, isOntology, isFK, ontmetadata, ontmetadataCount, NULL, -1);
	}
	
	else if (cs1->parentFreqIdx != -1 && cs2->parentFreqIdx == -1){
		existMergecsId = cs1->parentFreqIdx;
		existmergecs = &(freqCSset->items[existMergecsId]);
		mergeACStoExistingmergeCS(*cs2,freqId2, existmergecs);
		cs2->parentFreqIdx = existMergecsId; 
		updateLabel(ruleNum, freqCSset, labels, 0, existMergecsId, freqId1, freqId2, name, isType, isOntology, isFK, ontmetadata, ontmetadataCount, NULL, -1);
	}
	else if (cs1->parentFreqIdx != cs2->parentFreqIdx){
		mergecs1 = &(freqCSset->items[cs1->parentFreqIdx]);
		mergecs2 = &(freqCSset->items[cs2->parentFreqIdx]);
		
		mergeTwomergeCS(mergecs1, mergecs2, cs1->parentFreqIdx);

		//Re-map for all maxCS in mergecs2
		for (k = 0; k < mergecs2->numConsistsOf; k++){
			freqCSset->items[mergecs2->lstConsistsOf[k]].parentFreqIdx = cs1->parentFreqIdx;
		}
		updateLabel(ruleNum, freqCSset, labels, 0, cs1->parentFreqIdx, freqId1, freqId2, name, isType, isOntology, isFK, ontmetadata, ontmetadataCount, NULL, -1);
	}

}




static
str mergeFreqCSByS1(CSset *freqCSset, CSlabel** labels, oid *mergecsId, oid** ontmetadata, int ontmetadataCount,bat *mapbatid){
	int 		i, j; 
	CS		*cs1, *cs2;

	#if !USE_MULTIWAY_MERGING
	int		k;
	int 		freqId1, freqId2;
	int		tmpCount; 
	#else
	int		*lstDistinctFreqId = NULL;		
	int		numDistinct = 0;
	int		isNew = 0; 
	int  		mergeFreqIdx = -1; 
	#endif
	LabelStat	*labelStat = NULL; 
	oid		*name;

	#if ONLY_MERGE_URINAME_CS_S1
	ObjectType	objType; 	
	#endif

	#if ONLY_MERGE_ONTOLOGYBASEDNAME_CS_S1
	char		isOntName = 0; 
	BUN		tmpontClassPos = BUN_NONE; 
	#endif

	#if OUTPUT_FREQID_PER_LABEL
	FILE    	*fout;
	char*   	schema = "rdf";
	int		ret = 0;
	str		tmpLabel; 
	BAT		*mbat = NULL; 
	BATiter		mapi; 
	
	#if USE_SHORT_NAMES
	str canStrShort = NULL;
	#endif
	#endif
	(void) name; 
	(void) ontmetadata;
	(void) ontmetadataCount;
	#if     !NOT_MERGE_DIMENSIONCS_IN_S1
	(void) cs1;
	(void) cs2;
	#endif
	(void) mapbatid; 
	
	labelStat = initLabelStat(); 
	buildLabelStat(labelStat, (*labels), freqCSset, TOPK);
	printf("Num FreqCSadded before using S1 = %d \n", freqCSset->numCSadded);

	#if OUTPUT_FREQID_PER_LABEL

	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	mapi = bat_iterator(mbat); 

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	
	fout = fopen("freqIdPerLabel.txt","wt");
	#endif
	for (i = 0; i < labelStat->numLabeladded; i++){
		name = (oid*) Tloc(labelStat->labelBat, i);
		#if ONLY_MERGE_ONTOLOGYBASEDNAME_CS_S1
		tmpontClassPos = BUN_NONE; 
		isOntName = isOntologyName(*name, &tmpontClassPos);
		if (isOntName != 1 || tmpontClassPos == BUN_NONE){ 
			//printf("Name "BUNFMT" is not an ontology name \n", *name);
			continue; 
		}
		#endif
		#if ONLY_MERGE_URINAME_CS_S1
		objType = getObjType(*name); 
		if (objType != URI) continue; 
		#endif
		if (labelStat->lstCount[i] > 1){
			/*TODO: Multi-way merge */
			#if USE_MULTIWAY_MERGING	
			lstDistinctFreqId = mergeMultiCS(freqCSset,  labelStat->freqIdList[i], labelStat->lstCount[i], mergecsId, &numDistinct, &isNew, &mergeFreqIdx); 
			if (lstDistinctFreqId != NULL){
				updateLabel(S1, freqCSset, labels, isNew, mergeFreqIdx, -1, -1, *name, labelStat->freqIdList[i][0].isType, labelStat->freqIdList[i][0].isOntology, labelStat->freqIdList[i][0].isFK, ontmetadata, ontmetadataCount, lstDistinctFreqId, numDistinct); // use isType/isOntology/isFK information from first CS with that label
			}
			#else

			tmpCount = 0;
			for (k = 0; k < labelStat->lstCount[i]; k++){
				freqId1 = labelStat->freqIdList[i][k];
				cs1 = &(freqCSset->items[freqId1]);
				#if     NOT_MERGE_DIMENSIONCS_IN_S1
				if (cs1->type == DIMENSIONCS) continue;
				#endif
				tmpCount++;
				break; 
			}
			for (j = k+1; j < labelStat->lstCount[i]; j++){
				int isType = 0, isOntology = 0, isFK = 0;
				freqId2 = labelStat->freqIdList[i][j];
				cs2 = &(freqCSset->items[freqId2]);
				#if	NOT_MERGE_DIMENSIONCS_IN_S1
				if (cs2->type == DIMENSIONCS) continue; 
				#endif
				#if	INFO_WHERE_NAME_FROM
				if ((*labels)[freqId1].isType == 1 || (*labels)[freqId2].isType == 1) {
					isType = 1;
				} else if ((*labels)[freqId1].isOntology == 1 || (*labels)[freqId2].isOntology == 1) {
					isOntology = 1;
				} else if ((*labels)[freqId1].isFK == 1 || (*labels)[freqId2].isFK == 1) {
					isFK = 1;
				}
				#endif
				doMerge(freqCSset, S1, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, *name, isType, isOntology, isFK);
				tmpCount++;
			}

			#if OUTPUT_FREQID_PER_LABEL
			fprintf(fout, " %d freqCS merged as having same name (by Ontology, Type, FK). MergedCS has %d prop. \n", tmpCount, freqCSset->items[freqCSset->numCSadded -1].numProp);
			#endif
			
			#endif /* USE_MULTIWAY_MERGING */

			#if OUTPUT_FREQID_PER_LABEL
			//takeOid(*name, &tmpLabel); 
			getStringName(*name, &tmpLabel, mapi, mbat, 1);
			
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
void mergeFreqCSByS5(CSrel *csrelMergeFreqSet, CSset *freqCSset, CSlabel** labels, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId, oid** ontmetadata, int ontmetadataCount){
	int 		i; 
	int 		freqId;
	//int 		relId; 
	//CS*		cs1;
	CSrelSum 	*csRelSum; 
	int		maxNumRefPerCS = 0; 
	int 		j, k; 
	#if 		!USE_MULTIWAY_MERGING
	int 		freqId1, freqId2;
	CS		*cs1, *cs2;
	int		startIdx = 0; 
	#else
	int		*lstDistinctFreqId = NULL;		
	int		numDistinct = 0;
	int		isNew = 0; 
	int  		mergeFreqIdx = -1; 
	#endif	
	
	#if NO_OUTPUTFILE == 0
	char 		filename[100];
	FILE		*fout;
	int 		refFreqId;
	#endif
	int		maxNumPropInMergeCS =0;
	//int 		numCombinedP = 0; 
	PropStat 	*propStat;	//This is for checking whether the prop of the FK is common prop or not

	propStat = initPropStat();
	getPropStatisticsFromMergeCSs(propStat, curNumMergeCS, mergeCSFreqCSMap, freqCSset);

	//printf("Start merging CS by using S5[From FK] \n");
	
	#if NO_OUTPUTFILE == 0
	strcpy(filename, "csRelSum.txt");

	fout = fopen(filename,"wt"); 
	#endif

	for (i = 0; i < curNumMergeCS; i++){
		freqId = mergeCSFreqCSMap[i];
		if (csrelMergeFreqSet[freqId].numRef > maxNumRefPerCS)
		 	maxNumRefPerCS = csrelMergeFreqSet[freqId].numRef ; 		

		if (freqCSset->items[freqId].numProp > maxNumPropInMergeCS)
			maxNumPropInMergeCS = freqCSset->items[freqId].numProp;
	}
	//printf("maxNumRefPerCS = %d \n", maxNumRefPerCS);
	//printf("max number of prop in mergeCS: %d \n", maxNumPropInMergeCS);

	csRelSum = initCSrelSum(maxNumPropInMergeCS,maxNumRefPerCS);
	
	for (i = 0; i < curNumMergeCS; i++){
		freqId = mergeCSFreqCSMap[i];
		if (csrelMergeFreqSet[freqId].numRef != 0){
			generatecsRelSum(csrelMergeFreqSet[freqId], freqId, freqCSset, csRelSum,propStat);
			/* Check the number of */
			#if NO_OUTPUTFILE == 0
			fprintf(fout, "csRelSum " BUNFMT " (support: %d, coverage %d ): ",csRelSum->origFreqIdx, freqCSset->items[freqId].support, freqCSset->items[freqId].coverage);
			#endif
			for (j = 0; j < csRelSum->numProp; j++){
				if ( csRelSum->numPropRef[j] > 1){
					#if NO_OUTPUTFILE == 0
					fprintf(fout, "  P " BUNFMT " -->",csRelSum->lstPropId[j]);
					for (k = 0; k < csRelSum->numPropRef[j]; k++){
						refFreqId = csRelSum->freqIdList[j][k];
						fprintf(fout, " %d | ", refFreqId);
					}	
					#endif

					/* Merge each refCS into the first CS. 
					 * TODO: The Multi-way merging should be better
					 * */ 
					//mergeMultiPropList(freqCSset, csRelSum->freqIdList[j],csRelSum->numPropRef[j] , &numCombinedP);
					#if USE_MULTIWAY_MERGING	
					lstDistinctFreqId = mergeMultiCS(freqCSset, csRelSum->freqIdList[j],csRelSum->numPropRef[j], mergecsId, &numDistinct, &isNew, &mergeFreqIdx); 
					
					if (lstDistinctFreqId != NULL){
						updateLabel(S5, freqCSset, labels, isNew, mergeFreqIdx, -1, -1, BUN_NONE, 0, 0, 0, ontmetadata, ontmetadataCount, lstDistinctFreqId, numDistinct); // name, isType, isOntology, isFK are not used for case S5
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
						
						doMerge(freqCSset, S5, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, BUN_NONE, 0, 0, 0); // no name known

					}

					#endif /*If USE_MULTIWAY_MERGING */
				}
			}
			#if NO_OUTPUTFILE == 0
			fprintf(fout, "\n");
			#endif
		}
	}
	

	#if NO_OUTPUTFILE == 0
	fclose(fout); 
	#endif

	freeCSrelSum(maxNumPropInMergeCS, csRelSum);

	freePropStat(propStat);
}


static
char isSemanticSimilar(int freqId1, int freqId2, CSlabel* labels, OntoUsageNode *tree, int numOrigFreqCS, oid *ancestor, BAT *ontmetaBat, OntClass *ontclassSet){	/*Rule S1 S2 S3*/
	int i, j; 
	//int commonHierarchy = -1;
	int minCount = 0; 
	int hCount1, hCount2; 
	int level; 
	OntoUsageNode *tmpNode; 
		
	// Check for the most common ancestor
	hCount1 = labels[freqId1].hierarchyCount;
	hCount2 = labels[freqId2].hierarchyCount;
	minCount = (hCount1 > hCount2)?hCount2:hCount1;
	
	if (0){
	if ((freqId1 > numOrigFreqCS -1) || (freqId2 > numOrigFreqCS -1))
		return 0;
	}

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
		
		
		if (tmpNode->percentage < generalityThreshold) {
			//printf("Merge two CS's %d (Label: "BUNFMT") and %d (Label: "BUNFMT") using the common ancestor ("BUNFMT") at level %d (score: %f)\n",
			//		freqId1, labels[freqId1].name, freqId2, labels[freqId2].name,tmpNode->uri, i,tmpNode->percentage);
			oid classOid;
			BUN ontClassPos;
			classOid = tmpNode->uri;

			ontClassPos = BUNfnd(ontmetaBat, &classOid); 
			assert(ontClassPos != BUN_NONE);	
			
			/*
			if (ontClassPos != BUN_NONE){
				printf(" Specific level: %d \n", ontclassSet[ontClassPos].hierDepth);
			}
			*/
			
			if (ontclassSet[ontClassPos].hierDepth >= COMMON_ANCESTOR_LOWEST_SPECIFIC_LEVEL){
				(*ancestor) = tmpNode->uri;
				return 1;
			}
		}

	}


	return 0;
}

static
void initTFIDFInfos(TFIDFInfo *tfidfInfos, int curNumMergeCS, oid* mergeCSFreqCSMap, CSset *freqCSset, PropStat *propStat){
	int 	i, j; 
	int	freqId; 
	CS	*cs; 	
	oid	p; 
	float 	tfidfV = 0.0; 
	float	sum; 
	BUN	bun = BUN_NONE; 
	for (i = 0; i < curNumMergeCS; i++){
		freqId = mergeCSFreqCSMap[i];
		cs = (CS*) &(freqCSset->items[freqId]);
		tfidfInfos[i].freqId = freqId; 
		tfidfInfos[i].lsttfidfs = (float*)malloc(sizeof(float) * (cs->numProp)); 
		sum = 0.0; 
		for (j = 0; j < cs->numProp; j++){
			p = cs->lstProp[j]; 
			bun = BUNfnd(propStat->pBat,(ptr) &p);
			if (bun == BUN_NONE) {
				printf("This prop must be there!!!!\n");
			}
			else{
				tfidfV = propStat->tfidfs[bun]; 
				sum +=  tfidfV*tfidfV;	
			}
			tfidfInfos[i].lsttfidfs[j] = tfidfV; 

		}
		//assert(sum > 0); 
		tfidfInfos[i].totalTFIDF = sqrt(sum); 
	}
	
}
static 
void freeTFIDFInfo(TFIDFInfo *tfidfInfos, int curNumMergeCS){
	int i; 
	for (i = 0; i < curNumMergeCS; i++){
		free(tfidfInfos[i].lsttfidfs);
	}
	free(tfidfInfos);
}

static
void mergeCSByS2(CSset *freqCSset, CSlabel** labels, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId,OntoUsageNode *ontoUsageTree, oid **ontmetadata, int ontmetadataCount, BAT *ontmetaBat, OntClass *ontclassSet){
	int 		i, j; 
	int 		freqId1, freqId2; 

	char		isLabelComparable = 0; 
	oid		name;		/* Name of the common ancestor */
	

	
	(void) labels;
	(void) isLabelComparable;



	for (i = 0; i < curNumMergeCS; i++){		
		freqId1 = mergeCSFreqCSMap[i];

		isLabelComparable = 0; 
		if ((*labels)[freqId1].name != BUN_NONE) isLabelComparable = 1; // no "DUMMY"
				
		#if	NOT_MERGE_DIMENSIONCS
		if (freqCSset->items[freqId1].type == DIMENSIONCS) continue; 
		#endif

		if ((*labels)[freqId1].hierarchyCount < 1) continue; 

	 	for (j = (i+1); j < curNumMergeCS; j++){
			freqId2 = mergeCSFreqCSMap[j];
			#if	NOT_MERGE_DIMENSIONCS
			if (freqCSset->items[freqId2].type == DIMENSIONCS) continue; 
			#endif
			if (isLabelComparable == 1 && isSemanticSimilar(freqId1, freqId2, (*labels), ontoUsageTree,freqCSset->numOrigFreqCS, &name, ontmetaBat, ontclassSet) == 1){
				//printf("Same labels between freqCS %d and freqCS %d - Old simscore is %f \n", freqId1, freqId2, simscore);
				doMerge(freqCSset, S2, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, name, 0, 1, 0); // isOntology because of the common ancestor name that was found in isSemanticSimilar
			}

		}
	}

}

static
void mergeCSByS4(CSset *freqCSset, CSlabel** labels, oid* mergeCSFreqCSMap, int curNumMergeCS, oid *mergecsId,oid **ontmetadata, int ontmetadataCount){
	int 		i, j; 
	int 		freqId1, freqId2; 
	float 		simscore = 0.0; 
	CS		*cs1, *cs2;
	int             numCombineP = 0;

	PropStat	*propStat; 	/* Store statistics about properties */
	TFIDFInfo	*tfidfInfos;
	
	char		existDiscriminatingProp = 0; 
	#if ONLY_MERGE_ONTOLOGYBASEDNAME_CS_S1
	char 		isSameLabel = 0; 
	#endif

	int		oldNumCSadded = 0;
	(void) oldNumCSadded;
	/*
	int ret; 
	char*   schema = "rdf";
	str freqCSname1, freqCSname2; 

	TKNZRopen (NULL, &schema);
	*/
	#if UPDATE_NAME_BASEDON_POPULARTABLE
	oldNumCSadded = freqCSset->numCSadded;
	#endif

	(void) labels;

	propStat = initPropStat();
	getPropStatisticsFromMergeCSs(propStat, curNumMergeCS, mergeCSFreqCSMap, freqCSset); /*TODO: Get PropStat from MaxCSs or From mergedCS only*/
	tfidfInfos = (TFIDFInfo*)malloc(sizeof(TFIDFInfo) * curNumMergeCS); 
	initTFIDFInfos(tfidfInfos, curNumMergeCS, mergeCSFreqCSMap, freqCSset, propStat); 

	for (i = 0; i < curNumMergeCS; i++){		
		freqId1 = mergeCSFreqCSMap[i];
		//printf("Label of %d CS is %s \n", freqId1, (*labels)[freqId1].name);
				
		#if	NOT_MERGE_DIMENSIONCS
		if (freqCSset->items[freqId1].type == DIMENSIONCS) continue; 
		#endif
	 	for (j = (i+1); j < curNumMergeCS; j++){
			cs1 = (CS*) &(freqCSset->items[freqId1]);

			freqId2 = mergeCSFreqCSMap[j];
			cs2 = (CS*) &(freqCSset->items[freqId2]);
			#if	NOT_MERGE_DIMENSIONCS
			if (cs2->type == DIMENSIONCS) continue; 
			#endif
			
			if (cs1->parentFreqIdx != -1 && cs1->parentFreqIdx == cs2->parentFreqIdx) continue; //They have already been merged
			
			existDiscriminatingProp = 0;

			if(USINGTFIDF == 0){
				simscore = similarityScore(cs1->lstProp, cs2->lstProp,
					cs1->numProp,cs2->numProp,&numCombineP);

				//printf("simscore Jaccard = %f \n", simscore);
			}
			else{
				simscore = similarityScoreTFIDF(cs1->lstProp, cs2->lstProp,
					cs1->numProp,cs2->numProp,&numCombineP, tfidfInfos, i, j, &existDiscriminatingProp);
				
			}
			
			//simscore = 0.0;
			#if	USINGTFIDF	
			  #if ONLY_MERGE_ONTOLOGYBASEDNAME_CS_S1
			  isSameLabel = 0;
			  if ((*labels)[freqId1].name == (*labels)[freqId2].name) isSameLabel = 1;

			  if (simscore > simTfidfThreshold && (existDiscriminatingProp || isSameLabel)){
			  #else	
			  if (simscore > simTfidfThreshold && existDiscriminatingProp){	  
			  //if (simscore > simTfidfThreshold){	  
			  #endif
			#else	
			if (simscore > SIM_THRESHOLD) {
			#endif	
			 	/*
                               	if ((*labels)[freqId1].name != BUN_NONE){
					takeOid((*labels)[freqId1].name, &freqCSname1);
					printf("Merge %d (%s) and ",freqId1, freqCSname1);
					GDKfree(freqCSname1);
				}
				else{
					printf("Merge %d (DUMMY) and ",freqId1);
				}
				if ((*labels)[freqId2].name != BUN_NONE){
					takeOid((*labels)[freqId2].name, &freqCSname2);
					printf(" %d (%s) with simscore = %f \n",freqId2, freqCSname2, simscore);
					GDKfree(freqCSname2);
				}
				else{
					printf(" %d (DUMMY) with simscore = %f \n",freqId2, simscore);
				}
				*/
				doMerge(freqCSset, S4, freqId1, freqId2, mergecsId, labels, ontmetadata, ontmetadataCount, BUN_NONE, 0, 0, 0); // no name known
			}
		}
	}
	#if UPDATE_NAME_BASEDON_POPULARTABLE
	{
	int	tmpSubFreqId = -1;
	int	tmpFreqIdwithMaxSupport = -1;
	int	tmpmaxSupport = 0;
	int	k; 
	oid	oldName; 
	oid 	newName;
	for (i = oldNumCSadded; i < freqCSset->numCSadded; i++){
		freqId1 = i;
		cs1 = (CS*) &(freqCSset->items[freqId1]);
		oldName = (*labels)[freqId1].name;

		if (cs1->parentFreqIdx == -1 && oldName != BUN_NONE){
			tmpmaxSupport = 0; 
			newName = BUN_NONE; 
			for (j = 0; j < cs1->numConsistsOf; j++){
                        	tmpSubFreqId  = cs1->lstConsistsOf[j];
				if (freqCSset->items[tmpSubFreqId].support > tmpmaxSupport){
					tmpFreqIdwithMaxSupport = tmpSubFreqId;	
					tmpmaxSupport = freqCSset->items[tmpSubFreqId].support;	
				}
			}
			
			newName = (*labels)[tmpFreqIdwithMaxSupport].name;
			if (newName != BUN_NONE && newName != oldName){
				//update label
				(*labels)[freqId1].name = newName;
				//update candidates
				assert(oldName == (*labels)[freqId1].candidates[0]);
				for (k = 1; k < (*labels)[freqId1].candidatesCount; k++){
					//If newName is already in the candidates, swap the first candidate with this
					if ((*labels)[freqId1].candidates[k] == newName){	
						(*labels)[freqId1].candidates[k] = oldName; 
						(*labels)[freqId1].candidates[0] = newName;
						break;
					}	
				}
				//If no candidate has the new Name
				if ((*labels)[freqId1].candidates[0] != newName){
					(*labels)[freqId1].candidates[0] = newName;
				}
			}
		}
	}
	}
	#endif

	//TKNZRclose(&ret);

	freePropStat(propStat);
	freeTFIDFInfo(tfidfInfos, curNumMergeCS);

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

#if NO_OUTPUTFILE == 0
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
#endif

#if NO_OUTPUTFILE == 0
static void getStatisticFinalCSs(CSset *freqCSset, BAT *sbat, int freqThreshold, int numTables, int* mergeCSFreqCSMap, CSPropTypes* csPropTypes, CSlabel *labels){

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
		if (isCSTable(freqCSset->items[freqId],labels[freqId].name)){		// Check whether it is a maximumCS
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
	if (numMergeCS != 0) printf("Avg number of triples coverred by one final CS: %f \n", (float)(totalCoverage/numMergeCS));

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
		if (isCSTable(freqCSset->items[freqId], labels[freqId].name)){		// Check whether it is a maximumCS
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
					if ((csPropTypes[i].lstPropTypes[j].propFreq * k)  < freqCSset->items[freqId].support * infreqPropThreshold){
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
		if (isCSTable(freqCSset->items[freqId],labels[freqId].name)){		// Check whether it is a maximumCS
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
#endif

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
	csBats->hsKeyBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);

	BATseqbase(csBats->hsKeyBat, 0);
	
	if (csBats->hsKeyBat == NULL) {
		return NULL; 
	}

	(void)BAThash(csBats->hsKeyBat,0);
	if (!(csBats->hsKeyBat->T->hash)){
		return NULL;
	}

	csBats->hsValueBat = BATnew(TYPE_void, TYPE_int, smallbatsz, TRANSIENT);

	if (csBats->hsValueBat == NULL) {
		return NULL; 
	}
	csBats->pOffsetBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
	
	if (csBats->pOffsetBat == NULL) {
		return NULL; 
	}
	csBats->fullPBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
	
	if (csBats->fullPBat == NULL) {
		return NULL; 
	}

	//#if EXTRAINFO_FROM_RDFTYPE

	csBats->typeOffsetBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
	
	if (csBats->typeOffsetBat == NULL) {
		return NULL; 
	}
	csBats->fullTypeBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
	
	if (csBats->fullTypeBat == NULL) {
		return NULL; 
	}
	//#endif

	csBats->freqBat = BATnew(TYPE_void, TYPE_int, smallbatsz, TRANSIENT);
	
	if (csBats->freqBat == NULL) {
		return NULL; 
	}

	csBats->coverageBat = BATnew(TYPE_void, TYPE_int, smallbatsz, TRANSIENT);
	
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
	#if EXTRAINFO_FROM_RDFTYPE
	BBPreclaim(csBats->typeOffsetBat); 
	BBPreclaim(csBats->fullTypeBat); 
	#endif
	free(csBats);

}


#if NO_OUTPUTFILE == 0 
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
	outputBat = BATnew(TYPE_void, TYPE_int, numTbl, TRANSIENT);
	if (outputBat == NULL){
		return NULL; 
	}
	(void)BAThash(outputBat,0);
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
		bun = BUNfnd(outputBat,(ptr) &tmpIdx);
		if (bun == BUN_NONE) {
			/*New FreqIdx*/
			if (outputBat->T->hash && BATcount(outputBat) > 4 * outputBat->T->hash->mask) {
				HASHdestroy(outputBat);
				BAThash(outputBat, 2*BATcount(outputBat));
			}
			BUNappend(outputBat, (ptr) &tmpIdx, TRUE);
			i++;
		}
		numLoop++;
	}

	//Print the results
	printf("Get the sample tables after %d loop \n",numLoop );

	free(cumDist); 

	return outputBat; 
}
#endif




static 
BAT* buildTypeOidBat(void){
	int ret; 
	char*   schema = "rdf";
	BAT*	typeBat;
	oid	tmpAttributeOid; 
	int	i; 

	typeBat = BATnew(TYPE_void, TYPE_oid, typeAttributesCount + 1, TRANSIENT);
	
	if (typeBat == NULL){
		printf("In rdfschema.c/buildTypeOidBat: Cannot create new bat\n");
		return NULL;
	}	
	(void)BAThash(typeBat,0);
	
	if (!(typeBat->T->hash)){
		printf("rdfschema.c/buildTypeOidBat: Cannot allocate the hash for Bat \n");
		return NULL; 
	}

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		printf("rdfschema.c/buildTypeOidBat: Could not open the tokenizer\n");
		return NULL; 
	}

	for (i = 0; i < typeAttributesCount; ++i) {
		TKNZRappend(&tmpAttributeOid, &typeAttributes[i]);
		BUNappend(typeBat, &tmpAttributeOid, TRUE);
	}


	TKNZRclose(&ret);

	return typeBat; 
}




#if EXTRAINFO_FROM_RDFTYPE
//Check whether a prop is a rdf:type prop
static 
char isTypeAttribute(oid propId, BAT* typeBat){
	BUN	bun; 

	bun = BUNfnd(typeBat, &propId);
 	if (bun == BUN_NONE){
		return 0; 
	}
	else
		return 1; 
}

//Get the specific level of an object value 
//in ontology class hierarchy
//-1: Not an ontology class
//Higher number --> more specific



static
int getOntologySpecificLevel(oid valueOid, BUN *ontClassPos){	

	*ontClassPos = BUNfnd(ontmetaBat, &valueOid);
	if (*ontClassPos == BUN_NONE) 		//Not an ontology class
		return -1;
	else
		return ontclassSet[*ontClassPos].hierDepth;
}

#if DETECT_INCORRECT_TYPE_SUBJECT
static
char isSupSuperOntology(oid value1, oid value2){
	BUN ontclasspos1 = BUN_NONE;
	BUN ontclasspos2 = BUN_NONE;
	int tmpscPos = -1;
	int j;
	
	ontclasspos1 = BUNfnd(ontmetaBat, &value1);
	ontclasspos2 = BUNfnd(ontmetaBat, &value2);

	if (ontclasspos1 == BUN_NONE || ontclasspos2 == BUN_NONE) return 0;
	
	//check the superclass for value 1
	for (j = 0; j < ontclassSet[ontclasspos1].numsc; j++){
		tmpscPos = ontclassSet[ontclasspos1].scIdxes[j]; 
		if (tmpscPos == (int)ontclasspos2) return 1;
	}
	
	//check the superclass for value 2
	for (j = 0; j < ontclassSet[ontclasspos2].numsc; j++){
		tmpscPos = ontclassSet[ontclasspos2].scIdxes[j]; 
		if (tmpscPos == (int)ontclasspos1) return 1;
	}

	return 0;
}
#endif
	
static
PropStat* getPropStatisticsByOntologyClass(int numClass, OntClass *ontClassSet){

	int i, j; 

	PropStat* propStat; 
	
	propStat = initPropStat(); 

	for (i = 0; i < numClass; i++){
		for (j = 0; j < ontClassSet[i].numProp; j++){
			addaProp(propStat, ontClassSet[i].lstProp[j], i, j);
		}

		if (ontClassSet[i].numProp > propStat->maxNumPPerCS)
			propStat->maxNumPPerCS = ontClassSet[i].numProp;
	}

	for (i = 0; i < propStat->numAdded; i++){
		propStat->tfidfs[i] = tfidfComp(propStat->freqs[i],numClass);
	}


	return propStat; 
}

static
void initTFIDFInfosForOntologyClass(TFIDFInfo *tfidfInfos, int numClass, OntClass *ontClassSet, PropStat *propStat){
	int 	i, j; 
	oid	p; 
	float 	tfidfV = 0.0; 
	float	sum; 
	BUN	bun = BUN_NONE; 
	for (i = 0; i < numClass; i++){
		tfidfInfos[i].freqId = i; 
		tfidfInfos[i].lsttfidfs = (float*)malloc(sizeof(float) * (ontClassSet[i].numProp)); 
		sum = 0.0; 
		for (j = 0; j < ontClassSet[i].numProp; j++){
			p = ontClassSet[i].lstProp[j]; 
			bun = BUNfnd(propStat->pBat,(ptr) &p);
			if (bun == BUN_NONE) {
				printf("This prop must be there!!!!\n");
			}
			else{
				tfidfV = propStat->tfidfs[bun]; 
				sum +=  tfidfV*tfidfV;	
				tfidfInfos[i].lsttfidfs[j] = tfidfV; 
			}

		}
		//assert(sum > 0); 
		tfidfInfos[i].totalTFIDF = sqrt(sum); 
	}
}


// Compute the similarity score
// between a CS and an ontology class.
// It is not really TFIDF/cosin
// This is for the purpose of choosing the ontology name among 
// all the type values that can be best matched 
// for a CS. 
// We compute the TF-IDF score for each prop, then
// the similarity between the CS and an ontology name
// = Sum all TF-IDF scores of all common prop.
static 
float similarityScoreWithOntologyClass(oid* arr1, oid* arr2, int m, int n, 
		TFIDFInfo *tfidfInfos, int ontClassIdx, int *numOverlap){
	
	int i = 0, j = 0;
	int numCommon = 0; 
	float sumXY = 0.0;

	i = 0;
	j = 0;
	while( i < n && j < m )
	{
		if( arr1[j] < arr2[i] ){
			j++;

		}
		else if( arr1[j] == arr2[i] )
		{

			sumXY += tfidfInfos[ontClassIdx].lsttfidfs[j] * tfidfInfos[ontClassIdx].lsttfidfs[j];
			j++;
			i++;
			numCommon++;

		}
		else if( arr1[j] > arr2[i] )
			i++;
	}
	
	*numOverlap = numCommon;
	if (sumXY == 0) return 0; 

	return  ((float) sumXY);
}

#if COUNT_PERCENTAGE_ONTO_PROP_USED

static 
void countNumOverlapProp(oid* arr1, oid* arr2, int m, int n, 
		int *numOverlap){
	
	int i = 0, j = 0;
	int numCommon = 0; 

	i = 0;
	j = 0;
	while( i < n && j < m )
	{
		if( arr1[j] < arr2[i] ){
			j++;

		}
		else if( arr1[j] == arr2[i] )
		{
			j++;
			i++;
			numCommon++;

		}
		else if( arr1[j] > arr2[i] )
			i++;
	}
	
	*numOverlap = numCommon;

}
#endif
	
static
void getBestRdfTypeValue(oid *buff, int numP, oid *rdftypeOntologyValues, char *rdftypeSelectedValues, char *rdftypeSpecificLevels, BUN *rdftypeOntClassPos, int *numTypeValues, int maxSpecificLevel, TFIDFInfo *tfidfInfos){
	int i, j, k;
	int tmpPos;
	int tmpscPos = -1; 	//position of super class
	int numRemainValues = *numTypeValues; 
	float tmpSim = 0.0;
	float maxSim = 0.0;
	int	tmpNumOverlap = 0;
	oid	choosenTypeValue = BUN_NONE; 
	oid	mostSpecificValue = BUN_NONE;

	(void) rdftypeOntologyValues;

	//Step 1: Prune the candidates
	for (i = 0; i < *numTypeValues; i++){
		
		/*
		if (isShow){
			printf("Ontology value: "BUNFMT " Level: %d \n", rdftypeOntologyValues[i], rdftypeSpecificLevels[i]);
		}
		*/
		if (rdftypeSelectedValues[i] == 0) continue; 
		
		//Do not check non-specific type
		if (rdftypeSpecificLevels[i] < (maxSpecificLevel - 1)){	
			rdftypeSelectedValues[i] = 0;
			numRemainValues--;
			continue; 
		}

		//Do not check keep a super-class if its subclass is there
		tmpPos = (int) rdftypeOntClassPos[i];
		for (j = 0; j < ontclassSet[tmpPos].numsc; j++){
			tmpscPos = ontclassSet[tmpPos].scIdxes[j]; 
			//Go through and de-select all superclass
			for (k = 0; k < *numTypeValues; k++){
				if (tmpscPos == (int)rdftypeOntClassPos[k] && rdftypeSelectedValues[k] == 1){
					rdftypeSelectedValues[k] = 0;
					numRemainValues--;
				}
			}
		}
	}
	
	assert(numRemainValues > 0);

	//Step 2: Select the best one based on ontology class and specific level
	//if (isShow) printf("numRemainValues = %d \n",numRemainValues);

	if (numRemainValues > 1){
		//Compare the list of props with each ontology class
		for (i = 0; i < *numTypeValues; i++){
			if (rdftypeSelectedValues[i] == 0) continue;

			tmpPos = (int) rdftypeOntClassPos[i];
			tmpSim = similarityScoreWithOntologyClass(ontclassSet[tmpPos].lstProp, buff, ontclassSet[tmpPos].numProp,numP, tfidfInfos, tmpPos, &tmpNumOverlap);
			//if (isShow) printf("Sim with " BUNFMT " is %f \n",rdftypeOntologyValues[i],tmpSim);
			if (tmpSim > maxSim){
				choosenTypeValue = rdftypeOntologyValues[i];
				maxSim = tmpSim;
				//printf("It happens and the maxSim = %f \n",maxSim);
			}
			
			if (rdftypeSpecificLevels[i] == maxSpecificLevel){
				mostSpecificValue = rdftypeOntologyValues[i];
			}
		}

		if (choosenTypeValue == BUN_NONE){ //There is no overlapping prop, then choose the most specific one
			choosenTypeValue = mostSpecificValue;	
		}
	}
	else{
		for (i = 0; i < *numTypeValues; i++){
			if (rdftypeSelectedValues[i] == 0) continue;
			choosenTypeValue = rdftypeOntologyValues[i];
		}
	}
	
	assert(choosenTypeValue != BUN_NONE);

	//if (isShow) printf("Choosen value = "BUNFMT"\n", choosenTypeValue);

	//Only choose one value:
	*numTypeValues = 1;
	rdftypeOntologyValues[0] = choosenTypeValue;

}
#endif


#if COUNT_PERCENTAGE_ONTO_PROP_USED
/*
 * If the name of the CS comes from an ontology class, 
 * ontology contribution for the CS is computed as:
 * Ratio = (#_prop in the CS belonging to that ontology) / (# props of that ontology) 
 * Contribution = (Number of subject X Ratio)
 *
 * At the end, total 
 * */

static
str getOntologyContribution(CSset *freqCSset, CSlabel* labels){
	
	int i; 
	int noSubj; 
	int totalNoSubjWithOnt = 0; 
	int totalNoSubjWithOverlapP = 0; 
	int totalNoSubjWithNoP = 0; 
	int totalOntCSwithNoP = 0; 
	int totalNoSubj = 0; 
	float totalContrib = 0.0; 
	float contrib = 0.0;
	BUN tmpPos; 
	str nullURL2 = "<\x80>"; 	//str_nil
	str schema = "rdf";
	int ret; 
	oid nullURLid; 
	
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

	TKNRstringToOid(&nullURLid, &nullURL2); 

	//printf("Null URL Id is: "BUNFMT"\n", nullURLid);

	TKNZRclose(&ret); 
	

	for (i = 0; i < freqCSset->numCSadded; i++){
		CS cs = (CS)freqCSset->items[i];

		if (isOntologyName(labels[i].name, &tmpPos)){

			if (cs.parentFreqIdx != -1) continue;
			noSubj = cs.support;
			
			if (ontclassSet[tmpPos].numProp != 0 && ontclassSet[tmpPos].lstProp[0] != nullURLid){	//otherwise, we do not have the information for this ontology class
				int numOntProp = 0;

				countNumOverlapProp(ontclassSet[tmpPos].lstProp, cs.lstProp, ontclassSet[tmpPos].numProp,cs.numProp, &numOntProp);		
				contrib = (float) (numOntProp / (float) ontclassSet[tmpPos].numProp) * noSubj; 

				totalNoSubjWithOnt += noSubj;
				totalContrib += contrib;
					
				if (numOntProp == 0){
					totalOntCSwithNoP += 1; 
					totalNoSubjWithNoP += noSubj; 
				}
				else{
					totalNoSubjWithOverlapP += noSubj; 
				}
				
				//printf("CS %d has %d ontology props (/%d ontology props) \n",i, numOntProp,ontclassSet[tmpPos].numProp);
			}
		}

		totalNoSubj += cs.support;
	}

	printf("Ontology contribution is: %f  (If normalized by all subjs: %f | If only consider subject with overlapping P %f)\n", (float) totalContrib/totalNoSubjWithOnt, (float)totalContrib/totalNoSubj, (float) totalContrib/totalNoSubjWithOverlapP);
	printf("Number of CS with ontology name but having no prop from that ont class: %d  (%d subjs)\n", totalOntCSwithNoP, totalNoSubjWithNoP);

	return MAL_SUCCEED; 
}
#endif 

static 
str RDFassignCSId(int *ret, BAT *sbat, BAT *pbat, BAT *obat, BAT *ontbat, CSset *freqCSset, int *freqThreshold, CSBats* csBats, oid *subjCSMap, oid *maxCSoid, int *maxNumProp, int *maxNumPwithDup){

	int 	p; 
	oid 	sbt, pbt; 
	oid 	curS; 		/* current Subject oid */
	oid 	curP; 		/* current Property oid */
	oid 	CSoid = 0; 	/* Characteristic set oid */
	int 	numP; 		/* Number of properties for current S */
	int 	numPwithDup = 0; 
	oid*	buff; 	 
	oid*	_tmp;
	int 	INIT_PROPERTY_NUM = 100; 
	oid 	returnCSid; 
	oid	obt; 

	#if STOREFULLCS
	oid* 	buffObjs;
	oid* 	_tmpObjs; 
	#endif

	//Only keep the most specific ontology-based rdftype value 
	int	maxNumOntology = 20;		
	oid*	rdftypeOntologyValues = NULL; 
	char*   rdftypeSelectedValues = NULL; //Store which value is selected
	char* 	rdftypeSpecificLevels = NULL; //Store specific level for each value
	BUN*	rdftypeOntClassPos = NULL; //Position in the ontology class		     

	int	numTypeValues = 0;
	#if EXTRAINFO_FROM_RDFTYPE
	int	tmpMaxSpecificLevel = 0; 
	int	tmpSpecificLevel = 0; 
	BUN	tmpOntClassPos = BUN_NONE;  //index of the ontology class in the ontmetaBat
	PropStat        *ontPropStat = NULL;
	int		numOntClass = 0; 
	TFIDFInfo	*tfidfInfos = NULL;
	#endif
	
	int 	*buffOntologyNums = NULL;	//Number of instances in each ontology	
	int	numOnt = 0; 			//Number of ontology
	int     totalNumOntology = 0;		
	
	PropStat *fullPropStat; 	
	BAT	*typeBat;	//BAT contains oids of type attributes retrieved from tokenizer

	oid	*sbatCursor = NULL, *pbatCursor = NULL, *obatCursor = NULL;
	int	first, last;



	#if EXTRAINFO_FROM_RDFTYPE
	numOntClass = BATcount(ontmetaBat);
	ontPropStat = initPropStat();
	ontPropStat = getPropStatisticsByOntologyClass(numOntClass, ontclassSet);
	tfidfInfos = (TFIDFInfo*)malloc(sizeof(TFIDFInfo) * numOntClass);
	initTFIDFInfosForOntologyClass(tfidfInfos, numOntClass, ontclassSet, ontPropStat);
	#endif


	typeBat = buildTypeOidBat();

	printf("Number of attributes inserted into BAT: " BUNFMT "\n", BATcount(typeBat));
	rdftypeOntologyValues = (oid*)malloc(sizeof(oid) * maxNumOntology);
	rdftypeSelectedValues = (char*)malloc(sizeof(char) * maxNumOntology);
	rdftypeSpecificLevels = (char*)malloc(sizeof(char) * maxNumOntology);
	rdftypeOntClassPos = (BUN *) malloc(sizeof(BUN) * maxNumOntology);
		
	fullPropStat = initPropStat();
	
	buff = (oid *) malloc (sizeof(oid) * INIT_PROPERTY_NUM);
	#if STOREFULLCS
	buffObjs =  (oid *) malloc (sizeof(oid) * INIT_PROPERTY_NUM);
	#endif	

	numP = 0;
	curP = BUN_NONE; 
	curS = 0; 
	
	#if NO_OUTPUTFILE == 0
	{
	int i;
	char* schema = "rdf";

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

	numOnt = BATcount(ontbat); 
	buffOntologyNums = GDKmalloc(sizeof(int) * (numOnt+1));  //The last index stores number of non-ontology instances
	for (i = 0; i < (numOnt+1); i++){
		buffOntologyNums[i] = 0;
	}
	}
	#endif

	printf("freqThreshold = %d \n", *freqThreshold);	
	
	sbatCursor = (oid *) Tloc(sbat, BUNfirst(sbat));
	pbatCursor = (oid *) Tloc(pbat, BUNfirst(pbat));
	obatCursor = (oid *) Tloc(obat, BUNfirst(obat));

	first = 0; 
	last = BATcount(sbat) -1; 
	
	for (p = first; p <= last; p++){
		sbt = sbatCursor[p];
		if (sbt != curS){
			if (p != 0){	/* Not the first S */
				#if EXTRAINFO_FROM_RDFTYPE
				if (numTypeValues > 1){
					getBestRdfTypeValue(buff, numP, rdftypeOntologyValues, rdftypeSelectedValues, rdftypeSpecificLevels, rdftypeOntClassPos, &numTypeValues, tmpMaxSpecificLevel, tfidfInfos);
				}
				#endif
				#if STOREFULLCS
				returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, numTypeValues, rdftypeOntologyValues, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs, fullPropStat, ontbat, buffOntologyNums, &totalNumOntology,numOnt); 
				#else
				returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, numTypeValues, rdftypeOntologyValues, &CSoid, 1, *freqThreshold, freqCSset, fullPropStat, ontbat, buffOntologyNums, &totalNumOntology,numOnt); 
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
			curS = sbt; 
			curP = BUN_NONE;
			numP = 0;
			numPwithDup = 0; 

			numTypeValues = 0;
			#if EXTRAINFO_FROM_RDFTYPE
			tmpMaxSpecificLevel = 0;
			tmpSpecificLevel = 0;
			#endif
		}
				
		pbt = pbatCursor[p];

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

		#if EXTRAINFO_FROM_RDFTYPE
		if (isTypeAttribute(pbt, typeBat)){ //Check type attributes
			obt = obatCursor[p];
			tmpSpecificLevel = getOntologySpecificLevel(obt, &tmpOntClassPos);

			if (tmpOntClassPos != BUN_NONE){

				rdftypeSpecificLevels[numTypeValues] = tmpSpecificLevel;
				rdftypeOntClassPos[numTypeValues] = tmpOntClassPos;		
				rdftypeOntologyValues[numTypeValues] = obt;
				rdftypeSelectedValues[numTypeValues] = 1;
				numTypeValues++;

				if (tmpSpecificLevel > tmpMaxSpecificLevel){	
					tmpMaxSpecificLevel = tmpSpecificLevel;
					/*
					//only keep the most specific one
					rdftypeOntologyValues[0] = obt;
					numTypeValues = 1;
					*/
				}
			}

		}
		#endif
		
		if (curP != pbt){	/* Multi values property */		
			buff[numP] = pbt; 
			#if STOREFULLCS
			obt = obatCursor[p];
			buffObjs[numP] = obt; 
			#endif
			numP++; 
			curP = pbt; 

		}
	
		numPwithDup++;

		
	}
	
	#if EXTRAINFO_FROM_RDFTYPE
	if (numTypeValues > 1)
		getBestRdfTypeValue(buff, numP, rdftypeOntologyValues, rdftypeSelectedValues, rdftypeSpecificLevels, rdftypeOntClassPos, &numTypeValues, tmpMaxSpecificLevel, tfidfInfos);
	#endif	
	/*put the last CS */
	#if STOREFULLCS
	returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, numTypeValues, rdftypeOntologyValues, &CSoid, 1, *freqThreshold, freqCSset, curS, buffObjs, fullPropStat, ontbat, buffOntologyNums, &totalNumOntology,numOnt); 
	#else
	returnCSid = putaCStoHash(csBats, buff, numP, numPwithDup, numTypeValues, rdftypeOntologyValues, &CSoid, 1, *freqThreshold, freqCSset, fullPropStat, ontbat, buffOntologyNums, &totalNumOntology,numOnt); 
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
	
	#if NO_OUTPUTFILE == 0
	printf("Total number of ontologies in explored CSs: %d (/ "BUNFMT" CSs) --> %f per CS\n",totalNumOntology,CSoid, (float) totalNumOntology/CSoid);
	GDKfree(buffOntologyNums);
	TKNZRclose(ret);
	#endif

	#if FULL_PROP_STAT
	#if NO_OUTPUTFILE == 0
	printPropStat(fullPropStat,1);
	#endif
	#endif	
	
	BBPreclaim(typeBat);
	#if EXTRAINFO_FROM_RDFTYPE
	freePropStat(ontPropStat);
	freeTFIDFInfo(tfidfInfos, numOntClass);
	#endif
	freePropStat(fullPropStat); 
	free(rdftypeOntologyValues);
	free(rdftypeSelectedValues);
	free(rdftypeSpecificLevels);
	free(rdftypeOntClassPos);

	*ret = 1; 

	//Update the numOrigFreqCS for freqCS
	freqCSset->numOrigFreqCS = freqCSset->numCSadded; 

	return MAL_SUCCEED; 
}

#if DETECT_INCORRECT_TYPE_SUBJECT

static 
str RDFcheckWrongTypeSubject(BAT *sbat, BATiter si, BATiter pi, BATiter oi, CSset *freqCSset, int maxNumPwithDup, int numTables, int *mTblIdxFreqIdxMapping, LabelStat *labelStat, oid *subjCSMap, int *csFreqCSMapping){

	BUN 	p, q; 
	oid 	*sbt, *pbt, *obt; 
	oid 	curS; 		/* current Subject oid */
	oid	redirectS; 	/* Subject that are redirected from the curS*/
	oid 	curP; 		/* current Property oid */
	int 	numP; 		/* Number of properties for current S */
	int 	numPwithDup = 0; 
	oid*	buff; 	 

	//Only keep the most specific ontology-based rdftype value 
	int	maxNumOntology = 20;		
	oid*	rdftypeOntologyValues = NULL; 
	char*   rdftypeSelectedValues = NULL; //Store which value is selected
	char* 	rdftypeSpecificLevels = NULL; //Store specific level for each value
	BUN*	rdftypeOntClassPos = NULL; //Position in the ontology class		     

	int	numTypeValues = 0;
	#if EXTRAINFO_FROM_RDFTYPE
	int	tmpMaxSpecificLevel = 0; 
	int	tmpSpecificLevel = 0; 
	BUN	tmpOntClassPos = BUN_NONE;  //index of the ontology class in the ontmetaBat
	PropStat        *ontPropStat = NULL;
	int		numOntClass = 0; 
	TFIDFInfo	*tfidfInfos = NULL;
	#endif
	PropStat 	*propStat = NULL; //Store the propStat for the prop of the freqCS
					//corresponding to the final table
	char		isFoundWrongTypeSubj = 0;
	char		isExist = 0;
	CS		*cs; 	
	int		tmpfreqIdx = -1; 

	BAT	*typeBat;	//BAT contains oids of type attributes retrieved from tokenizer
	oid	markedName = BUN_NONE;  
	int	i,j;
	BUN	bun, bunprop; 
	oid	prop;

	oid	*subjTypeMap = NULL; 
	oid	*maxSoid;


	(void) mTblIdxFreqIdxMapping;
	(void) numTables;



	#if EXTRAINFO_FROM_RDFTYPE
	numOntClass = BATcount(ontmetaBat);
	ontPropStat = initPropStat();
	ontPropStat = getPropStatisticsByOntologyClass(numOntClass, ontclassSet);
	tfidfInfos = (TFIDFInfo*)malloc(sizeof(TFIDFInfo) * numOntClass);
	initTFIDFInfosForOntologyClass(tfidfInfos, numOntClass, ontclassSet, ontPropStat);
	#endif


	typeBat = buildTypeOidBat();

	printf("Number of attributes inserted into BAT: " BUNFMT "\n", BATcount(typeBat));
	rdftypeOntologyValues = (oid*)malloc(sizeof(oid) * maxNumOntology);
	rdftypeSelectedValues = (char*)malloc(sizeof(char) * maxNumOntology);
	rdftypeSpecificLevels = (char*)malloc(sizeof(char) * maxNumOntology);
	rdftypeOntClassPos = (BUN *) malloc(sizeof(BUN) * maxNumOntology);
		
	buff = (oid *) malloc (sizeof(oid) * (maxNumPwithDup + 1));

	numP = 0;
	curP = BUN_NONE; 
	curS = 0; 
	
	#if USING_FINALTABLE
	{
	int	numdistinctMCS = 0;
	propStat =  getPropStatisticsByTable(numTables, mTblIdxFreqIdxMapping, freqCSset,  &numdistinctMCS);
	}
	#else
	{
	int curNumMergeCS = countNumberMergeCS(freqCSset);
	oid* mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
        initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);
	propStat = initPropStat();
	getPropStatisticsFromMergeCSs(propStat, curNumMergeCS, mergeCSFreqCSMap, freqCSset);
	}
	#endif

	maxSoid = (BUN *) Tloc(sbat, BUNlast(sbat) - 1);

	assert(*maxSoid != BUN_NONE); 

	subjTypeMap = (oid *) malloc (sizeof(oid) * ((*maxSoid) + 1)); 
	initArray(subjTypeMap, (*maxSoid) + 1, BUN_NONE);

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				if (numTypeValues > 1){
					getBestRdfTypeValue(buff, numP, rdftypeOntologyValues, rdftypeSelectedValues, rdftypeSpecificLevels, rdftypeOntClassPos, &numTypeValues, tmpMaxSpecificLevel, tfidfInfos);
					
					//Only check for subject that have type value
					markedName = rdftypeOntologyValues[0];

					assert(markedName != BUN_NONE);
					subjTypeMap[curS] = markedName;

					bun = BUNfnd(labelStat->labelBat, &markedName); 
					if (bun != BUN_NONE){	//There is table to compare			
						int freqId = csFreqCSMapping[subjCSMap[curS]];
						if (freqId == -1) {
							printf("An infrequent CS subject "BUNFMT" marked with type "BUNFMT"\n",curS, markedName);
						}
						isFoundWrongTypeSubj = 0;
						for (i = 0; i < numP; i++){
							//Check each prop
							isExist = 0;
							prop = buff[i];
							bunprop = BUNfnd(propStat->pBat,(ptr) &prop);
							if (bunprop == BUN_NONE){
								printf("Subj "BUNFMT" of type "BUNFMT" has an prop "BUNFMT" not in any final table \n", 
																curS,markedName, prop);
								isFoundWrongTypeSubj = 1;
							}
							else{ 
								if (propStat->tfidfs[bunprop] > MIN_TFIDF_PROP_FINALTABLE){ //A discriminate prop
								//if (propStat->tfidfs[bunprop] > 0){ //A discriminate prop
									//check if any table has this prop
									for (i = 0; i < labelStat->lstCount[bun]; i++){
										//Check table i
										#if USING_FINALTABLE
										tmpfreqIdx = mTblIdxFreqIdxMapping[labelStat->freqIdList[bun][i]];
										#else	
										tmpfreqIdx = labelStat->freqIdList[bun][i];
										#endif
										cs = (CS*) &(freqCSset->items[tmpfreqIdx]);
										for (j = 0; j < cs->numProp; j++){
											if (cs->lstProp[j] == prop){
												isExist = 1; 
											}
											if (cs->lstProp[j] > prop) break; //don't need to check after
										}
										if (isExist == 1) break; 
									}
									if (isExist == 0){
										printf("Subj "BUNFMT" of type "BUNFMT" has an prop "BUNFMT" not belong to table of the same typee \n", curS,markedName, prop);	
										isFoundWrongTypeSubj = 1;
									}
									
								}
									
							}
							if (isFoundWrongTypeSubj) break;
						}
					}

				}
				 
			}
			curS = *sbt; 
			curP = BUN_NONE;
			numP = 0;
			numPwithDup = 0; 

			numTypeValues = 0;
			#if EXTRAINFO_FROM_RDFTYPE
			tmpMaxSpecificLevel = 0;
			tmpSpecificLevel = 0;
			#endif
		}
				
		pbt = (oid *) BUNtloc(pi, p); 

		#if EXTRAINFO_FROM_RDFTYPE
		if (isTypeAttribute(*pbt, typeBat)){ //Check type attributes
			obt = (oid *) BUNtloc(oi, p);
			tmpSpecificLevel = getOntologySpecificLevel(*obt, &tmpOntClassPos);

			if (tmpOntClassPos != BUN_NONE){

				rdftypeSpecificLevels[numTypeValues] = tmpSpecificLevel;
				rdftypeOntClassPos[numTypeValues] = tmpOntClassPos;		
				rdftypeOntologyValues[numTypeValues] = *obt;
				rdftypeSelectedValues[numTypeValues] = 1;
				numTypeValues++;

				if (tmpSpecificLevel > tmpMaxSpecificLevel){	
					tmpMaxSpecificLevel = tmpSpecificLevel;
				}
			}

		}
		#endif
		
		if (curP != *pbt){	/* Multi values property */		
			buff[numP] = *pbt; 
			numP++; 
			curP = *pbt; 
		}
		numPwithDup++;
	}
	
	#if EXTRAINFO_FROM_RDFTYPE
	if (numTypeValues > 1)
		getBestRdfTypeValue(buff, numP, rdftypeOntologyValues, rdftypeSelectedValues, rdftypeSpecificLevels, rdftypeOntClassPos, &numTypeValues, tmpMaxSpecificLevel, tfidfInfos);
	#endif	
	/*put the last CS */
	//TODO: Check the last subject, copy from above
	
	
	//Run again and check if a subj and it pageRedirect has the same type
	{
	char*   schema = "rdf";
	int	ret = 0;
	oid redirectAttributeOid = BUN_NONE; 
	char* redirectAttributes = "<http://dbpedia.org/ontology/wikiPageRedirects>";
	
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

	TKNZRappend(&redirectAttributeOid,&redirectAttributes);

	assert(redirectAttributeOid != BUN_NONE);

	printf("<http://dbpedia.org/ontology/wikiPageRedirects> id is "BUNFMT"\n",redirectAttributeOid);

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
					//Only check for subject that have type value
		pbt = (oid *) BUNtloc(pi, p); 

		if (*pbt == redirectAttributeOid && subjTypeMap[*sbt] != BUN_NONE){ //Check redirect value
			obt = (oid *) BUNtloc(oi, p);
			redirectS = *obt; 
			if (redirectS < *maxSoid){
				if (subjTypeMap[redirectS] != BUN_NONE){
					if (subjTypeMap[*sbt] != subjTypeMap[redirectS]){
						str curSstr; 
						str redirectSstr; 
						str curStype;
						str redirecttype;
						takeOid(*sbt, &curSstr);
						takeOid(redirectS, &redirectSstr);
						takeOid(subjTypeMap[*sbt], &curStype);
						takeOid(subjTypeMap[redirectS],&redirecttype);
						printf("Subject %s [Type: %s] redirects to %s [Type: %s]",
								curSstr,curStype,redirectSstr,redirecttype);
						GDKfree(curSstr);
						GDKfree(redirectSstr);
						GDKfree(curStype);
						GDKfree(redirecttype);
						
						if (isSupSuperOntology(subjTypeMap[*sbt],subjTypeMap[redirectS]) == 0){
							printf (" [NOT IN SAME HIERARCHY] \n");
						} else {
							printf ("\n");
						}
					}
				
				}
			}
		}
	}

	TKNZRclose(&ret);	
	}

	free (buff); 
	
	free(subjTypeMap);

	#if EXTRAINFO_FROM_RDFTYPE
	freePropStat(ontPropStat);
	freeTFIDFInfo(tfidfInfos, numOntClass);
	#endif
	freePropStat(propStat);
	free(rdftypeOntologyValues);
	free(rdftypeSelectedValues);
	free(rdftypeSpecificLevels);
	free(rdftypeOntClassPos);

	BBPreclaim(typeBat);
	

	return MAL_SUCCEED; 
}

#endif

static 
str RDFgetRefCounts(int *ret, BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, int maxNumProp, BUN maxSoid, int *refCount){

	BUN 		p, q; 
	oid 		*sbt, *pbt, *obt; 
	oid 		curS; 		/* current Subject oid */
	oid 		curP; 		/* current Property oid */
	int 		numP; 		/* Number of properties for current S */
	oid*		buff; 	 

	ObjectType	objType;
	oid		realObjOid; 	

	buff = (oid *) malloc (sizeof(oid) * maxNumProp);

	numP = 0;
	curP = BUN_NONE; 
	curS = 0; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			curS = *sbt; 
			curP = BUN_NONE;
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


pthread_mutex_t lock;

//This function insert the relationships for freqCS modularly by the thread id
//
#if NEEDSUBCS
static 
str addRels_from_a_partition(int tid, int nthreads, int first, int last, BAT *sbat, BAT *pbat, BAT *obat, 
		oid *subjCSMap, oid *subjSubCSMap, SubCSSet *csSubCSSet, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup,int *csIdFreqIdxMap){
#else
static
str addRels_from_a_partition(int tid, int nthreads, int first, int last, BAT *sbat, BAT *pbat, BAT *obat,
		oid *subjCSMap, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup,int *csIdFreqIdxMap){
#endif	
	
	oid 		sbt = 0, obt, pbt;
	oid 		curS; 		/* current Subject oid */
	int 		numPwithDup;	/* Number of properties for current S */
	ObjectType	objType;
	#if NEEDSUBCS
	oid 		returnSubCSid; 
	#endif
	char* 		buffTypes; 
	oid		realObjOid; 	
	char 		isBlankNode; 
	oid		curP;
	int		from, to; 
	
	oid		*sbatCursor = NULL, *pbatCursor = NULL, *obatCursor = NULL; 
	int		p;
	
	(void) tid;
	(void ) nthreads;

	buffTypes = (char *) malloc(sizeof(char) * (maxNumPwithDup + 1)); 

	numPwithDup = 0;
	curS = 0; 
	curP = BUN_NONE; 

	sbatCursor = (oid *) Tloc(sbat, BUNfirst(sbat));
	pbatCursor = (oid *) Tloc(pbat, BUNfirst(pbat));
	obatCursor = (oid *) Tloc(obat, BUNfirst(obat));


	for (p = first; p <= last; p++){
		sbt = sbatCursor[p];		
		from = csIdFreqIdxMap[subjCSMap[sbt]];
		#if GETSUBCS_FORALL == 0
		if ( from == -1) continue; /* Do not consider infrequentCS */
		#endif

		#if USEMULTITHREAD
		if (from % nthreads != tid) continue; 
		#endif

		if (sbt != curS){
			#if NEEDSUBCS
			if (p != 0){	/* Not the first S */
				returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[curS], csSubCSSet);

				//Get the subCSId
				subjSubCSMap[curS] = returnSubCSid; 

			}
			#endif
			curS = sbt; 
			numPwithDup = 0;
			curP = BUN_NONE; 
		}
				
		pbt = pbatCursor[p];

		obt = obatCursor[p]; 
		/* Check type of object */
		objType = getObjType(obt);

		/* Look at the referenced CS Id using subjCSMap */
		isBlankNode = 0;
		//if (objType == URI || objType == BLANKNODE){
		if ((objType == URI || objType == BLANKNODE) && from != -1){
			realObjOid = (obt) - ((oid) objType << (sizeof(BUN)*8 - 4));

			/* Only consider references to freqCS */	
			if (realObjOid <= maxSoid && subjCSMap[realObjOid] != BUN_NONE && csIdFreqIdxMap[subjCSMap[realObjOid]] != -1){
				to = csIdFreqIdxMap[subjCSMap[realObjOid]];
				if (objType == BLANKNODE) isBlankNode = 1;

				//pthread_mutex_lock(&lock);
				addReltoCSRel(from, to, pbt, &csrelSet[from], isBlankNode);
				//pthread_mutex_unlock(&lock);
			}
		}

		if (curP == pbt){
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
			curP = pbt; 
		}
	}
	
	#if NEEDSUBCS
	/* Check for the last CS */
	returnSubCSid = addSubCS(buffTypes, numPwithDup, subjCSMap[sbt], csSubCSSet);
	subjSubCSMap[sbt] = returnSubCSid; 
	#endif

	free (buffTypes); 

	return MAL_SUCCEED; 
}

#if USEMULTITHREAD

static void*
addRels_Thread(void *arg_p){
	
	csRelThreadArg *arg = (csRelThreadArg*) arg_p;
	
	printf("Start thread %d \n", arg->tid);

	#if NEEDSUBCS
	addRels_from_a_partition(arg->tid, arg->nthreads, arg->first, arg->last, arg->sbat, arg->pbat, arg->obat, arg->subjCSMap, arg->subjSubCSMap, arg->csSubCSSet, arg->csrelSet, arg->maxSoid, arg->maxNumPwithDup, arg->csIdFreqIdxMap);
	#else
	addRels_from_a_partition(arg->tid, arg->nthreads, arg->first, arg->last, arg->sbat, arg->pbat, arg->obat, arg->subjCSMap, arg->csrelSet, arg->maxSoid, arg->maxNumPwithDup, arg->csIdFreqIdxMap);
	#endif

	pthread_exit(NULL);
}
#endif

#if NEEDSUBCS
static 
str RDFrelationships(int *ret, BAT *sbat, BAT *pbat, BAT *obat, 
		oid *subjCSMap, oid *subjSubCSMap, SubCSSet *csSubCSSet, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup,int *csIdFreqIdxMap, int numFreqCS){
#else
static
str RDFrelationships(int *ret, BAT *sbat, BAT *pbat, BAT *obat,
		oid *subjCSMap, CSrel *csrelSet, BUN maxSoid, int maxNumPwithDup,int *csIdFreqIdxMap, int numFreqCS){
#endif	
	
	#if USEMULTITHREAD
	int 		i, first, last; 
	//oid 		*sbatCursor = NULL; 
	csRelThreadArg 	*threadArgs = NULL;
	pthread_t 	*threads = NULL; 
	int 		nthreads = NUMTHEAD_CSREL;
	int		ntp = 0; 	//Number of triples per partition
	//int		tmplast =0; 
	

	/*
	if (pthread_mutex_init(&lock, NULL) != 0)
	{
		throw (MAL, "rdf.RDFrelationships", "Failed to create threads mutex");
	}
	*/

	if (numFreqCS < 100){		//Don't use multi thread with small number of cs rels
		nthreads = 1;
	}

	first = 0; 
	last = BATcount(sbat) -1; 
	ntp = (last + 1)/nthreads;
	
	printf("Number of triples per partition %d\n", ntp);

	//sbatCursor = (oid *) Tloc(sbat, BUNfirst(sbat)); 
	threadArgs = (csRelThreadArg *) GDKmalloc(sizeof(csRelThreadArg) * nthreads);
	threads = (pthread_t *) GDKmalloc(sizeof(pthread_t) * nthreads);
	
	for (i = 0; i < nthreads; i++){
		threadArgs[i].tid = i; 
		
		/* Old code for partitioning the BAT of subjects
		threadArgs[i].first = (i == 0) ? first:(threadArgs[i-1].last + 1);
		tmplast = (i == (nthreads - 1))?last: (ntp * (i + 1));
		//Go to all the triples of the current subjects
		while ((i < (nthreads - 1)) 
			&& (sbatCursor[tmplast] == sbatCursor[tmplast+1])){
			tmplast++;
		}
		threadArgs[i].last = tmplast;
		*/
		threadArgs[i].first = first; 
		threadArgs[i].last = last;
		threadArgs[i].nthreads = nthreads;

		threadArgs[i].sbat = sbat;
		threadArgs[i].pbat = pbat;
		threadArgs[i].obat = obat;
		threadArgs[i].subjCSMap = subjCSMap;
		#if NEEDSUBCS
		threadArgs[i].subjSubCSMap = subjSubCSMap;
		threadArgs[i].csSubCSSet = csSubCSSet;
		#endif
		threadArgs[i].csrelSet = csrelSet;
		threadArgs[i].maxSoid = maxSoid;
		threadArgs[i].maxNumPwithDup = maxNumPwithDup;
		threadArgs[i].csIdFreqIdxMap = csIdFreqIdxMap;
			
	}


	for (i = 0; i < nthreads; i++) {
		 if (pthread_create(&threads[i], NULL, addRels_Thread, &threadArgs[i])) {
			GDKfree(threadArgs);
			GDKfree(threads);
			throw (MAL, "rdf.RDFrelationships", "Failed to creat threads %d",i);
		 }
	}

	for (i = 0; i < nthreads; i++) {
		if (pthread_join(threads[i], NULL)) {
			GDKfree(threadArgs);
			GDKfree(threads);
			throw (MAL, "rdf.RDFrelationships", "Failed to join threads %d",i);
		}
	}
	
	//pthread_mutex_destroy(&lock);
	GDKfree(threadArgs);
	GDKfree(threads);

	#else 	//NOT USEMULTITHREAD

	(void) numFreqCS;
	#if NEEDSUBCS
	addRels_from_a_partition(0, 1, 0, BATcount(sbat) - 1, sbat, pbat, obat, subjCSMap, subjSubCSMap, csSubCSSet, csrelSet, maxSoid, maxNumPwithDup, csIdFreqIdxMap);
	#else
	addRels_from_a_partition(0, 1, 0, BATcount(sbat) - 1, sbat, pbat, obat, subjCSMap, csrelSet, maxSoid, maxNumPwithDup, csIdFreqIdxMap);
	#endif

	#endif 	/* USEMULTITHREAD */

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
str RDFExtractCSPropTypes(int *ret, BAT *sbat, BAT *pbat, BAT *obat,  
		oid *subjCSMap, int* csTblIdxMapping, CSPropTypes* csPropTypes, int maxNumPwithDup){

	oid 		sbt , obt, pbt;
	oid 		curS; 		/* current Subject oid */
	//oid 		CSoid = 0; 	/* Characteristic set oid */
	int 		numPwithDup;	/* Number of properties for current S */
	int*		buffCoverage;	/* Number of triples coverage by each property. For deciding on MULTI-VALUED P */
	ObjectType	objType;
	char* 		buffTypes; 
	int		**buffTypesCoverMV; /*Store the types of each value in a multi-value prop */		
	oid*		buffP;
	oid		curP; 
	int 		i, p;

	oid		*sbatCursor = NULL, *pbatCursor = NULL, *obatCursor = NULL; 
	int		first, last;

	buffTypes = (char *) malloc(sizeof(char) * (maxNumPwithDup + 1)); 
	buffTypesCoverMV = (int **)malloc(sizeof(int*) * (maxNumPwithDup + 1));
	for (i = 0; i < (maxNumPwithDup + 1); i++){
		buffTypesCoverMV[i] = (int *) malloc(sizeof(int) * (MULTIVALUES)); 
	}
	buffP = (oid *) malloc(sizeof(oid) * (maxNumPwithDup + 1));
	buffCoverage = (int *)malloc(sizeof(int) * (maxNumPwithDup + 1));

	numPwithDup = 0;
	curS = 0; 
	curP = BUN_NONE; 

	sbatCursor = (oid *) Tloc(sbat, BUNfirst(sbat));
	pbatCursor = (oid *) Tloc(pbat, BUNfirst(pbat));
	obatCursor = (oid *) Tloc(obat, BUNfirst(obat));

	first = 0; 
	last = BATcount(sbat) -1; 
	
	for (p = first; p <= last; p++){
		sbt = sbatCursor[p];
		
		if (sbt != curS){
			if (p != 0){	/* Not the first S */
				addPropTypes(buffTypes, buffP, numPwithDup, buffCoverage, buffTypesCoverMV, subjCSMap[curS], csTblIdxMapping, csPropTypes);
			}
			curS = sbt; 
			numPwithDup = 0;
			curP = BUN_NONE; 
		}
				
		obt = obatCursor[p];
		/* Check type of object */
		objType = getObjType(obt);	/* Get two bits 63th, 62nd from object oid */
		
		if (objType == BLANKNODE){	//BLANKNODE object values will be stored in the same column with URI object values	
			objType = URI; 
		}

		pbt = pbatCursor[p];

		if (curP == pbt){
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
			buffP[numPwithDup] = pbt;
			buffCoverage[numPwithDup] = 1; 
			numPwithDup++; 
			curP = pbt; 
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
void printNumTypePerProp(CSPropTypes* csPropTypes, int numCS, CSset *freqCSset){
	
	int 	i,j,k; 
	CS	cs;
	int	tmpNumType = 0; 
	int	tmpSumNumType = 0; 	
	int	totalNumProp = 0; 
	int	totalNumTypes = 0; 
	int	numMultiTypeProp = 0;
	int	tmpnumMultiTypeProp = 0;
	int	numPropByNumType[MULTIVALUES];
	FILE	*fout;


	for (k = 0; k < MULTIVALUES; k++){
		numPropByNumType[k] = 0;
	}
	fout = fopen("csPropTypeBasicCS.txt","wt");
	fprintf(fout, "#FreqCSId	#NumProp #NumTypes #AvgNumType/Prop  #MultiTypeProp \n");
	for (i = 0; i < numCS; i++){
	 	cs = freqCSset->items[i];
		tmpSumNumType = 0; 
		tmpnumMultiTypeProp = 0;
		for (j = 0; j < cs.numProp; j++){
				tmpNumType = 0;
				for (k = 0; k < MULTIVALUES; k++){
					if (csPropTypes[i].lstPropTypes[j].lstFreqWithMV[k] > 0){
						tmpNumType++;
					}
				}
				tmpSumNumType += tmpNumType;
				numPropByNumType[tmpNumType]++; 
				if (tmpNumType > 1) tmpnumMultiTypeProp++;
		}
		
		fprintf(fout, "%d	%d	%d	%.2f	%d \n",i,cs.numProp,tmpSumNumType, (float) tmpSumNumType/cs.numProp,tmpnumMultiTypeProp);
		totalNumProp += cs.numProp;
		totalNumTypes += tmpSumNumType;
		numMultiTypeProp += tmpnumMultiTypeProp;
	}

	printf("Average number of types per prop in freqCS: %f \n", (float)totalNumTypes/totalNumProp);
	printf("Number of multi-types prop %d \n",numMultiTypeProp);
	for (k = 0; k < MULTIVALUES; k++){
		printf("Number prop with %d types: %d \n",k, numPropByNumType[k]);
	}

	fclose(fout); 
}

#if NO_OUTPUTFILE == 0 
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
#endif

#if NO_OUTPUTFILE == 0 
static
void getSubjIdFromTablePosition(int tblIdx, int pos, oid *sOid){
	oid id; 
	id = pos;
	id |= (BUN)(tblIdx + 1) << (sizeof(BUN)*8 - NBITS_FOR_CSID);
	*sOid = id; 
}

static
str getOrigSbt(oid *sbt, oid *origSbt, BAT *lmap, BAT *rmap){
	BUN pos; 
	oid *tmp; 
	pos = BUNfnd(rmap,sbt);
	if (pos == BUN_NONE){
		throw(RDF, "rdf.RDFdistTriplesToCSs", "This encoded subject must be in rmap");
	}
	tmp = (oid *) Tloc(lmap, pos);
	if (*tmp == BUN_NONE){
		throw(RDF, "rdf.RDFdistTriplesToCSs", "The encoded subject must be in lmap");
	}

	*origSbt = *tmp; 		

	return MAL_SUCCEED; 
}

static
str getOrigObt(oid *obt, oid *origObt, BAT *lmap, BAT *rmap){
	BUN pos; 
	oid *tmp; 
	oid	tmporigOid = BUN_NONE; 
	ObjectType objType; 
	BUN	maxObjectURIOid =  ((oid)1 << (sizeof(BUN)*8 - NBITS_FOR_CSID - 1)) - 1; //Base on getTblIdxFromS

	objType = getObjType(*obt); 

	if (objType == URI || objType == BLANKNODE){
		tmporigOid = (*obt) - ((oid)objType << (sizeof(BUN)*8 - 4));
	}
	
	if (tmporigOid > maxObjectURIOid){
		pos = BUNfnd(rmap,&tmporigOid);
		if (pos == BUN_NONE){
			throw(RDF, "rdf.RDFdistTriplesToCSs", "This encoded object must be in rmap");
		}
		tmp = (oid *) Tloc(lmap, pos);
		if (*tmp == BUN_NONE){
			throw(RDF, "rdf.RDFdistTriplesToCSs", "The encoded object must be in lmap");
		}

		*origObt = *tmp; 		
	}
	else{
		*origObt = tmporigOid;
	}

	return MAL_SUCCEED; 
}
#endif

static
oid getFirstEncodedSubjId(int tblIdx){
	
	return (BUN)(tblIdx + 1) << (sizeof(BUN)*8 - NBITS_FOR_CSID);
}

//Encoded subject BAT contains 
//sequential numbers from getFirstEncodedSubjId()
//to getFirstEncodedSubjId() + numberofelements 

BAT* createEncodedSubjBat(int tblIdx, int num){
	BAT* subjBat = NULL; 
	
	subjBat = BATnew(TYPE_void, TYPE_void , num + 1, TRANSIENT);
	BATsetcount(subjBat,num);
	BATseqbase(subjBat, 0);
	BATseqbase(BATmirror(subjBat), getFirstEncodedSubjId(tblIdx));

	subjBat->T->nonil = 1;
	subjBat->tkey = 0;
	subjBat->tsorted = 1;
	subjBat->trevsorted = 0;
	subjBat->tdense = 1;

	return subjBat; 
}

#if NO_OUTPUTFILE == 0
static
char getObjTypeFromBATtype(int battype){
	
	if (battype == TYPE_timestamp){	//This type is not constant
		return DATETIME; 
	}

	switch (battype)						  
	{
		case TYPE_oid:
			return URI;
			break;
		case TYPE_str:
			return STRING; 
			break;
		case TYPE_int:
			return INTEGER;
			break;
		case TYPE_dbl:
			return DOUBLE;
			break;
		default:
			return 100;
			break; 
	}
}

static
int getObjValueFromMVBat(ValPtr returnValue, ValPtr castedValue, BUN pos, ObjectType objType, BAT *tmpBat, BAT *lmap, BAT *rmap){
	str	tmpStr; 
	str	inputStr; 
	double	*realDbl; 
	int	*realInt; 
	oid	*tmpUriOid; 
	oid	realUriOid = BUN_NONE;
	BATiter	tmpi; 
	timestamp *ts; 

	tmpi = bat_iterator(tmpBat);

	switch (objType)
	{
		case STRING:
			//printf("A String object value: %s \n",objStr);
			tmpStr = BUNtail(tmpi, pos); 
			if (strcmp(tmpStr,str_nil) != 0){
				// remove quotes and language tags
				str tmpStrShort;
				getStringBetweenQuotes(&tmpStrShort, tmpStr);
				inputStr = GDKmalloc(sizeof(char) * strlen(tmpStrShort) + 1); 
				memcpy(inputStr, tmpStrShort, sizeof(char) * strlen(tmpStrShort) + 1);
				GDKfree(tmpStrShort);
				VALset(returnValue, TYPE_str, inputStr);
				if (rdfcast(objType, STRING, returnValue, castedValue) != 1){
					printf("Everything should be able to cast to String \n");
				}

				return 1;
			}
			else{
				return 0;
			}
			break; 
		case DATETIME:
			//printf("A Datetime object value: %s \n",objStr);
			ts = (timestamp *) BUNtail(tmpi, pos);
			if (!ts_isnil(*ts)){
			        int lenbuf = 128;
			        char buf[128], *s1 = buf;
			        *s1 = 0;

				timestamp_tostr(&s1, &lenbuf, (const timestamp *)ts);
				inputStr = GDKmalloc(sizeof(char) * strlen(s1) + 1);
				memcpy(inputStr, s1, sizeof(char) * strlen(s1) + 1);
				VALset(castedValue, TYPE_str, inputStr);

				return 1;
			}
			else{
				return 0; 
			}
			break; 
		case INTEGER:
			realInt = (int *) BUNtail(tmpi, pos);
			if (*realInt != int_nil){
				VALset(returnValue, TYPE_int, realInt);
				if (rdfcast(objType, STRING, returnValue, castedValue) != 1){
					printf("Everything should be able to cast to String \n");
				}
				return 1;
			}
			else{
				return 0;
			}
			break; 
		case DOUBLE:
			//printf("Full object value: %s \n",objStr);
			realDbl = (double *)BUNtail(tmpi, pos);
			if (*realDbl != dbl_nil){
				VALset(returnValue, TYPE_dbl, realDbl);
				if (rdfcast(objType, STRING, returnValue, castedValue) != 1){
					printf("Everything should be able to cast to String \n");
				}

				return 1; 
			}
			else{
				return 0; 
			}
			break; 
		default: //URI or BLANK NODE		
			tmpUriOid = (oid *)BUNtail(tmpi, pos);
			if (*tmpUriOid != oid_nil){
				str tmpUriStr; 
				str tmpShortUriStr; 
				if (getOrigObt(tmpUriOid, &realUriOid, lmap, rmap) != MAL_SUCCEED){
					printf("[ERROR] Problem in getting the orignal obt \n");
					return -1;
				}
				
				takeOid(realUriOid,&tmpUriStr);
				getPropNameShort(&tmpShortUriStr, tmpUriStr);

				VALset(returnValue,TYPE_str, tmpShortUriStr);
				if (rdfcast(STRING, STRING, returnValue, castedValue) != 1){
					printf("Everything should be able to cast to String \n");
				}

				//GDKfree(tmpShortUriStr);
				GDKfree(tmpUriStr);

				return 1; 
			}
			else{
				return 0; 
			}

			break; 
	}



}

static 
str initFullSampleData(CSSampleExtend *csSampleEx, int *mTblIdxFreqIdxMapping, CSlabel *label, CStableStat* cstablestat, CSPropTypes *csPropTypes, CSset *freqCSset, int numTables,  bat *lmapbatid, bat *rmapbatid){
	int 	i, j, k; 
	int	freqId; 
	int	tmpNumcand; 
	oid	tmpCandidate; 
	int 	randValue = 0; 
	int	ranPosition = 0; 	//random position of the instance in a table
	int	tmpNumCols; 
	int 	colIdx;
	int	mvColIdx; 
	int	tmpNumMVCols;
	char	tmpObjType; 
	ValRecord       vrRealObjValue;
	ValRecord       vrCastedObjValue;
	BUN	tmpPos = BUN_NONE;

	BAT     *tmpbat = NULL;
	BATiter tmpi; 
	BAT	*cursamplebat = NULL; 
	int	tmpNumRows = 0; 
	oid	tmpSoid = BUN_NONE, origSoid = BUN_NONE;  
	oid	origOid = BUN_NONE; 
	BAT	*lmap = NULL, *rmap = NULL; 
	BAT     *tmpmvBat = NULL;       // Multi-values BAT
	BAT	*tmpmvKeyBat = NULL;	// KeyBat in MV table
	oid	mvRefOid;
	oid	*tmpmvRefOid = NULL;
	char*   schema = "rdf";
	int	ret = 0;

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}

	if ((lmap = BATdescriptor(*lmapbatid)) == NULL) {
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	
	if ((rmap = BATdescriptor(*rmapbatid)) == NULL) {
		BBPunfix(lmap->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	srand(123456); 
	for (i = 0; i < numTables; i++){
		freqId = mTblIdxFreqIdxMapping[i];
		csSampleEx[i].freqIdx = freqId;
		tmpNumcand = (NUM_SAMPLE_CANDIDATE > label[freqId].candidatesCount)?label[freqId].candidatesCount:NUM_SAMPLE_CANDIDATE;
		csSampleEx[i].name = cstablestat->lstcstable[i].tblname; 
		csSampleEx[i].candidateCount = tmpNumcand;
		csSampleEx[i].candidates = (oid*)malloc(sizeof(oid) * tmpNumcand); 
		csSampleEx[i].candidatesOrdered = (oid*)malloc(sizeof(oid) * tmpNumcand); 
		for (k = 0; k < tmpNumcand; k++){
			csSampleEx[i].candidates[k] = label[freqId].candidates[k]; 
			csSampleEx[i].candidatesOrdered[k] = label[freqId].candidates[k]; 
		}
		csSampleEx[i].candidatesNew = label[freqId].candidatesNew;
		csSampleEx[i].candidatesOntology = label[freqId].candidatesOntology;
		csSampleEx[i].candidatesType = label[freqId].candidatesType;
		csSampleEx[i].candidatesFK = label[freqId].candidatesFK;
		//Randomly exchange the value, change the position k with a random pos
		for (k = 0; k < tmpNumcand; k++){
			randValue = rand() % tmpNumcand;
			tmpCandidate = csSampleEx[i].candidates[k];
			csSampleEx[i].candidates[k] = csSampleEx[i].candidates[randValue];
			csSampleEx[i].candidates[randValue] = tmpCandidate;
		}

		csSampleEx[i].lstSubjOid = (oid*)malloc(sizeof(oid) * NUM_SAMPLE_INSTANCE);
		for (k = 0; k < NUM_SAMPLE_INSTANCE; k++)
			csSampleEx[i].lstSubjOid[k] = BUN_NONE; 

		tmpNumCols = csPropTypes[i].numProp -  csPropTypes[i].numInfreqProp; //already remove infrequent column;
		csSampleEx[i].numProp = tmpNumCols;
		
		assert(tmpNumCols > 0); 
			
		csSampleEx[i].lstProp = (oid*)malloc(sizeof(oid) * tmpNumCols); 
		csSampleEx[i].lstPropSupport = (int*)malloc(sizeof(int) * tmpNumCols); 
		csSampleEx[i].lstIsInfrequentProp = (char*)malloc(sizeof(char) * tmpNumCols); 
		csSampleEx[i].lstIsMVCol = (char*)malloc(sizeof(char) * tmpNumCols); 
		csSampleEx[i].colBats = (BAT**)malloc(sizeof(BAT*) * tmpNumCols);
		colIdx = -1;
		csSampleEx[i].numInstances = 0;
		for(j = 0; j < csPropTypes[i].numProp; j++){
			#if     REMOVE_INFREQ_PROP
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1)     continue;  //Infrequent prop
			#endif
			colIdx++;
			csSampleEx[i].lstProp[colIdx] = csPropTypes[i].lstPropTypes[j].prop;
			csSampleEx[i].lstPropSupport[colIdx] = csPropTypes[i].lstPropTypes[j].propFreq;

			if (csPropTypes[i].lstPropTypes[j].propFreq == 0) printf("[Verify] Empty Bat at table %d col %d  Prop "BUNFMT "\n",i,colIdx,csPropTypes[i].lstPropTypes[j].prop);
		
			//Mark whther this col is a MV col
			csSampleEx[i].lstIsMVCol[colIdx] = csPropTypes[i].lstPropTypes[j].isMVProp;
			
			//if this is a multivalue column, set the data type to string, combine all the values 
			//into a single value as a list
			
			if (csPropTypes[i].lstPropTypes[j].isMVProp){
				csSampleEx[i].colBats[colIdx] = BATnew(TYPE_void, TYPE_str , NUM_SAMPLE_INSTANCE + 1, TRANSIENT);
			} 
			else{
				csSampleEx[i].colBats[colIdx] = BATnew(TYPE_void, cstablestat->lstcstable[i].colBats[colIdx]->ttype , NUM_SAMPLE_INSTANCE + 1, TRANSIENT);
			}


			//Mark whether this col is infrequent sample cols
			if ( isInfrequentSampleCol(freqCSset->items[freqId], csPropTypes[i].lstPropTypes[j])){
				csSampleEx[i].lstIsInfrequentProp[colIdx] = 1;
			}
			else
				csSampleEx[i].lstIsInfrequentProp[colIdx] = 0;



		}
		assert(colIdx == (tmpNumCols - 1)); 

		
		// Inserting instances to csSampleEx
		
		tmpNumRows = BATcount(cstablestat->lstcstable[i].colBats[0]);
		
		for (k = 0; k < NUM_SAMPLE_INSTANCE; k++){
			ranPosition = rand() % tmpNumRows;

			getSubjIdFromTablePosition(i, ranPosition, &tmpSoid);	
			
			if (getOrigSbt(&tmpSoid, &origSoid, lmap, rmap) != MAL_SUCCEED){
				throw(RDF, "rdf.RDFdistTriplesToCSs","Problem in getting the orignal sbt ");
			} 

			csSampleEx[i].lstSubjOid[k] = origSoid;

			for (j = 0; j < tmpNumCols; j++){
				cursamplebat = csSampleEx[i].colBats[j];

				tmpbat = cstablestat->lstcstable[i].colBats[j]; 	
				tmpi = bat_iterator(tmpbat);

				if (csSampleEx[i].lstIsMVCol[j] == 1){ //Multi-value colum
					str 	tmpMVSampleStr = NULL; 	//sample string for MV values
					str	s; 

					int	curStrLen =0; 
					int	tmpStrLen =0; 
					oid *tmpOid = (oid *) BUNtail(tmpi, ranPosition);
					if (*tmpOid != oid_nil){

						//tmpOid refer to the keyBat of the mv bats

						//Get the range of multi-values in keyBat
						tmpmvKeyBat = cstablestat->lstcstable[i].lstMVTables[j].keyBat; 
						
						mvRefOid = *tmpOid;
						tmpmvRefOid = (oid *) Tloc(tmpmvKeyBat, mvRefOid);
						assert(tmpmvRefOid != NULL);
						
						//printf("First position for multivalues in keybat %d \n", (int) (*tmpmvRefOid));

						tmpNumMVCols = cstablestat->lstcstable[i].lstMVTables[j].numCol;
						//printf("Table %d colum %d is a mv col with %d types \n",i,j,tmpNumMVCols);
						
						tmpPos = *tmpOid;
						while (*tmpmvRefOid == mvRefOid){
							//Concat the data from each column
							for (mvColIdx =0; mvColIdx < tmpNumMVCols; mvColIdx++){
								tmpmvBat = cstablestat->lstcstable[i].lstMVTables[j].mvBats[mvColIdx];
								tmpObjType = getObjTypeFromBATtype(tmpmvBat->ttype); 
								if (getObjValueFromMVBat(&vrRealObjValue, &vrCastedObjValue, tmpPos, (ObjectType)tmpObjType, tmpmvBat, lmap, rmap) == 1){
									//printf("Casted value at mvBat %d is %s \n",mvColIdx,vrCastedObjValue.val.sval);
									tmpStrLen = strlen(vrCastedObjValue.val.sval);
									if (tmpMVSampleStr == NULL){ 
										tmpMVSampleStr = (str) GDKmalloc(tmpStrLen + 1);
										s = tmpMVSampleStr;
									}else{
										tmpMVSampleStr = (str) GDKrealloc(tmpMVSampleStr, curStrLen + tmpStrLen + 2);
										s = tmpMVSampleStr;
										s += curStrLen;
									}
									
									strcpy(s, vrCastedObjValue.val.sval);
									s += tmpStrLen;
									*s++ = ';';
									*s = '\0';

									curStrLen = strlen(tmpMVSampleStr);
									//printf("Current tmpMVSampleStr String %s --> curLen = %d \n",tmpMVSampleStr, curStrLen);

									VALclear(&vrCastedObjValue);
									VALclear(&vrRealObjValue);
								}
							}
							

							//Get next 
							tmpPos++;
							if (tmpPos == BATcount(tmpmvKeyBat)) break; 

							tmpmvRefOid = (oid *) Tloc(tmpmvKeyBat, tmpPos);
						}

					}
					//else{
						//printf("[Null] There is no set of multiple values for this subject");
					
					//}
					if (tmpMVSampleStr != NULL){
						tmpMVSampleStr = (str) GDKrealloc(tmpMVSampleStr, curStrLen + 1);
						tmpMVSampleStr[curStrLen] = '\0';
					}

					//printf("Final MV string : %s \n",tmpMVSampleStr);
					if (tmpMVSampleStr != NULL){
						BUNappend(cursamplebat, tmpMVSampleStr, TRUE);
						GDKfree(tmpMVSampleStr);
					}
					else
						BUNappend(cursamplebat, ATOMnilptr(TYPE_str), TRUE);

				}
				else if (tmpbat->ttype == TYPE_oid){
					//Get the original object oid
					oid *tmpOid = (oid *) BUNtail(tmpi, ranPosition);
					if(*tmpOid != oid_nil){
						if (getOrigObt(tmpOid, &origOid, lmap, rmap) != MAL_SUCCEED){
							throw(RDF, "rdf.RDFdistTriplesToCSs","Problem in getting the orignal obt ");
						}
						BUNappend(cursamplebat, &origOid, TRUE);
					}
					else{
						BUNappend(cursamplebat, ATOMnilptr(TYPE_oid), TRUE);
					}

				}
				else
					BUNappend(cursamplebat, BUNtail(tmpi, ranPosition), TRUE);


				
			}
			csSampleEx[i].numInstances++;
		}
		
		/*
		if (i == 0)
			for (j = 0; j < tmpNumCols; j++){
				//BATprint(cstablestat->lstcstable[i].colBats[j]);
				BATprint(csSampleEx[i].colBats[j]);
			}
			*/
		
	}
	
	TKNZRclose(&ret);
	BBPunfix(lmap->batCacheid);
	BBPunfix(rmap->batCacheid);

	return MAL_SUCCEED;

}
#endif

#if NO_OUTPUTFILE == 0 
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
#endif

#if NO_OUTPUTFILE == 0
static 
void freeSampleExData(CSSampleExtend *csSampleEx, int numCand){
	int i, j; 
	for (i = 0; i < numCand; i++){
		free(csSampleEx[i].lstProp);
		free(csSampleEx[i].lstPropSupport);
		free(csSampleEx[i].lstIsInfrequentProp);
		free(csSampleEx[i].lstIsMVCol);
		free(csSampleEx[i].candidates); 
		free(csSampleEx[i].candidatesOrdered); 
		free(csSampleEx[i].lstSubjOid);
		for (j = 0; j < csSampleEx[i].numProp; j++){
			BBPunfix(csSampleEx[i].colBats[j]->batCacheid);
		}
		free(csSampleEx[i].colBats);
	}

	free(csSampleEx);
}
#endif


#if NO_OUTPUTFILE == 0 
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
#endif


#if NO_OUTPUTFILE == 0 
/*
static 
void getObjStr(BAT *mapbat, BATiter mapi, oid objOid, str *objStr, char *retObjType){

	BUN bun; 

	ObjectType objType = getObjType(objOid); 

	if (objType == URI || objType == BLANKNODE){
		objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
		takeOid(objOid, objStr); 
	}
	else{
		objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   // Get the real objOid from Map or Tokenizer  
		bun = BUNfirst(mapbat);
		*objStr = (str) BUNtail(mapi, bun + objOid); 
	}

	*retObjType = objType; 

}
*/
#endif

//Assume Tokenizer is openned 
//
/*
void getSqlName(char *name, oid nameId){
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
*/

void getSqlName(str *name, oid nameId, BATiter mapi, BAT *mbat){
	str canStr = NULL; 
	str canStrShort = NULL;
	char    *pch;
	int 	lngth = 0; 
	str	s;
	int	i; 

	if (nameId != BUN_NONE){
		//takeOid(nameId, &canStr);
		getStringName(nameId, &canStr, mapi, mbat, 1);
		assert(canStr != NULL); 
		getPropNameShort(&canStrShort, canStr);

		if (strstr (canStrShort,".") != NULL || 
			strcmp(canStrShort,"") == 0 || 
			strstr(canStrShort,"-") != NULL	){	// WEBCRAWL specific problem with Table name a.jpg, b.png....
	
			lngth = 6;
			*name = (str)GDKmalloc(sizeof(char) * lngth + 1);
			s = *name;
			strcpy(s,"noname");
			s += lngth;
			*s = '\0';
		}
		else {
			pch = strstr (canStrShort,"(");
			if (pch != NULL) *pch = '\0';	//Remove (...) characters from table name
			
			lngth = strlen(canStrShort);
			*name = (str)GDKmalloc(sizeof(char) * lngth + 1);
			s = *name; 
			strcpy(s,canStrShort);

			for (i = 0; i < lngth; i++){
				//Convert to lower case
				/*
				if (s[i] >= 65 && s[i] <= 90){
					s[i] = s[i] | 32;
				}
				*/

				//Replace all non-alphabet character by ___
				if (!isalpha(s[i])){
					s[i] = '_';
				}
			}
			s += lngth;
			*s = '\0';
		}

		GDKfree(canStr);

		GDKfree(canStrShort); 
	}
	else {
		lngth = 6;
		*name = (str)GDKmalloc(sizeof(char) * lngth + 1);
		s = *name;
		strcpy(s,"noname");
		s += lngth;
		*s = '\0';
	}

}

#if NO_OUTPUTFILE == 0 
static 
str printSampleData(CSSample *csSample, CSset *freqCSset, BAT *mbat, int num, int sampleVersion){

	int 	i,j, k; 
	FILE 	*fout, *fouttb, *foutis; 
	char 	filename[100], filename2[100], filename3[100];
	char 	tmpStr[20], tmpStr2[20], tmpStr3[20];
	int 	ret;

	str 	propStr; 
	str	subjStr; 
	char*   schema = "rdf";
	CSSample	sample; 
	CS		freqCS; 
	int	numPropsInSampleTable;
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
	sprintf(tmpStr, "%d_v%d", num,sampleVersion);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");
	
	strcpy(filename2, "createSampleTable");
	sprintf(tmpStr2, "%d_v%d", num,sampleVersion);
	strcat(filename2, tmpStr2);
	strcat(filename2, ".sh");
	
	strcpy(filename3, "loadSampleToMonet");
	sprintf(tmpStr3, "%d_v%d", num,sampleVersion);
	strcat(filename3, tmpStr3);
	strcat(filename3, ".sh");
	
	fout = fopen(filename,"wt"); 
	fouttb = fopen(filename2,"wt");
	foutis = fopen(filename3,"wt");

	for (i = 0; i < num; i++){
		sample = csSample[i];
		freqCS = freqCSset->items[sample.freqIdx];
		fprintf(fout,"Table %d\n", i);
		for (j = 0; j < (int)sample.candidateCount; j++){
			//fprintf(fout,"  "  BUNFMT,sample.candidates[j]);
			if (sample.candidates[j] != BUN_NONE){
#if USE_SHORT_NAMES
				str canStrShort = NULL;
#endif
				//takeOid(sample.candidates[j], &canStr); 
				getStringName(sample.candidates[j], &canStr, mapi, mbat, 1);			
#if USE_SHORT_NAMES
				getPropNameShort(&canStrShort, canStr);
				if (j+1 == (int)sample.candidateCount) fprintf(fout, "%s",  canStrShort);
				else fprintf(fout, "%s;", canStrShort);
				GDKfree(canStrShort);
#else
				if (j+1 == (int)sample.candidateCount) fprintf(fout, "%s",  canStr);
				else fprintf(fout, "%s;", canStr);
#endif
				GDKfree(canStr); 
			
			}
		}
		fprintf(fout, "\n");
		

		if (sample.name != BUN_NONE){
			str canStrShort = NULL;
			//takeOid(sample.name, &canStr);
			getStringName(sample.name, &canStr, mapi, mbat, 1);
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

		//Number of tuples
		fprintf(fout, "%d\n", freqCS.support);

		//List of columns
		fprintf(fout,"Subject");
		fprintf(fouttb,"SubjectCol string");
		isTitle = 0; 
		isUrl = 0;
		isType = 0; 
		isDescription = 0; 
		isImage = 0;
		isSite = 0; 
		numPropsInSampleTable = (sample.numProp>NUM_PROP_SAMPLE)?(NUM_PROP_SAMPLE):sample.numProp;
		for (j = 0; j < numPropsInSampleTable; j++){
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
			
			for (j = 0; j < numPropsInSampleTable; j++){
				int index = j;
				objOid = sample.lstObj[index][k];
				if (objOid == BUN_NONE){
					fprintf(fout,";NULL");
					fprintf(foutis,"|NULL");
				}
				else{
					objStr = NULL;
					//getObjStr(mbat, mapi, objOid, &objStr, &objType);
					getStringName(objOid, &objStr, mapi, mbat, 0);
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
			//takeOid(sample.name, &canStr);
			getStringName(sample.name, &canStr, mapi, mbat, 1); 

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
#endif

#if NO_OUTPUTFILE == 0
static
void printPropertyWithMarkers(FILE *fout, str propStr, CSSampleExtend *csSampleEx, CSPropTypes *csPropTypes, int tblId, int propId, oid prop, BATiter mapi, BAT *mbat) {
	int origPropIdx = -1; 	//Index of prop in the freqCS which is used by PropTypes
	int i; 
	// print property string
	fprintf(fout, "%s", propStr);

	// add star (*) if multi-valued
	if (csSampleEx[tblId].lstIsMVCol[propId]) {
		fprintf(fout, "*");
	}
	
	for (i = 0; i < csPropTypes[tblId].numProp; i++){
		if (csPropTypes[tblId].lstPropTypes[i].prop == prop){
			origPropIdx = i;
			break;
		}
		
	}
	assert(origPropIdx != -1);
	// add reference (->) if FK
	if (csPropTypes[tblId].lstPropTypes[origPropIdx].isFKProp == 1) {
		str nameStr;
		int refTblId = csPropTypes[tblId].lstPropTypes[origPropIdx].refTblId;
		if (csSampleEx[refTblId].candidatesOrdered[0] != BUN_NONE) { // table name (= best candidate) available
#if USE_SHORT_NAMES
			str nameStrShort;
#endif
			getStringName(csSampleEx[refTblId].candidatesOrdered[0], &nameStr, mapi, mbat, 1);
#if USE_SHORT_NAMES
			getPropNameShort(&nameStrShort, nameStr);
			fprintf(fout, "->%s", nameStrShort);
			GDKfree(nameStrShort);
#else
			fprintf(fout, "->%s", nameStr);
#endif
			GDKfree(nameStr);
		} else { // no table name
			fprintf(fout, "->Table%d", refTblId);
		}
	}
}
#endif

#if NO_OUTPUTFILE == 0
// Compute property order and number of properties that are printed, and the list of remaining properties that is printed without sample data
static
int* createPropertyOrder(int *numPropsInSampleTable, int **remainingProperties, CSset *freqCSset, CSSampleExtend *csSampleEx, int tblId, CSPropTypes *csPropTypes, PropStat *propStat, char* isTypeProp) {
	int		i;
	CSSampleExtend	sample;
	CSPropTypes	csPropType;
	int		support; // support of this table
	int		*propOrder; // return value, order of properties
	int		propsAdded = 0; // how many properties are already added to propOrder
	int		propsAddedRemaining = 0; // how many properties are already added to remainingProperties
	int		*propOrderTfidf;
	float		*tfidfValues;
	char		*isFilled; // property is >=50% filled (non-NULL)
	char		*isTextDate; // value has type string or datetime
	char		*isAdded; // property is added to propOrder

	sample = csSampleEx[tblId];
	csPropType = csPropTypes[tblId];

	// number of properties that will be shown to the user
	(*numPropsInSampleTable) = (sample.numProp>NUM_PROP_SAMPLE)?(NUM_PROP_SAMPLE):sample.numProp;

	// init arrays
	propOrder = GDKmalloc(sizeof(int) * (*numPropsInSampleTable));
	propOrderTfidf = GDKmalloc(sizeof(int) * sample.numProp);
	tfidfValues = GDKmalloc(sizeof(float) * sample.numProp);
	isFilled = GDKmalloc(sizeof(char) * sample.numProp);
	isTextDate = GDKmalloc(sizeof(char) * sample.numProp);
	isAdded = GDKmalloc(sizeof(char) * sample.numProp);
	if (!propOrder || !propOrderTfidf || !tfidfValues || !isFilled || !isTextDate || !isAdded) {
		fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	}
	for (i = 0; i < sample.numProp; ++i) {
		propOrderTfidf[i] = i; // initial order
		isFilled[i] = 0;
		isTextDate[i] = 0;
		isAdded[i] = 0;
	}
	if (sample.numProp > (*numPropsInSampleTable)) {
		(*remainingProperties) = GDKmalloc(sizeof(int) * (sample.numProp - (*numPropsInSampleTable)));
		if (!(*remainingProperties)) {
			fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		}
	}

	// create isFilled
	support = freqCSset->items[sample.freqIdx].support;
	for (i = 0; i < sample.numProp; ++i) {
		int propSupport = sample.lstPropSupport[i];
		if (propSupport*2 >= support) { // threshold 50%
			isFilled[i] = 1;
		}
	}

	// create isTextDate
	for (i = 0; i < sample.numProp; ++i) {
		ObjectType defaultType = csPropType.lstPropTypes[i].defaultType;
		if (defaultType == DATETIME || defaultType == STRING) {
			isTextDate[i] = 1;
		}
	}

	// create tfidfValues
	for (i = 0; i < sample.numProp; ++i) {
		float tfidf = 0.0;
		BUN bun = BUNfnd(propStat->pBat,(ptr) &sample.lstProp[i]);
		if (bun == BUN_NONE) {
			printf("Error: property not found\n");
		} else {
			tfidf = propStat->tfidfs[bun];
		}
		tfidfValues[i] = tfidf;
	}

	// create propOrderIdf: sort descending by tfidfValues (result: discriminating properties are sortedto the begin of the propOrderTfidf array) using insertion sort
	for (i = 1; i < sample.numProp; ++i) {
		int tmpPos = propOrderTfidf[i];
		float tmpVal = tfidfValues[tmpPos];
		int j = i - 1;
		while (j >= 1 && tfidfValues[propOrderTfidf[j]] < tmpVal) { // sort descending
			propOrderTfidf[j + 1] = propOrderTfidf[j];
			j--;
		}
		propOrderTfidf[j + 1] = tmpPos;
	}

	// now add properties to propOrder array
	// add all type properties
	for (i = 0; i < sample.numProp; ++i) {
		if (propsAdded >= (*numPropsInSampleTable)) break; // enough properties found
		if (isTypeProp[i]) { // do not use 'index' because the isTypeProp array uses the old order of properties
			propOrder[propsAdded] = i;
			isAdded[i] = 1;
			propsAdded++;
		}
	}

	// first round: properties with isFilled=1 and isTextDate=1, ordered by tfidfValues descending
	for (i = 0; i < sample.numProp; ++i) {
		int index = propOrderTfidf[i];
		if (propsAdded >= (*numPropsInSampleTable)) break; // enough properties found
		if (isFilled[index] && isTextDate[index] && !isAdded[index]) {
			// add
			propOrder[propsAdded] = index;
			isAdded[index] = 1;
			propsAdded++;
		}
	}

	// second round: properties with isFilled=1 and isTextDate=0, ordered by tfidfValues descending
	for (i = 0; i < sample.numProp; ++i) {
		int index = propOrderTfidf[i];
		if (propsAdded >= (*numPropsInSampleTable)) break; // enough properties found
		if (isFilled[index] && !isTextDate[index] && !isAdded[index]) {
			// add
			propOrder[propsAdded] = index;
			isAdded[index] = 1;
			propsAdded++;
		}
	}

	// third round: properties ordered b tfidfValues descending
	for (i = 0; i < sample.numProp; ++i) {
		int index = propOrderTfidf[i];
		if (propsAdded >= (*numPropsInSampleTable)) break; // enough properties found
		if (!isAdded[index]) {
			// add
			propOrder[propsAdded] = index;
			isAdded[index] = 1;
			propsAdded++;
		}
	}

	// propOrder is finished now

	// create remainingProperties: add all properties that have not been added to propOrder
	for (i = 0; i < sample.numProp; ++i) {
		int index = propOrderTfidf[i];
		if (!isAdded[index]) {
			// add
			(*remainingProperties)[propsAddedRemaining] = index;
			propsAddedRemaining++;
		}
	}

	GDKfree(propOrderTfidf);
	GDKfree(tfidfValues);
	GDKfree(isFilled);
	GDKfree(isTextDate);
	GDKfree(isAdded);

	return propOrder;
}
#endif

#if NO_OUTPUTFILE == 0
static 
str printFullSampleData(CSSampleExtend *csSampleEx, int num, BAT *mbat, PropStat *propStat, CSset *freqCSset, CSPropTypes *csPropTypes){

	int 	i,j, k; 
	FILE 	*fout, *foutrand, *foutsol, *fouttb, *foutis; 
	char 	filename[100], filename4[100], filename2[100], filename3[100];
	int 	ret;

	str 	propStr; 
	str	subjStr; 
	char*   schema = "rdf";
	CSSampleExtend	sample; 
	str	objStr; 	
	oid	*objOid = NULL; 
	double	*objDbl = NULL; 
	int	*objInt = NULL; 
	timestamp *objTs = NULL;
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
	BAT	*tmpBat = NULL; 
	BATiter	tmpi; 
	BATiter mapi;
#if USE_SHORT_NAMES
	str	propStrShort = NULL;
	char 	*pch; 
#endif
	CS	freqCS;

	// property order
	oid	*typeAttributesOids;
	char	*isTypeProp; // 1 if property is in typeAttributes[]
	int	numPropsInSampleTable;
	int	*remainingProperties;
	int	*propOrder;

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	
	// get oids for typeAttributes[]
	typeAttributesOids = GDKmalloc(sizeof(oid) * typeAttributesCount);
	if (!typeAttributesOids){
		fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	}
	for (i = 0; i < typeAttributesCount; ++i) {
		TKNZRappend(&typeAttributesOids[i], &typeAttributes[i]);
	}

	mapi = bat_iterator(mbat);

	strcpy(filename, "sampleDataFull");
	strcat(filename, ".txt");

	strcpy(filename4, "sampleDataFullSolution");
	strcat(filename4, ".txt");
	
	strcpy(filename2, "createSampleTableFull");
	strcat(filename2, ".sh");
	
	strcpy(filename3, "loadSampleToMonetFull");
	strcat(filename3, ".sh");
	
	fout = fopen(filename,"wt"); 
	foutsol = fopen(filename4,"wt");
	foutrand = fopen("sampleDataFullRandom.txt","wt");
	fouttb = fopen(filename2,"wt");
	foutis = fopen(filename3,"wt");

	fprintf(foutrand, "Table|Name|Rating\n");
	for (i = 0; i < num; i++){
		sample = csSampleEx[i];
		if ((int)sample.candidateCount == 1 && sample.candidates[0] == BUN_NONE) continue; // do not print tables withoud candidates
		freqCS = freqCSset->items[sample.freqIdx];
		fprintf(fout,"Table %d, %d tuples\n", i, freqCS.support);
		fprintf(foutrand,"Table %d, %d tuples", i, freqCS.support);
		fprintf(foutsol, "Table %d\n", i);
		for (j = 0; j < (int)sample.candidateCount; j++){
			//fprintf(fout,"  "  BUNFMT,sample.candidates[j]);
			if (sample.candidates[j] != BUN_NONE){
#if USE_SHORT_NAMES
				str canStrShort = NULL;
#endif
				//takeOid(sample.candidates[j], &canStr); 
				getStringName(sample.candidates[j], &canStr, mapi, mbat, 1);
#if USE_SHORT_NAMES
				getPropNameShort(&canStrShort, canStr);
				fprintf(foutrand, "|%s\n",  canStrShort);
				GDKfree(canStrShort);
#else
				fprintf(foutrand, "%s",  canStr);

#endif
				GDKfree(canStr); 
			
			}
			// ordered candidates for solution
			if (sample.candidatesOrdered[j] != BUN_NONE){
#if USE_SHORT_NAMES
				str canStrShort = NULL;
#endif
				getStringName(sample.candidatesOrdered[j], &canStr, mapi, mbat, 1);
#if USE_SHORT_NAMES
				getPropNameShort(&canStrShort, canStr);
				if (j+1 == (int)sample.candidateCount) fprintf(foutsol, "%s (%s)",  canStrShort, canStr);
				else fprintf(foutsol, "%s (%s)|", canStrShort, canStr);
				GDKfree(canStrShort);
#else
				if (j+1 == (int)sample.candidateCount) fprintf(foutsol, "%s",  canStr);
				else fprintf(foutsol, "%s|", canStr);

#endif
				GDKfree(canStr); 
			
			}
		}
		fprintf(foutsol, "\n");

		// print origin of candidates for solutions file
		fprintf(foutsol, "New %d, Type %d, Ontology %d, FK %d\n", sample.candidatesNew, sample.candidatesType, sample.candidatesOntology, sample.candidatesFK);
		
		if (sample.name != BUN_NONE){
			str canStrShort = NULL;
			//takeOid(sample.name, &canStr);
			getStringName(sample.name, &canStr, mapi, mbat, 1);
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

		// mark type columns, because their sample data is represented without <...>
		isTypeProp = GDKmalloc(sizeof(char) * sample.numProp);
		if (!isTypeProp){
			fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		}
		for (j = 0; j < sample.numProp; ++j) {
			isTypeProp[j] = 0;
		}
		for (j = 0; j < sample.numProp; ++j) {
			for (k = 0; k < typeAttributesCount; ++k) {
				if (sample.lstProp[j] == typeAttributesOids[k]) {
					// found a type property
					isTypeProp[j] = 1;
					break;
				}
			}
		}

		// order properties and get list of "remaining" properties that will be printed without sample data
		remainingProperties = NULL;
		numPropsInSampleTable = 0;
		propOrder = createPropertyOrder(&numPropsInSampleTable, &remainingProperties, freqCSset, csSampleEx, i, csPropTypes, propStat, isTypeProp);

		// print list of columns that did not make it to propOrder and are therefore printed without sample data
		if (sample.numProp > numPropsInSampleTable) {
			fprintf(fout, "Additional columns: ");
			for (j = 0; j < (sample.numProp - numPropsInSampleTable); ++j) {
				int index = remainingProperties[j]; // apply mapping to change order of properties
				propStr = NULL;
#if USE_SHORT_NAMES
				propStrShort = NULL;
#endif
				takeOid(sample.lstProp[index], &propStr);
#if USE_SHORT_NAMES
				getPropNameShort(&propStrShort, propStr);
				if (j != 0) fprintf(fout, ", "); // separator
				printPropertyWithMarkers(fout, propStrShort, csSampleEx, csPropTypes, i, index, sample.lstProp[index], mapi, mbat);
				GDKfree(propStrShort);
#else
				if (j != 0) fprintf(fout, ", "); // separator
				printPropertyWithMarkers(fout, propStr, csSampleEx, csPropTypes, i, index, sample.lstProp[index], mapi, mbat);
#endif
				GDKfree(propStr);
			}
			fprintf(fout, "\n");
		} else {
			// we have to print an empty row to ensure that all tables have the same height, this simplifies the survey layouting in a spreadsheet programm
			fprintf(fout, "\n");
		}

		//List of columns
		fprintf(fout,"Subject");
		fprintf(fouttb,"SubjectCol string");
		isTitle = 0; 
		isUrl = 0;
		isType = 0; 
		isDescription = 0; 
		isImage = 0;
		isSite = 0; 
		for (j = 0; j < numPropsInSampleTable; j++){
			int index = propOrder[j]; // apply mapping to change order of properties
#if USE_SHORT_NAMES
			propStrShort = NULL;
#endif
			takeOid(sample.lstProp[index], &propStr);	
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			fprintf(fout,"|");
			printPropertyWithMarkers(fout, propStrShort, csSampleEx, csPropTypes, i, index, sample.lstProp[index], mapi, mbat);

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
				fprintf(fouttb,",\n%s_%d string",propStrShort,index);
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
			fprintf(fout, "|");
			printPropertyWithMarkers(fout, propStr, csSampleEx, csPropTypes, i, index, sample.lstProp[index], mapi, mbat);
#endif
			GDKfree(propStr);
		}
		fprintf(fout, "\n");
		fprintf(fouttb, "\n); \n \n");
		
		fprintf(foutis, "echo \"");
		//All the instances 
		for (k = 0; k < sample.numInstances; k++){
			takeOid(sample.lstSubjOid[k], &subjStr); 
			fprintf(foutis,"<%s>", subjStr);
			fprintf(fout,"%s", subjStr);
			GDKfree(subjStr); 
			
			for (j = 0; j < numPropsInSampleTable; j++){
				int index = propOrder[j]; // apply mapping to change order of properties
				tmpBat = sample.colBats[index];
				tmpi = bat_iterator(tmpBat);
				
				if (tmpBat->ttype == TYPE_oid){	//URI or BLANK NODE  or MVCol
					objOid = (oid *) BUNtail(tmpi, k);
					if (*objOid == oid_nil){
						fprintf(fout,"|NULL");
						fprintf(foutis,"|NULL");
					}
					else{
						str objStrShort = NULL;
						takeOid(*objOid, &objStr);
						getPropNameShort(&objStrShort, objStr);

						if (isTypeProp[index]) {
							// type props are printed without <...>
							fprintf(fout,"|%s", objStrShort);
						} else {
							fprintf(fout,"|<%s>", objStrShort);
						}
						fprintf(foutis,"|<%s>", objStrShort);
						GDKfree(objStrShort);
						GDKfree(objStr);
					}
				}
				else if (tmpBat->ttype == TYPE_dbl){
					objDbl = (double *) BUNtail(tmpi, k); 
					if (*objDbl == dbl_nil){
						fprintf(fout,"|NULL");
						fprintf(foutis,"|NULL");
					} 
					else{
						fprintf(fout,"|%f", *objDbl);
						fprintf(foutis,"|%f", *objDbl);

					}
				}
				else if (tmpBat->ttype == TYPE_int){
					objInt = (int *) BUNtail(tmpi, k);
					if (*objInt == int_nil){
						fprintf(fout,"|NULL");
						fprintf(foutis,"|NULL");
					}
					else{
						fprintf(fout,"|%d", *objInt);
						fprintf(foutis,"|%d", *objInt);
					}
				
				}
				else if (tmpBat->ttype == TYPE_timestamp){	//Datetime
					objTs = (timestamp *) BUNtail(tmpi, k);
					if (ts_isnil(*objTs)){
						fprintf(fout,"|NULL");
						fprintf(foutis,"|NULL");
					}
					else{	//get datetime string from timestamp
						char buff[64], *s1 = buff; 
						int len = 64; 
						*s1 = 0; 

						timestamp_tostr(&s1,&len,objTs);
						fprintf(fout,"|%s", s1);
						fprintf(foutis,"| %s", s1);
					}

				}
				else{ //tmpBat->ttype == TYPE_str, MV column also has string type (concate list of values)
					objStr = NULL; 
					objStr = BUNtail(tmpi, k);
					if (strcmp(objStr, str_nil) == 0){
						fprintf(fout,"|NULL");
						fprintf(foutis,"|NULL");
					}
					else{
						str objStrShort;
						getStringBetweenQuotes(&objStrShort, objStr); // remove quotes and language tags
						fprintf(fout,"|%s", objStrShort);
						fprintf(foutis,"| %s", objStr);
						GDKfree(objStrShort);
					}
				}


			}
			fprintf(fout, "\n");
			fprintf(foutis, "\n");
		}

		if (sample.numProp > numPropsInSampleTable) {
			GDKfree(remainingProperties);
		}
		GDKfree(propOrder);

		fprintf(fout, "\n");
		fprintf(foutsol, "\n");
		fprintf(foutis, "\" > tmp.txt \n \n");

		if (sample.name != BUN_NONE){
			str canStrShort = NULL;
			//takeOid(sample.name, &canStr);
			getStringName(sample.name, &canStr, mapi, mbat, 1); 
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

			
		GDKfree(isTypeProp);
	}

	GDKfree(typeAttributesOids);

	fclose(fout);
	fclose(foutsol);
	fclose(foutrand);
	fclose(fouttb); 
	fclose(foutis); 
	
	TKNZRclose(&ret);


	return MAL_SUCCEED;
}
#endif


#if NO_OUTPUTFILE == 0 
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
	curP = BUN_NONE; 

	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);		
		if (*sbt != curS){
			if (p != 0){	/* Not the first S */
				tblIdx = csTblIdxMapping[subjCSMap[curS]];
				if (tblIdx != -1){
				
					sampleIdx = BUNfnd(tblCandBat,(ptr) &tblIdx);
					if (sampleIdx != BUN_NONE) {
						assert(!(numP > csSample[sampleIdx].numProp));
						if (csSample[sampleIdx].numInstances < NUM_SAMPLE_INSTANCE){	
							addSampleInstance(curS, buffO, buffP, numP, sampleIdx, csSample);
							totalInstance++;
						}
					}
				}
			}
			curS = *sbt; 
			numP = 0;
			curP = BUN_NONE; 
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
#endif

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
CSrel* getFKBetweenTableSet(CSrel *csrelFreqSet, CSset *freqCSset, CSPropTypes* csPropTypes, int* mfreqIdxTblIdxMapping, int numTables, CSlabel *labels){
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
		if (!isCSTable(freqCSset->items[i], labels[i].name)) continue; 
		rel = csrelFreqSet[i];
		from = mfreqIdxTblIdxMapping[i];
		assert(from < numTables);
		assert(from != -1); 
		// update the 'from' value
		for (j = 0; j < rel.numRef; ++j) {
			toFreqId = rel.lstRefFreqIdx[j];
			assert(freqCSset->items[toFreqId].parentFreqIdx == -1);
			if (!isCSTable(freqCSset->items[toFreqId], labels[toFreqId].name)) continue; 
			// add relation to new data structure

			//Compare with prop coverage from csproptype	
			if (rel.lstCnt[j]  < freqCSset->items[toFreqId].support * infreqTypeThreshold)	continue; 

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
			if (freqCSset->items[i].coverage > minTableSize){
				if (csPropTypes[from].lstPropTypes[propIdx].propCover * (1 - infreqTypeThreshold) > rel.lstCnt[j]) continue; 
				else if (csPropTypes[from].lstPropTypes[propIdx].propCover == rel.lstCnt[j])
					csPropTypes[from].lstPropTypes[propIdx].isDirtyFKProp = 0;
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

#if NO_OUTPUTFILE == 0 
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
#endif

#if NO_OUTPUTFILE == 0
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
#endif

#if NO_OUTPUTFILE == 0
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
#endif


#if NO_OUTPUTFILE == 0 
static 
str getSampleData(int *ret, bat *mapbatid, int numTables, CSset* freqCSset, BAT *sbat, BATiter si, BATiter pi, BATiter oi, int* mTblIdxFreqIdxMapping, 
		CSlabel* labels, int* csTblIdxMapping, int maxNumPwithDup, oid* subjCSMap, int sampleVersion){

	BAT		*outputBat = NULL, *mbat = NULL;
	CSSample 	*csSample; 
	int		numSampleTbl = 0;  

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
	printsubsetFromCSset(freqCSset, outputBat, mbat, numSampleTbl, mTblIdxFreqIdxMapping, labels, sampleVersion);
	printSampleData(csSample, freqCSset, mbat, numSampleTbl, sampleVersion);
	freeSampleData(csSample, numSampleTbl);
	BBPreclaim(outputBat);
	BBPunfix(mbat->batCacheid);
	
	return MAL_SUCCEED; 
}
#endif

#if NO_OUTPUTFILE == 0
static
str getFullSampleData(CStableStat* cstablestat, CSPropTypes *csPropTypes, int *mTblIdxFreqIdxMapping, CSlabel *labels, int numTables,  bat *lmapbatid, bat *rmapbatid, CSset *freqCSset, bat *mapbatid, PropStat *propStat){

	CSSampleExtend *csSampleEx;
	BAT *mbat = NULL; 

	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	csSampleEx = (CSSampleExtend *)malloc(sizeof(CSSampleExtend) * numTables);
	
	initFullSampleData(csSampleEx, mTblIdxFreqIdxMapping, labels, cstablestat, csPropTypes, freqCSset, numTables, lmapbatid, rmapbatid);

	printFullSampleData(csSampleEx, numTables, mbat, propStat, freqCSset, csPropTypes);
	
	freeSampleExData(csSampleEx, numTables);
	BBPunfix(mbat->batCacheid);

	return MAL_SUCCEED; 
}
#endif


#if NO_OUTPUTFILE == 0 
static
str printFinalStructure(CStableStat* cstablestat, CSPropTypes *csPropTypes, int numTables, int freqThreshold, bat *mapbatid){

	int 		i,j; 
	int		tmpNumDefaultCol; 
	FILE		*fout;
	char    	filename[100];
	char    	tmpStr[20];
	int		ret; 
	str		subjStr; 
	str		propStr; 
	int 		numNoNameTable = 0;
	char*		schema = "rdf";
	BATiter		mapi;
	BAT		*mbat = NULL;  
	#if COUNT_PERCENTAGE_ONTO_PROP_USED
	int		numOntologyName = 0; 
	int		numOntologyProp = 0;
	int		numOntologyPropUsed = 0;
	int		tmpNumOverlap = 0;
	BUN		tmpPos = BUN_NONE; 
	#endif

	printf("Summarizing the final table information \n"); 
	// allocate memory space for cstablestat
	strcpy(filename, "finalSchema");
	sprintf(tmpStr, "%d", freqThreshold);
	strcat(filename, tmpStr);
	strcat(filename, ".txt");

	fout = fopen(filename,"wt"); 
	
	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	mapi = bat_iterator(mbat); 
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	for (i = 0; i < numTables; i++){

		tmpNumDefaultCol = csPropTypes[i].numProp -  csPropTypes[i].numInfreqProp;
		assert(tmpNumDefaultCol ==  cstablestat->lstcstable[i].numCol);
		
		if (cstablestat->lstcstable[i].tblname != BUN_NONE){
			str subjStrShort = NULL;
			//takeOid(cstablestat->lstcstable[i].tblname, &subjStr);
			getStringName(cstablestat->lstcstable[i].tblname, &subjStr, mapi, mbat, 1); 
			getPropNameShort(&subjStrShort, subjStr);
			#if COUNT_PERCENTAGE_ONTO_PROP_USED
			tmpNumOverlap = 0;
			tmpPos = BUNfnd(ontmetaBat, &cstablestat->lstcstable[i].tblname);
			if (tmpPos != BUN_NONE){
				if (ontclassSet[tmpPos].numProp != 0){	//otherwise, we do not have the information for this ontology class
					countNumOverlapProp(ontclassSet[tmpPos].lstProp, cstablestat->lstcstable[i].lstProp , 
								ontclassSet[tmpPos].numProp,tmpNumDefaultCol, &tmpNumOverlap);		

					numOntologyPropUsed += tmpNumOverlap;
					numOntologyName += 1;
					numOntologyProp += ontclassSet[tmpPos].numProp;
				}
				
			}
			fprintf(fout, "Table %d (Name: %s | NumCols: %d | (Num onto prop. used: %d)\n", i, subjStrShort, tmpNumDefaultCol,tmpNumOverlap);			
			#else
			fprintf(fout, "Table %d (Name: %s | NumCols: %d)\n", i, subjStrShort, tmpNumDefaultCol);
			#endif

			GDKfree(subjStrShort);
			GDKfree(subjStr);
		}
		else{
			fprintf(fout, "Table %d (Name: <NoName> | NumCols: %d)\n", i, tmpNumDefaultCol);
			numNoNameTable++;
		}


		for(j = 0; j < csPropTypes[i].numProp; j++){
			str propStrShort = NULL;
			#if     REMOVE_INFREQ_PROP
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1)     continue;  //Infrequent prop
			#endif
			takeOid(csPropTypes[i].lstPropTypes[j].prop, &propStr);
			getPropNameShort(&propStrShort, propStr);

			fprintf(fout, "     Prop ("BUNFMT"): %s",csPropTypes[i].lstPropTypes[j].prop, propStr);

			if (csPropTypes[i].lstPropTypes[j].isFKProp == 1){
				fprintf(fout, " == FK ==> %d \n", csPropTypes[i].lstPropTypes[j].refTblId);
			}
			else
				fprintf(fout, "\n");
			GDKfree(propStrShort);
			GDKfree(propStr);
		}

		fprintf(fout, "\n");

	}
	
	printf(" Number of no-name table: %d | (Total: %d)\n",numNoNameTable,numTables);
	#if COUNT_PERCENTAGE_ONTO_PROP_USED
	if (numOntologyProp != 0){
		printf(" Percentage of ontology prop. used is %f (in %d ontology names used) \n", (float) numOntologyPropUsed/numOntologyProp,numOntologyName);
	}
	else{
		printf(" Percentage of ontology prop. used: There is no ontology attributes for this dataset \n");
	}
	#endif

	fclose(fout); 
	
	BBPunfix(mbat->batCacheid);
	TKNZRclose(&ret);

	return MAL_SUCCEED; 
}
#endif


static
str getRefTables(CSPropTypes *csPropTypes, int numTables, char* isRefTables){
	int i, j; 
	for (i = 0; i < numTables; i++){

		for(j = 0; j < csPropTypes[i].numProp; j++){
			#if     REMOVE_INFREQ_PROP
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1)     continue;  //Infrequent prop
			#endif
			if (csPropTypes[i].lstPropTypes[j].isFKProp == 1 && csPropTypes[i].lstPropTypes[j].isDirtyFKProp == 0){
				isRefTables[csPropTypes[i].lstPropTypes[j].refTblId] = 1;
			}

		}
	}
	
	return MAL_SUCCEED; 
}


static
void initCSTableIdxMapping(CSset* freqCSset, int* csTblIdxMapping, int* csFreqCSMapping, int* mfreqIdxTblIdxMapping, int* mTblIdxFreqIdxMapping, int *numTables, CSlabel *labels){

int 		i, k; 
CS 		cs;
	int		tmpParentidx; 

	k = 0; 
	for (i = 0; i < freqCSset->numCSadded; i++){
		if (isCSTable(freqCSset->items[i], labels[i].name)){	// Only use the not-removed maximum or merge CS  
			mfreqIdxTblIdxMapping[i] = k; 
			mTblIdxFreqIdxMapping[k] = i; 
			k++; 
		}
	}
	
	*numTables = k; 

	// Mapping the csid directly to the index of the table ==> csTblIndxMapping
	
	for (i = 0; i < freqCSset->numOrigFreqCS; i++){
		cs = (CS)freqCSset->items[i];
		csFreqCSMapping[cs.csId] = i; 
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
#if STORE_PERFORMANCE_METRIC_INFO
static 
void setInitialMetricsInfo(int* refCount, CSset *freqCSset){
	int i;
	int total = 0; 
	oid csId;
	CS* cs;
	for (i = 0; i < freqCSset->numCSadded; i++){
		cs = &(freqCSset->items[i]);
		csId = cs->csId;
		cs->numInRef = refCount[csId];
		cs->numFill = cs->numProp * cs->support; 
		total += cs->numFill; 
	}

	freqCSset->totalInRef = total; 
}

static
Pscore computeMetricsQ(CSset *freqCSset){
	float* fillRatio;
	float* refRatio;
	float* weight;
	int tblIdx = -1;
	CS cs;	

	int	totalCov = 0; 
	float	totalPrecision = 0.0; 
	lng	overalFill = 0; 
	lng 	overalMaxFill = 0;

	int	totalExpFinalCov = 0; 
	float	totalExpFinalPrecision = 0.0; 
	lng	overalExpFinalFill = 0; 
	lng 	overalExpFinalMaxFill = 0;
	float	expFinalQ = 0.0; 

	float	Q = 0.0;
	int	i;
	int	numExpFinalTbl = 0;	//Expected number of table after removing small table
	Pscore  pscore; 

	int curNumMergeCS = countNumberMergeCS(freqCSset);

	fillRatio = (float*)malloc(sizeof(float) * curNumMergeCS);
	refRatio = (float*)malloc(sizeof(float) * curNumMergeCS);
	weight = (float*)malloc(sizeof(float) * curNumMergeCS);

	for (i = 0; i < freqCSset->numCSadded; i ++){
		if (freqCSset->items[i].parentFreqIdx == -1){
			//Table i;
			cs = freqCSset->items[i];
			tblIdx++;	
			fillRatio[tblIdx] = (float) cs.numFill /((float)cs.numProp *  cs.support);
			refRatio[tblIdx] = (float) cs.numInRef / freqCSset->totalInRef;
			weight[tblIdx] = (float) cs.coverage * (fillRatio[tblIdx] + refRatio[tblIdx]); 
			//weight[tblIdx] = (float) cs.coverage * ( fillRatio[tblIdx]);  //If do not consider reference ratio
			totalCov += cs.coverage;
			totalPrecision += fillRatio[tblIdx];
			overalFill += cs.numFill;
			overalMaxFill += cs.numProp *  cs.support;

			//if ((cs.numProp *  cs.support) > 1000000) printf("FreqCS %d has %d prop and support %d (Fill Ratio %f )\n",i,cs.numProp,cs.support,fillRatio[tblIdx]);
			
			Q += weight[tblIdx];

			if (isCSTable(freqCSset->items[i], 1)){ 
				totalExpFinalCov += cs.coverage;
				totalExpFinalPrecision += fillRatio[tblIdx];
				overalExpFinalFill += cs.numFill;
				overalExpFinalMaxFill += cs.numProp *  cs.support;
				expFinalQ += weight[tblIdx];
				numExpFinalTbl++;
			}
		}
	}
	printf("Performance metric Q = (weighting %f)/(totalCov %d * numTbl %d) \n", Q,totalCov, curNumMergeCS);
	//printf("Average precision = %f\n",(float)totalPrecision/curNumMergeCS);
	//printf("Overall precision = %f (overfill %lld / overalMaxFill %lld)\n", (float) overalFill/overalMaxFill, overalFill, overalMaxFill);
	//printf("Average precision = %f\n",(float)totalPrecision/totalCov);

	Q = Q/((float)totalCov * curNumMergeCS);

	printf("==> Performance metric Q = %f \n", Q);

	expFinalQ = expFinalQ/((float)totalExpFinalCov * numExpFinalTbl);

	pscore.avgPrec = (float)totalPrecision/curNumMergeCS; 
	pscore.overallPrec = (float) overalFill/overalMaxFill;
	pscore.Qscore = Q;
	//pscore.Cscore = 
	pscore.nTable = curNumMergeCS;
	pscore.nFinalTable = numExpFinalTbl;
	pscore.avgPrecFinal = (float)totalExpFinalPrecision/numExpFinalTbl;
	pscore.overallPrecFinal = (float) overalExpFinalFill/overalExpFinalMaxFill;
	pscore.QscoreFinal = expFinalQ;

	free(fillRatio); 
	free(refRatio); 
	free(weight); 
	
	return pscore;
}


//Compute the metric for table after removing infrequent props
static
void computeMetricsQForRefinedTable(CSset *freqCSset,CSPropTypes *csPropTypes,int *mfreqIdxTblIdxMapping, int *mTblIdxFreqIdxMapping, int numTables){
	float* fillRatio;
	float* refRatio;
	float* weight;
	CS cs;	
	int	totalCov = 0; 
	float	Q = 0.0;
	int	i,j;
	int 	tmpFinalFreqIdx, tmpTblIdx, tmpPropIdx;
	int	tmpNumFreqProps;
	int	*numRefinedFills = NULL;
	int 	*numRefinedSupport = NULL;
	#if NO_OUTPUTFILE == 0	
	PropStat *propStat = NULL; 	
	int	numdistinctMCS = 0;
	int 	numSubjWithoutDiscProp = 0;
	int	numTriplesWihtoutDiscProp = 0;
	char 	isContainedDiscProp = 0;
	oid	p;
	oid	*pbt;	
	BUN	bun;
	FILE	*fout; 
	char	filename[100];
	#endif

	float	totalPrecision = 0.0; 
	lng	overalFill = 0; 
	lng 	overalMaxFill = 0;

	fillRatio = (float*)malloc(sizeof(float) * numTables);
	refRatio = (float*)malloc(sizeof(float) * numTables);
	weight = (float*)malloc(sizeof(float) * numTables);
		
	numRefinedFills = (int*)malloc(sizeof(int) * numTables);
	numRefinedSupport = (int*)malloc(sizeof(int) * numTables);
	//At the beginning
	for (i = 0; i < numTables; i ++){
		numRefinedFills[i] = freqCSset->items[mTblIdxFreqIdxMapping[i]].numFill;
		numRefinedSupport[i] =  freqCSset->items[mTblIdxFreqIdxMapping[i]].support; 
	}
	
	
	#if NO_OUTPUTFILE == 0
	
	propStat = getPropStatisticsByTable(numTables, mTblIdxFreqIdxMapping, freqCSset,  &numdistinctMCS);
	//Print the TF-IDF score of each prop in each table
	
	strcpy(filename,"propStatWithFinalSchema.txt");
	fout = fopen(filename,"wt"); 
	fprintf(fout, "PropertyOid #ofCSs tfidfscore");	
	for (i = 0; i < propStat->numAdded; i++){
		pbt = (oid *) Tloc(propStat->pBat, i);
		fprintf(fout, BUNFMT "	%d	%f \n", *pbt, propStat->plCSidx[i].numAdded,propStat->tfidfs[i]);
	}
	fclose(fout);
	#endif

	


	//Removing LOTSOFNULL_SUBJECT_THRESHOLD	
	//Check which freqCS having small number of prop
	//--> they will be removed from the final table.
	for (i = 0; i < freqCSset->numOrigFreqCS; i++){
		tmpFinalFreqIdx = i;
		while (freqCSset->items[tmpFinalFreqIdx].parentFreqIdx != -1){
			tmpFinalFreqIdx = freqCSset->items[tmpFinalFreqIdx].parentFreqIdx;
		}
		
		if (mfreqIdxTblIdxMapping[tmpFinalFreqIdx] == -1) continue; //This mergedCS does not become the final table, because of e.g.,small size

		tmpTblIdx = mfreqIdxTblIdxMapping[tmpFinalFreqIdx];
		tmpNumFreqProps = csPropTypes[tmpTblIdx].numProp - csPropTypes[tmpTblIdx].numInfreqProp;

		if (freqCSset->items[i].numProp < tmpNumFreqProps * LOTSOFNULL_SUBJECT_THRESHOLD){
			int tmpNumFreqProp = freqCSset->items[i].numProp;	//Init
			//This CS will be removed

			//Check number of InfreqProp exist in that CS
			//Since they will finally be removed by removing Infrequent Prop from final tabl
			//the reducing of numofFill caused by these props will not be counted 
			//when removing this freqCS i.
			tmpPropIdx = 0;
			for (j = 0; j < freqCSset->items[i].numProp; j++){
				oid checkProp = freqCSset->items[i].lstProp[j];
				//Check if prop j is a infrquent prop
				while (tmpPropIdx < csPropTypes[tmpTblIdx].numProp && csPropTypes[tmpTblIdx].lstPropTypes[tmpPropIdx].prop != checkProp){
					tmpPropIdx++;
				}

				if (tmpPropIdx == csPropTypes[tmpTblIdx].numProp) break; //No more check
				
				//if found the index of the prop, check if it is infrequent
				if ( isInfrequentProp(csPropTypes[tmpTblIdx].lstPropTypes[tmpPropIdx], freqCSset->items[tmpFinalFreqIdx])){
					tmpNumFreqProp--;
				}
				
			}
			
			numRefinedSupport[tmpTblIdx] = numRefinedSupport[tmpTblIdx] - freqCSset->items[i].support; 
			numRefinedFills[tmpTblIdx] = numRefinedFills[tmpTblIdx] - (freqCSset->items[i].support * tmpNumFreqProp);		
				

		}
		
		#if NO_OUTPUTFILE == 0
		//Get the number of subject having no discriminating props in final Table
		cs = freqCSset->items[i];
		
		isContainedDiscProp = 0;
		for (j = 0; j < cs.numProp; j++){
			p = cs.lstProp[j]; 
			bun = BUNfnd(propStat->pBat,(ptr) &p);
			if (bun == BUN_NONE) {
				printf("FreqCS: %d, prop "BUNFMT" --> This prop must be in propStat!!!!\n",i,p);
			}
			else{
				 if (propStat->tfidfs[bun] > MIN_TFIDF_PROP_FINALTABLE)	{
					isContainedDiscProp = 1;
					break;
				}
			}
		}
		if (isContainedDiscProp == 0){	//There is no discriminating prop in this CS	
			numSubjWithoutDiscProp += cs.support;
			numTriplesWihtoutDiscProp += cs.coverage;
		}
	

		#endif
	}
	
	#if NO_OUTPUTFILE == 0
	printf("Number of Subject having no discriminating props is: %d\n",numSubjWithoutDiscProp);
	printf(" ==> Removing these subject will remove %d triples \n",numTriplesWihtoutDiscProp);
	#endif	
	
	for (i = 0; i < numTables; i++){
		tmpFinalFreqIdx = mTblIdxFreqIdxMapping[i];
		cs = freqCSset->items[tmpFinalFreqIdx];
		
		//Reduce the number of fill when removing infrequent props
		for (j = 0; j < csPropTypes[i].numProp; j++){
			if ( isInfrequentProp(csPropTypes[i].lstPropTypes[j], cs)){
				numRefinedFills[i] = numRefinedFills[i] - csPropTypes[i].lstPropTypes[j].propFreq;
			}
		}
		tmpNumFreqProps = csPropTypes[i].numProp - csPropTypes[i].numInfreqProp;
		assert(tmpNumFreqProps > 0);
		assert( numRefinedSupport[i] > 0); 
		fillRatio[i] = (float) numRefinedFills[i] /((float)tmpNumFreqProps * numRefinedSupport[i]);
		assert( fillRatio[i] > 0);
		refRatio[i] = (float) cs.numInRef / freqCSset->totalInRef;
		weight[i] = (float) cs.coverage * ( fillRatio[i] + refRatio[i]); 
		totalCov += cs.coverage;
			
		totalPrecision += fillRatio[i];
		overalFill += numRefinedFills[i];
		overalMaxFill += tmpNumFreqProps * numRefinedSupport[i];

		Q += weight[i];
	}
	printf("Refined Table: Performance metric Q = (weighting %f)/(totalCov %d * numTbl %d) \n", Q,totalCov, numTables);
	printf("Average precision = %f\n",(float)totalPrecision/numTables);
	printf("Overall precision = %f (overfill %lld / overalMaxFill %lld)\n", (float) overalFill/overalMaxFill, overalFill, overalMaxFill);

	Q = Q/((float)totalCov * numTables);

	printf("==> Performance metric Q = %f \n", Q);

	free(fillRatio); 
	free(refRatio); 
	free(weight); 
	free(numRefinedFills);
	free(numRefinedSupport);
	#if NO_OUTPUTFILE == 0
	freePropStat(propStat);
	#endif

}
#endif

static 
void getSampleBeforeMerging(int *ret, CSset *freqCSset, CSlabel* labels, BAT *sbat, BATiter si, BATiter pi, BATiter oi,  bat *mapbatid, oid maxCSoid, oid *subjCSMap, int maxNumPwithDup){

	 //Get SAMPLE DATA
	int numTables = 0; 
	int *csTblIdxMapping, *mfreqIdxTblIdxMapping, *mTblIdxFreqIdxMapping, *csFreqCSMapping;
	

	csTblIdxMapping = (int *) malloc (sizeof (int) * (maxCSoid + 1)); 
	initIntArray(csTblIdxMapping, (maxCSoid + 1), -1);

	csFreqCSMapping = (int *) malloc (sizeof (int) * (maxCSoid + 1));
	initIntArray(csFreqCSMapping, (maxCSoid + 1), -1);


	mfreqIdxTblIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded); 
	initIntArray(mfreqIdxTblIdxMapping , freqCSset->numCSadded, -1);

	mTblIdxFreqIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded);  // TODO: little bit reduntdant space
	initIntArray(mTblIdxFreqIdxMapping , freqCSset->numCSadded, -1);

	//Mapping from from CSId to TableIdx 
	printf("Init CS tableIdxMapping \n");
	initCSTableIdxMapping(freqCSset, csTblIdxMapping, csFreqCSMapping, mfreqIdxTblIdxMapping, mTblIdxFreqIdxMapping, &numTables, labels);


	#if NO_OUTPUTFILE == 0 
	getSampleData(ret, mapbatid, numTables, freqCSset, sbat, si, pi, oi, 
			mTblIdxFreqIdxMapping, labels, csTblIdxMapping, maxNumPwithDup, subjCSMap, 1);
	#endif


	free(csTblIdxMapping);
	free(mfreqIdxTblIdxMapping);
	free(mTblIdxFreqIdxMapping);
	free(csFreqCSMapping);

	
}


static
void RDFmergingTrial(CSset *freqCSset, CSrel *csrelSet, CSlabel** labels, oid maxCSoid, bat *mapbatid, OntoUsageNode *ontoUsageTree, float simTfidfThreshold, Pscore *pscore){

	oid		*mergeCSFreqCSMap; 
	int		curNumMergeCS = 0; 
	oid		mergecsId = 0; 
	int 		tmpNumRel = 0;
	CSrel		*tmpCSrelToMergeCS = NULL; 
	clock_t 	curT;
	clock_t		tmpLastT; 

	tmpLastT = clock(); 
	curNumMergeCS = countNumberMergeCS(freqCSset);
	//printf("Before using rules: Number of freqCS is: %d \n",curNumMergeCS);
	
	/* ---------- S1 ------- */
	mergecsId = maxCSoid + 1; 

	mergeFreqCSByS1(freqCSset, labels, &mergecsId, ontmetadata, ontmetadataCount, mapbatid); /*S1: Merge all freqCS's sharing top-3 candidates */
	
	curNumMergeCS = countNumberMergeCS(freqCSset);
	//printf("S1: Number of mergeCS: %d \n", curNumMergeCS);

	#if STORE_PERFORMANCE_METRIC_INFO	
	//computeMetricsQ(freqCSset);
	#endif
	
	/* ---------- S5 ------- */
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);
	
	/* S5: Merged CS referred from the same CS via the same property */
	tmpCSrelToMergeCS = generateCsRelToMergeFreqSet(csrelSet, freqCSset);
	tmpNumRel = freqCSset->numCSadded; 

	mergeFreqCSByS5(tmpCSrelToMergeCS, freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS,  &mergecsId, ontmetadata, ontmetadataCount);
	
	freeCSrelSet(tmpCSrelToMergeCS,tmpNumRel);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	//printf("S5: Number of mergeCS: %d \n", curNumMergeCS);
	#if STORE_PERFORMANCE_METRIC_INFO	
	//computeMetricsQ(freqCSset);
	#endif

	//S2: Common ancestor
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	mergeCSByS2(freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS, &mergecsId, ontoUsageTree, ontmetadata, ontmetadataCount, ontmetaBat, ontclassSet);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	//printf("S2: Number of mergeCS: %d \n", curNumMergeCS);

	#if STORE_PERFORMANCE_METRIC_INFO	
	//computeMetricsQ(freqCSset);
	#endif

	//S4: TF/IDF similarity
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	mergeCSByS4(freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS, &mergecsId, ontmetadata, ontmetadataCount);
	free(mergeCSFreqCSMap);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	//printf("S4: Number of mergeCS: %d \n", curNumMergeCS);

	#if STORE_PERFORMANCE_METRIC_INFO	
	printf("Metric scores for %f\n",simTfidfThreshold);
	*pscore = computeMetricsQ(freqCSset);
	#endif

	curT  = clock(); 
	printf ("Trial merging took %f. (Number of mergeCS: %d) \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS);	

}

static
void RDFmerging(CSset *freqCSset, CSrel *csrelSet, CSlabel** labels, oid maxCSoid,BAT *mbat, BAT *ontbat, bat *mapbatid, int freqThreshold, OntoUsageNode *ontoUsageTree){

	oid		*mergeCSFreqCSMap; 
	int		curNumMergeCS = 0; 
	oid		mergecsId = 0; 
	int 		tmpNumRel = 0;
	CSrel		*tmpCSrelToMergeCS = NULL; 
	clock_t 	curT;
	clock_t		tmpLastT; 

	tmpLastT = clock(); 
	curNumMergeCS = countNumberMergeCS(freqCSset);
	printf("Before using rules: Number of freqCS is: %d \n",curNumMergeCS);
	
	/* ---------- S1 ------- */
	mergecsId = maxCSoid + 1; 

	mergeFreqCSByS1(freqCSset, labels, &mergecsId, ontmetadata, ontmetadataCount, mapbatid); /*S1: Merge all freqCS's sharing top-3 candidates */
	
	curNumMergeCS = countNumberMergeCS(freqCSset);

	curT = clock(); 
	printf("Merging with S1 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));
	printf("Number of added CS after S1: %d \n", freqCSset->numCSadded);

	#if NO_OUTPUTFILE == 0
	printMergedFreqCSSet(freqCSset, mbat, ontbat, 1, freqThreshold, *labels, 1); 
	#endif

	#if STORE_PERFORMANCE_METRIC_INFO	
	computeMetricsQ(freqCSset);
	#endif
	tmpLastT = curT;
	
	/* ---- S3 --- */
	//Merge two CS's having the subset-superset relationship 
	if (0){
		mergeCSbyS3(freqCSset, labels, mergeCSFreqCSMap,curNumMergeCS, ontmetadata, ontmetadataCount, ontoUsageTree);
	}

	/* ---------- S5 ------- */
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);
	
	/* S5: Merged CS referred from the same CS via the same property */
	tmpCSrelToMergeCS = generateCsRelToMergeFreqSet(csrelSet, freqCSset);
	tmpNumRel = freqCSset->numCSadded; 

	mergeFreqCSByS5(tmpCSrelToMergeCS, freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS,  &mergecsId, ontmetadata, ontmetadataCount);

	freeCSrelSet(tmpCSrelToMergeCS,tmpNumRel);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf("Merging with S5 took %f. (Number of mergeCS: %d | NumconsistOf: %d) \n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS, countNumberConsistOfCS(freqCSset));

	#if NO_OUTPUTFILE == 0
	printMergedFreqCSSet(freqCSset, mbat, ontbat, 1, freqThreshold, *labels, 3); 
	#endif

	#if STORE_PERFORMANCE_METRIC_INFO	
	computeMetricsQ(freqCSset);
	#endif

	tmpLastT = curT; 		
	
	//S2: Common ancestor
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	mergeCSByS2(freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS, &mergecsId, ontoUsageTree, ontmetadata, ontmetadataCount, ontmetaBat, ontclassSet);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf ("Merging with S2 took %f. (Number of mergeCS: %d) \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS);	

	#if NO_OUTPUTFILE == 0
	printMergedFreqCSSet(freqCSset, mbat, ontbat, 1, freqThreshold, *labels, 4); 
	#endif

	#if STORE_PERFORMANCE_METRIC_INFO	
	computeMetricsQ(freqCSset);
	#endif

	tmpLastT = curT; 		


	//S4: TF/IDF similarity
	free(mergeCSFreqCSMap);
	mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);

	mergeCSByS4(freqCSset, labels, mergeCSFreqCSMap, curNumMergeCS, &mergecsId, ontmetadata, ontmetadataCount);
	free(mergeCSFreqCSMap);

	curNumMergeCS = countNumberMergeCS(freqCSset);
	curT = clock(); 
	printf ("Merging with S4 took %f. (Number of mergeCS: %d) \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC, curNumMergeCS);	

	#if NO_OUTPUTFILE == 0
	printMergedFreqCSSet(freqCSset, mbat,ontbat, 1, freqThreshold, *labels, 5); 
	#endif

	#if STORE_PERFORMANCE_METRIC_INFO	
	computeMetricsQ(freqCSset);
	#endif


}

static
CS* copyCS(CS *srcCS){
	
	CS *cs = (CS*)malloc(sizeof(CS)); 

	cs->numProp = srcCS->numProp; 
	cs->lstProp =  (oid*) malloc(sizeof(oid) * srcCS->numProp);

	copyOidSet(cs->lstProp, srcCS->lstProp, cs->numProp); 

	#if COLORINGPROP
	if (srcCS->lstPropSupport != NULL){
		cs->lstPropSupport =  (int*) malloc(sizeof(int) * cs->numProp);
		copyIntSet(cs->lstPropSupport,srcCS->lstPropSupport,cs->numProp);
	}
	else
		cs->lstPropSupport = NULL; 
	#endif

	cs->csId = srcCS->csId;
	cs->numAllocation = srcCS->numAllocation; 

	/*By default, this CS is not known to be a subset of any other CS*/
	#if STOREFULLCS
	cs->subject = srcCS->subject; 
	if (cs->subject != BUN_NONE){
		cs->lstObj =  (oid*) malloc(sizeof(oid) * cs->numProp);
		copyOidSet(cs->lstObj, srcCS->lstObj, cs->numProp); 
	}
	else
		cs->lstObj = NULL; 
	#endif

	cs->type = srcCS->type; 

	// This value is set for the 
	cs->parentFreqIdx = srcCS->parentFreqIdx; 
	cs->support = srcCS->support;
	cs->coverage = srcCS->coverage; 

	// For using in the merging process
	cs->numConsistsOf = srcCS->numConsistsOf;
	cs->lstConsistsOf = (int *) malloc(sizeof(int) * cs->numConsistsOf); 
	copyIntSet(cs->lstConsistsOf,srcCS->lstConsistsOf,cs->numConsistsOf);

	#if EXTRAINFO_FROM_RDFTYPE
	cs->typevalues = NULL; 
	cs->numTypeValues = 0;
	#endif

	#if STORE_PERFORMANCE_METRIC_INFO
	cs->numInRef = srcCS->numInRef;
	cs->numFill = srcCS->numFill;
	#endif

	return cs; 
}

static
CSset* copyCSset(CSset *srcCSset){
	int i; 
	CSset *csSet = (CSset*) malloc(sizeof(CSset));
	csSet->items = (CS*) malloc(sizeof(CS) * srcCSset->numAllocation);
	csSet->numAllocation = srcCSset->numAllocation;

	csSet->numCSadded = srcCSset->numCSadded;
	csSet->numOrigFreqCS = srcCSset->numOrigFreqCS;
	
	//copy each value
	
	for (i = 0; i < srcCSset->numCSadded; i++){
		CS  *srccs = NULL;
		CS *tmpcs = NULL;
	        srccs = &(srcCSset->items[i]);
		tmpcs =	copyCS(srccs); 

		csSet->items[i] = *tmpcs; 
	}

	#if STORE_PERFORMANCE_METRIC_INFO
	csSet->totalInRef = srcCSset->totalInRef;
	#endif
	return csSet;


}

static
void setFinalsimTfidfThreshold(Pscore *pscores, int numRun){
	int i; 


	printf("SimThreshold|avgPrecision|OvrallPrecision|Qscore|numTable|avgPrecisionFinal|OvrallPrecisionFinal|QscoreFinal|FinalTable|precRatio|tblRatio|precFinalRatio|finalTblRatio\n");
	for ( i = 0; i < numRun; i++){
		float numFinTblRatio = 1.0;
		float numTblRatio = 1.0;
		float precRatio = 1.0; 
		float precRatioFinal = 1.0;
		if (i > 0 && i < (numRun - 1)){
			numFinTblRatio = (float)(pscores[i+1].nFinalTable - pscores[i].nFinalTable)/(pscores[i].nFinalTable - pscores[i-1].nFinalTable);
			numTblRatio  = (float)(pscores[i+1].nTable - pscores[i].nTable)/(pscores[i].nTable - pscores[i-1].nTable);
			precRatio = (float)(pscores[i].overallPrec - pscores[i-1].overallPrec)/(pscores[i+1].overallPrec - pscores[i].overallPrec);
			precRatioFinal = (float)(pscores[i].overallPrecFinal - pscores[i-1].overallPrecFinal)/(pscores[i+1].overallPrecFinal - pscores[i].overallPrecFinal);
		}
		printf("%f|%f|%f|%f|%d|%f|%f|%f|%d|%f|%f|%f|%f\n",0.5 + i * 0.05,pscores[i].avgPrec, pscores[i].overallPrec, pscores[i].Qscore, pscores[i].nTable,
								     pscores[i].avgPrecFinal, pscores[i].overallPrecFinal, pscores[i].QscoreFinal, pscores[i].nFinalTable,
								     precRatio,numTblRatio, precRatioFinal, numFinTblRatio);
	}
	
	//Method 1: Get the point where the precision show a long tail of 10% in the graph
	/*
	{
	float cumgap;
	float totalgap; 
	totalgap = pscores[numRun-1].overallPrec - pscores[0].overallPrec;
	for ( i = 0; i < numRun; i++){	
		cumgap = pscores[i].overallPrec - pscores[0].overallPrec;
		if (cumgap > 0.9 * totalgap){
			simTfidfThreshold = 0.5 + i * 0.05;
			break;
		}
	}
	}
	*/

	//Method 2: Get the derivation for the normalized number of tables and the precision and 
	//then, find the intersecting point of them
	{
	float *numTableNormalized = NULL; 
	float *precisionNormalize = NULL; 
	float *precisionDelta = NULL; 
	float *nTblDelta = NULL; 
	int intersectingPoint = 1;
	if (pscores[numRun-1].nTable == pscores[0].nTable  || pscores[numRun-1].overallPrec == pscores[0].overallPrec){
		printf("Merging with different threshold doesnot show any different");
		simTfidfThreshold = 0.5 + (numRun - 1) * 0.05;
		return; 
	}

	numTableNormalized = (float*)malloc(sizeof(float) * numRun); 
	precisionNormalize = (float*)malloc(sizeof(float) * numRun); 
	precisionDelta = (float*)malloc(sizeof(float) * numRun); 
	nTblDelta = (float*)malloc(sizeof(float) * numRun);  

	//Normalize precision and number of tables
	printf("numRun|normalizedPrecision|normalizedTbl\n");
	for ( i = 0; i < numRun; i++){
		precisionNormalize[i] = (float)(pscores[i].overallPrec - pscores[0].overallPrec)/(pscores[numRun-1].overallPrec - pscores[0].overallPrec);		
		numTableNormalized[i] = (float)(pscores[i].nTable - pscores[0].nTable )/(pscores[numRun-1].nTable  - pscores[0].nTable );		
		printf("%d|%f|%f\n",i,precisionNormalize[i],numTableNormalized[i]);
	}
	
	//Get derivations
	printf("numRun|precisionDelta|nTblDelta\n");
	precisionDelta[0] = 0;
	nTblDelta[0] = 0;
	printf("0|0|0\n");
	for ( i = 1; i < numRun; i++){
		precisionDelta[i] = precisionNormalize[i] - precisionNormalize[i-1];
		nTblDelta[i] = numTableNormalized[i] - numTableNormalized[i-1];
		printf("%d|%f|%f\n",i,precisionDelta[i],nTblDelta[i]);
		//Find insterection point
		if (precisionDelta[i] == nTblDelta[i])  intersectingPoint = i; 
		else{
			if ((precisionDelta[i] - nTblDelta[i])*(precisionDelta[i-1] - nTblDelta[i-1]) < 0){
				intersectingPoint = i;	
			}
		}

	}
	simTfidfThreshold = 0.5 + intersectingPoint * 0.05;

	free(numTableNormalized);	
	free(precisionNormalize);
	free(nTblDelta); 
	free(precisionDelta); 
	}
}


/* Extract CS from SPO triples table */
str
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, bat *ontbatid, int *freqThreshold, void *_freqCSset, oid **subjCSMap, oid *maxCSoid, int *maxNumPwithDup, CSlabel** labels, CSrel **csRelMergeFreqSet){

	BAT 		*sbat = NULL, *pbat = NULL, *obat = NULL, *mbat = NULL; 
	BAT		*ontbat;  //Contain list of ontologies	
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

	CSset		*freqCSset; 
	clock_t 	curT;
	clock_t		tmpLastT; 
	OntoUsageNode	*ontoUsageTree = NULL;

	float		*curIRScores = NULL; 
	
	int		nIterIR = 3; 	//number of iteration for detecting dimension table with PR algorithm

	//printf("Number of type attributes is: %d \n",typeAttributesCount);

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if (!(sbat->tsorted)){
		BBPunfix(sbat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", "sbat is not sorted");
	}

	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}

	if ((ontbat = BATdescriptor(*ontbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	(void)BAThash(ontbat,0);
	if (!(ontbat->T->hash)){
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

	totalNumberOfTriples = 	BATcount(sbat); 
	acceptableTableSize = totalNumberOfTriples / (2 * upperboundNumTables);
	printf("Acceptable table size = %d \n", acceptableTableSize);
	
	tmpLastT = clock();

	*maxNumPwithDup	 = 0;
	//Phase 1: Assign an ID for each CS
	RDFassignCSId(ret, sbat, pbat, obat, ontbat,freqCSset, freqThreshold, csBats, *subjCSMap, maxCSoid, &maxNumProp, maxNumPwithDup);

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

	#if STORE_PERFORMANCE_METRIC_INFO
	setInitialMetricsInfo(refCount, freqCSset);		
	computeMetricsQ(freqCSset);
	#endif

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

	RDFrelationships(ret, sbat, pbat, obat, *subjCSMap, subjSubCSMap, csSubCSSet, csrelSet, *maxSoid, *maxNumPwithDup, csIdFreqIdxMap, freqCSset->numCSadded);
	#else
	RDFrelationships(ret, sbat, pbat, obat, *subjCSMap, csrelSet, *maxSoid, *maxNumPwithDup, csIdFreqIdxMap, freqCSset->numCSadded);
	#endif

	curT = clock(); 
	printf (" ----- Exploring subCSs and FKs took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	#if NO_OUTPUTFILE == 0
	printCSrelSet(csrelSet, freqCSset, freqCSset->numCSadded, *freqThreshold);  
	#endif

	#if NEEDSUBCS
	setdefaultSubCSs(csSubCSSet,*maxCSoid + 1, sbat, subjSubCSMap, *subjCSMap, subjdefaultMap);
	
	#if NO_OUTPUTFILE == 0 
	printSubCSInformation(csSubCSSet, csBats->freqBat, *maxCSoid + 1, 1, *freqThreshold); 
	#endif
	#endif

	printf("Number of frequent CSs is: %d \n", freqCSset->numCSadded);

	//createTreeForCSset(freqCSset); 	// DOESN'T HELP --> REMOVE
	
	/*get the statistic */
	//getTopFreqCSs(csMap,*freqThreshold);
	#if COUNT_NUMTYPES_PERPROP
	{
	
	/* Get possible types of each property in a table (i.e., mergedCS) */
	CSPropTypes *csPropTypes = (CSPropTypes*)GDKmalloc(sizeof(CSPropTypes) * (freqCSset->numCSadded)); 
	initCSPropTypesForBasicFreqCS(csPropTypes, freqCSset, freqCSset->numCSadded);
	
	printf("Extract CSPropTypes from basic CS's \n");
	RDFExtractCSPropTypes(ret, sbat, pbat, obat, *subjCSMap, csIdFreqIdxMap, csPropTypes, *maxNumPwithDup);
	printNumTypePerProp(csPropTypes, freqCSset->numCSadded, freqCSset);

	freeCSPropTypes(csPropTypes, freqCSset->numCSadded);
	
	curT = clock(); 
	printf (" Took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		
	}
	#endif

	/*Run merging for few times to get good parameters*/
	{
	int i = 0; 
	int numRun = 10; 
	Pscore   *pscores = NULL;
	pscores = (Pscore *) malloc (sizeof(Pscore) * numRun); 

	for ( i = 0; i < numRun; i++){
		simTfidfThreshold = 0.5 + i * 0.05;	
		printf("---- Trial with simTfidfThreshold=%f----\n",simTfidfThreshold);
		{
		CSlabel 	*tmplabels = NULL; 
		OntoUsageNode	*tmpontoUsageTree = NULL;
		CSset	*tmpFreqCSset = NULL; 
		
		tmpFreqCSset = copyCSset(freqCSset); 
		tmplabels = createLabels(tmpFreqCSset, csrelSet, tmpFreqCSset->numCSadded, sbat, si, pi, oi, *subjCSMap, csIdFreqIdxMap, ontattributes, ontattributesCount, ontmetadata, ontmetadataCount, &tmpontoUsageTree, ontmetaBat, ontclassSet);

		RDFmergingTrial(tmpFreqCSset,csrelSet, &tmplabels, *maxCSoid, mapbatid, tmpontoUsageTree,simTfidfThreshold,&(pscores[i])); 
	
		freeOntoUsageTree(tmpontoUsageTree);
		freeLabels(tmplabels, tmpFreqCSset);
		freeCSset(tmpFreqCSset);
		
		}
	}
	
	setFinalsimTfidfThreshold(pscores, numRun); 
	
	free(pscores); 
	}
	
	printf("The final simTfidfThreshold is %f\n",simTfidfThreshold);
	// Create label per freqCS

	printf("Using ontologies with %d ontattributesCount and %d ontmetadataCount \n",ontattributesCount,ontmetadataCount);
	
	(*labels) = createLabels(freqCSset, csrelSet, freqCSset->numCSadded, sbat, si, pi, oi, *subjCSMap, csIdFreqIdxMap, ontattributes, ontattributesCount, ontmetadata, ontmetadataCount, &ontoUsageTree, ontmetaBat, ontclassSet);
	
	curT = clock(); 
	printf("Done labeling!!! Took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT;

	#if COUNT_PERCENTAGE_ONTO_PROP_USED
	getOntologyContribution(freqCSset, *labels);
	#endif

	#if NO_OUTPUTFILE == 0
	printMergedFreqCSSet(freqCSset, mbat, ontbat,1, *freqThreshold, *labels, 0); 
	#endif
	
	if (0) getSampleBeforeMerging(ret, freqCSset, *labels, sbat, si, pi, oi, mapbatid, *maxCSoid, *subjCSMap, *maxNumPwithDup);

	RDFmerging(freqCSset,csrelSet, labels, *maxCSoid, mbat, ontbat, mapbatid, *freqThreshold, ontoUsageTree); 

	tmpLastT = curT; 		


	updateParentIdxAll(freqCSset); 

	//Finally, re-create mergeFreqSet
	
	*csRelMergeFreqSet = generateCsRelBetweenMergeFreqSet(csrelSet, freqCSset);

	#if NO_OUTPUTFILE == 0 
	printCSRel(freqCSset, *csRelMergeFreqSet, *freqThreshold);
	#endif
	
	curT = clock(); 
	printf ("Get the final relationships between mergeCS took %f. \n",((float)(curT - tmpLastT))/CLOCKS_PER_SEC);	
	tmpLastT = curT; 		
	
	#if NO_OUTPUTFILE == 0
	printmergeCSSet(freqCSset, *freqThreshold);
	//getStatisticCSsBySize(csMap,maxNumProp); 

	getStatisticCSsBySupports(csBats->pOffsetBat, csBats->freqBat, csBats->coverageBat, csBats->fullPBat, 1, *freqThreshold);
	#endif

	/* Get the number of indirect refs in order to detect dimension table */
	if(1)	{
	//nIterIR = getDiameter(3, freqCSset->numCSadded,*csRelMergeFreqSet);
	nIterIR = getDiameterExact(freqCSset->numCSadded,*csRelMergeFreqSet);
	if (nIterIR > MAX_ITERATION_NO) nIterIR = MAX_ITERATION_NO;

	refCount = (int *) malloc(sizeof(int) * (freqCSset->numCSadded));
	curIRScores = (float *) malloc(sizeof(float) * (freqCSset->numCSadded));
	
	initIntArray(refCount, freqCSset->numCSadded, 0); 

	getOrigRefCount(*csRelMergeFreqSet, freqCSset, freqCSset->numCSadded, refCount);  
	getIRNums(*csRelMergeFreqSet, freqCSset, freqCSset->numCSadded, refCount, curIRScores, nIterIR);  
	updateFreqCStype(freqCSset, freqCSset->numCSadded, curIRScores, refCount);

	free(refCount); 
	free(curIRScores);
	
	curT = clock(); 
	printf("Get number of indirect referrences to detect dimension tables !!! Took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT;
	}

	BBPunfix(sbat->batCacheid); 
	BBPunfix(pbat->batCacheid); 
	BBPunfix(obat->batCacheid);
	BBPunfix(mbat->batCacheid);
	BBPunfix(ontbat->batCacheid);

	freeOntoUsageTree(ontoUsageTree);
	
	#if NEEDSUBCS
	free (subjSubCSMap);
	free (subjdefaultMap);
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
		BBPunfix(sbat->batCacheid);
		throw(MAL, "rdf.RDFextractCS", RUNTIME_OBJECT_MISSING);
	}

	if (BATcount(pbat) == 0) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		throw(RDF, "rdf.RDFextractPfromPSO", "pbat must not be empty");
		/* otherwise, variable bt is not initialized and thus
		 * cannot be dereferenced after the BATloop below */
	}
	
	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 

	/* Init a hashmap */
	pMap = hashmap_new(); 
	curP = BUN_NONE; 
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
	ObjectType objType; 

	origobat = COLcopy(obat, obat->ttype, TRUE, TRANSIENT);
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
void getTblIdxFromS(oid Soid, int *tbidx, oid *baseSoid){
	
	*tbidx = (int) ((Soid >> (sizeof(BUN)*8 - NBITS_FOR_CSID))  &  ((1 << (NBITS_FOR_CSID-1)) - 1)) ;
	
	*baseSoid = Soid - ((oid) (*tbidx) << (sizeof(BUN)*8 - NBITS_FOR_CSID));

	*tbidx = *tbidx - 1; 

	//return freqCSid; 
}

int getTblId_from_S_simple(oid Soid){
	
	int tbidx = (int) ((Soid >> (sizeof(BUN)*8 - NBITS_FOR_CSID))  &  ((1 << (NBITS_FOR_CSID-1)) - 1)) ;
	
	tbidx = tbidx - 1; 

	return tbidx; 
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
	ppos = BUNfnd(rmap,pbt);
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
BAT* BATnewPropSet(int ht, int tt, BUN cap){
	BAT	*tmpBat = NULL; 	
	tmpBat = BATnew(ht, tt, cap, TRANSIENT); 
	tmpBat->T->nil = 0;
	tmpBat->T->nonil = 0;
	tmpBat->tkey = 0;
	tmpBat->tsorted = 0;
	tmpBat->trevsorted = 0;
	tmpBat->tdense = 0;
	return tmpBat;
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
	#if EVERYTHING_AS_OID==1
	mapObjBATtypes[URI] = TYPE_oid; 
	mapObjBATtypes[DATETIME] = TYPE_oid;
	mapObjBATtypes[INTEGER] = TYPE_oid; 
	mapObjBATtypes[DOUBLE] = TYPE_oid; 
	mapObjBATtypes[STRING] = TYPE_oid; 
	mapObjBATtypes[BLANKNODE] = TYPE_oid;
	mapObjBATtypes[MULTIVALUES] = TYPE_oid;
	#else
	mapObjBATtypes[URI] = TYPE_oid; 
	mapObjBATtypes[DATETIME] = TYPE_timestamp;
	mapObjBATtypes[INTEGER] = TYPE_int; 
	mapObjBATtypes[DOUBLE] = TYPE_dbl; 
	mapObjBATtypes[STRING] = TYPE_str; 
	mapObjBATtypes[BLANKNODE] = TYPE_oid;
	mapObjBATtypes[MULTIVALUES] = TYPE_oid;
	#endif
	
	printf("Start initCStables \n"); 
	// allocate memory space for cstablestat
	cstablestat->numTables = numTables; 
	cstablestat->lstbatid = (bat**) malloc(sizeof (bat*) * numTables); 
	cstablestat->lstfreqId = (int*) malloc(sizeof (int) * numTables); 
	initIntArray(cstablestat->lstfreqId, numTables, -1); 
	cstablestat->numPropPerTable = (int*) malloc(sizeof (int) * numTables); 

	cstablestat->pbat = BATnewPropSet(TYPE_void, TYPE_oid, smallbatsz);
	cstablestat->sbat = BATnewPropSet(TYPE_void, TYPE_oid, smallbatsz);
	cstablestat->obat = BATnewPropSet(TYPE_void, TYPE_oid, smallbatsz);
	BATseqbase(cstablestat->pbat, 0);
	BATseqbase(cstablestat->sbat, 0);
	BATseqbase(cstablestat->obat, 0);

	#if TRIPLEBASED_TABLE
	cstablestat->resbat = NULL; 
	cstablestat->repbat = NULL; 
	cstablestat->reobat = NULL; 
	#endif

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
		cstablestat->lstfreqId[i] = mTblIdxFreqIdxMapping[i]; 
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
				//cstablestat->lstcstable[i].colBats[colIdx] = BATnewPropSet(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], smallbatsz);
				cstablestat->lstcstable[i].colBats[colIdx] = BATnewPropSet(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], freqCSset->items[csPropTypes[i].freqCSId].support + 1);
				cstablestat->lstcstable[i].lstMVTables[colIdx].numCol = 0; 	//There is no MV Tbl for this prop
				//TODO: use exact size for each BAT
			}
			else{
				//cstablestat->lstcstable[i].colBats[colIdx] = BATnewPropSet(TYPE_void, TYPE_oid, smallbatsz);
				cstablestat->lstcstable[i].colBats[colIdx] = BATnewPropSet(TYPE_void, TYPE_oid, freqCSset->items[csPropTypes[i].freqCSId].support + 1);
				BATseqbase(cstablestat->lstcstable[i].colBats[colIdx], 0);	
				cstablestat->lstcstable[i].lstMVTables[colIdx].numCol = csPropTypes[i].lstPropTypes[j].numMvTypes;
				if (cstablestat->lstcstable[i].lstMVTables[colIdx].numCol != 0){
					cstablestat->lstcstable[i].lstMVTables[colIdx].colTypes = (ObjectType *)malloc(sizeof(ObjectType)* cstablestat->lstcstable[i].lstMVTables[colIdx].numCol);
					cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats = (BAT **)malloc(sizeof(BAT*) * cstablestat->lstcstable[i].lstMVTables[colIdx].numCol);
			
					mvColIdx = 0;	//Go through all types
					cstablestat->lstcstable[i].lstMVTables[colIdx].colTypes[0] = csPropTypes[i].lstPropTypes[j].defaultType; //Default type for this MV col
					//Init the first col (default type) in MV Table
					//cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[0] = BATnewPropSet(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], smallbatsz);
					cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[0] = BATnewPropSet(TYPE_void, mapObjBATtypes[(int)csPropTypes[i].lstPropTypes[j].defaultType], csPropTypes[i].lstPropTypes[j].propCover + 1);
					for (k = 0; k < MULTIVALUES; k++){
						if (k != (int) csPropTypes[i].lstPropTypes[j].defaultType && csPropTypes[i].lstPropTypes[j].TableTypes[k] == MVTBL){
							mvColIdx++;
							cstablestat->lstcstable[i].lstMVTables[colIdx].colTypes[mvColIdx] = (ObjectType)k;
							//cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[mvColIdx] = BATnewPropSet(TYPE_void, mapObjBATtypes[k], smallbatsz);
							cstablestat->lstcstable[i].lstMVTables[colIdx].mvBats[mvColIdx] = BATnewPropSet(TYPE_void, mapObjBATtypes[k],csPropTypes[i].lstPropTypes[j].propCover + 1);
						}	
					}

					//Add a bat for storing FK to the main table
					//cstablestat->lstcstable[i].lstMVTables[colIdx].keyBat = BATnewPropSet(TYPE_void, TYPE_oid, smallbatsz);
					cstablestat->lstcstable[i].lstMVTables[colIdx].keyBat = BATnewPropSet(TYPE_void, TYPE_oid, csPropTypes[i].lstPropTypes[j].propCover + 1);
					cstablestat->lstcstable[i].lstMVTables[colIdx].subjBat = BATnewPropSet(TYPE_void, TYPE_oid, csPropTypes[i].lstPropTypes[j].propCover + 1);
				}

				//BATseqbase(cstablestat->lstcstable[i].mvExBats[colIdx], 0);
			}
			

			//For ex-type columns
			#if CSTYPE_TABLE == 1
			for (t = 0; t < csPropTypes[i].lstPropTypes[j].numType; t++){
				if ( csPropTypes[i].lstPropTypes[j].TableTypes[t] == TYPETBL){
					//cstablestat->lstcstableEx[i].colBats[colExIdx] = BATnewPropSet(TYPE_void, mapObjBATtypes[t], smallbatsz);
					cstablestat->lstcstableEx[i].colBats[colExIdx] = BATnewPropSet(TYPE_void, mapObjBATtypes[t], freqCSset->items[csPropTypes[i].freqCSId].support + 1);
					//Set mainTblColIdx for ex-table
					cstablestat->lstcstableEx[i].colTypes[colExIdx] = (ObjectType)t; 
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
				BBPunfix(cstablestat->lstcstable[i].lstMVTables[j].subjBat->batCacheid); 
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
#if TRIPLEBASED_TABLE
	if (cstablestat->resbat != NULL) BBPunfix(cstablestat->resbat->batCacheid); 
	if (cstablestat->repbat != NULL) BBPunfix(cstablestat->repbat->batCacheid); 
	if (cstablestat->reobat != NULL) BBPunfix(cstablestat->reobat->batCacheid); 
#endif


	free(cstablestat->lstbatid); 
	free(cstablestat->lastInsertedS); 
	free(cstablestat->lstcstable); 
	#if CSTYPE_TABLE == 1
	free(cstablestat->lastInsertedSEx); 
	free(cstablestat->lstcstableEx);
	#endif
	free(cstablestat->lstfreqId); 
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
		setofBats[ptl.lstIdx[i]] = BATnew(HeadType, TailType, smallbatsz, TRANSIENT);	
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
			bunfastapp(curBat, ATOMnilptr(curBat->ttype));
		}
	}

	return MAL_SUCCEED; 
	
   bunins_failed:
	throw(RDF, "fillMissingvalues","Failed in fast inserting\n");
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
			#if STORE_ALL_EXCEPTION_IN_PSO
			continue; 
			#endif
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
		//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
		
		bunfastapp(tmpBat, ATOMnilptr(tmpBat->ttype));
		
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
			#if STORE_ALL_EXCEPTION_IN_PSO
			continue; 
			#endif
			tmpColExIdx = csPropTypes[tblIdx].lstPropTypes[propIdx].colIdxes[i]; 
			tmpBat = cstablestat->lstcstableEx[tblIdx].colBats[tmpColExIdx];
			//Fill all missing values from From to To
			for(k = from; k < to; k++){
				//printf("Append null to ex table: Col: %d \n", tmpColExIdx);
				//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
				bunfastapp(tmpBat, ATOMnilptr(tmpBat->ttype));
			}

			if (tblType == MAINTBL){
				//printf("Append null to not to-be-inserted col in ex table: Col: %d  (# colIdxEx = %d) \n", tmpColExIdx, colIdxEx);
				//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
				bunfastapp(tmpBat, ATOMnilptr(tmpBat->ttype));
			}
			else if (tmpColExIdx != colIdxEx){
				//printf("Append null to not to-be-inserted col in ex table: Col: %d (WHILE tblType = %d,  colIdxEx = %d) \n", tmpColExIdx, tblType, colIdxEx);
				//BUNappend(tmpBat, ATOMnilptr(tmpBat->ttype), TRUE);
				bunfastapp(tmpBat, ATOMnilptr(tmpBat->ttype));
			}
		}
		
	}

	return MAL_SUCCEED; 
	
   bunins_failed:
	throw(RDF, "fillMissingValueByNils","Failed in fast inserting\n");
}

#if EVERYTHING_AS_OID == 0
/*
 * Extend VALget for handling DATETIME 
 */
static 
void * VALgetExtend(ValPtr v, ObjectType objType, timestamp *ts){
	if (objType == DATETIME){
		convert_encodedLng_toTimestamp(v->val.lval, ts);
		return (void *) ts;
	} 
	else{
		return VALget(v);
	}

}

#else	/*EVERYTHING_AS_OID == 1*/

/*
 * Convert any type-specific value backto the oid 
 */
static 
void * VALgetExtend_alloid(ValPtr v, ObjectType objType, timestamp *ts, oid *obt){
	(void) v; 
	(void) objType; 
	(void) ts; 
	return obt; 
}

#endif

static
void getRealValue(ValPtr returnValue, oid objOid, ObjectType objType, BATiter mapi, BAT *mapbat){
	str 	objStr; 
	str	tmpStr; 
	BUN	bun; 	
	BUN	maxObjectURIOid =  ((oid)1 << (sizeof(BUN)*8 - NBITS_FOR_CSID - 1)) - 1; //Base on getTblIdxFromS
	oid     realUri;

	//printf("objOid = " BUNFMT " \n",objOid);
	if (objType == URI || objType == BLANKNODE){
		oid oldoid = objOid;
		objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));

		assert(oldoid == objOid); 
		
		if (objOid < maxObjectURIOid){
			//takeOid(objOid, &objStr); 		//TODO: Do we need to get URI string???
			//printf("From tokenizer URI object value: "BUNFMT " (str: %s) \n", objOid, objStr);
		}
		
		realUri = objOid;
		VALset(returnValue,TYPE_oid, &realUri);

		//else, this object value refers to a subject oid
		//IDEA: Modify the function for calculating new subject Id:
		//==> subjectID = TBLID ... tmpSoid .... 	      
	}
	else if (objType == STRING){
		objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */ 
		bun = BUNfirst(mapbat);
		objStr = (str) BUNtail(mapi, bun + objOid); 
		
		tmpStr = GDKmalloc(sizeof(char) * strlen(objStr) + 1);
		memcpy(tmpStr, objStr, sizeof(char) * strlen(objStr) + 1);
		VALset(returnValue, TYPE_str, tmpStr);

		//printf("From mapbat BATcount= "BUNFMT" at position " BUNFMT ": %s \n", BATcount(mapbat),  bun + objOid,objStr);
	} 
	else{	//DATETIME, INTEGER, DOUBLE  
		decodeValueFromOid(objOid, objType, returnValue);
	}
}

static
void updatePropTypeForRemovedTriple(CSPropTypes *csPropTypes, int* tmpTblIdxPropIdxMap, int tblIdx, oid *subjCSMap, int* csTblIdxMapping, oid sbt, oid pbt, oid *lastRemovedProp, oid* lastRemovedSubj, char isMultiToSingleProp){
	int tmptblIdx, tmpPropIdx;

	if (tblIdx == -1)
		tmptblIdx = csTblIdxMapping[subjCSMap[sbt]];
	else 
		tmptblIdx = tblIdx;

	tmpPropIdx = tmpTblIdxPropIdxMap[tmptblIdx];
	//if (tmptblIdx == 3 && tmpPropIdx == 51) printf("Removing <p> <s> : " BUNFMT "  |   " BUNFMT "\n",pbt,sbt);
	//Update PropTypes
	if (isMultiToSingleProp){
		csPropTypes[tmptblIdx].lstPropTypes[tmpPropIdx].propCover--;
		return; 
	}

	if (pbt != *lastRemovedProp || sbt != *lastRemovedSubj){
		csPropTypes[tmptblIdx].lstPropTypes[tmpPropIdx].propCover--;
		csPropTypes[tmptblIdx].lstPropTypes[tmpPropIdx].propFreq--;

		*lastRemovedProp = pbt;
		*lastRemovedSubj = sbt; 
	} 
	else{	//Multivalue
		csPropTypes[tmptblIdx].lstPropTypes[tmpPropIdx].propCover--;
	}
}

//Macro for inserting to PSO
#define insToPSO(pb, sb, ob, pbt, sbt, obt)	\
	do{					\
			bunfastapp(pb, pbt);	\
			bunfastapp(sb, sbt);	\
			bunfastapp(ob, obt);	\
	}while (0)

#if EVERYTHING_AS_OID == 1
str RDFdistTriplesToCSs_alloid(int *ret, bat *sbatid, bat *pbatid, bat *obatid,  bat *mbatid, bat *lmapbatid, bat *rmapbatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId, char *isLotsNullSubj, oid *subjCSMap, int* csTblIdxMapping){
	
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
	int	numEmptyBat = 0; 

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
	timestamp	ts; 
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

	oid	lastRemovedSubj = BUN_NONE; 
	oid	lastRemovedProp = BUN_NONE; 

	(void) isLotsNullSubj;

	maxOrigPbt = ((oid)1 << (sizeof(BUN)*8 - NBITS_FOR_CSID)) - 1; 
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "RDFdistTriplesToCSs",
				"could not open the tokenizer\n");
	}

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	if ((mbat = BATdescriptor(*mbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	if ((lmap = BATdescriptor(*lmapbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		BBPunfix(mbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	
	if ((rmap = BATdescriptor(*rmapbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		BBPunfix(mbat->batCacheid);
		BBPunfix(lmap->batCacheid);
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


		if (tblIdx == -1){	
			#if REMOVE_LOTSOFNULL_SUBJECT
			if (isLotsNullSubj[*sbt] == 0){
				// This is for irregular triples, put them to pso table
				insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
				//printf(" ==> To PSO \n");
				isFKCol = 0;
				continue; 
			}
			#else
				insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
				isFKCol = 0;
				continue;
			#endif
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
			ppos = BUNfnd(propStat->pBat, &origPbt);
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

		#if REMOVE_LOTSOFNULL_SUBJECT
		if (tblIdx == -1 && isLotsNullSubj[*sbt]){	
			// A lots-of-null subject
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			
			//Update propTypes
			updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap, tblIdx, subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

			continue; 
		}
		#endif

		objType = getObjType(*obt); 
		assert (objType != BLANKNODE);


		tmpPropIdx = tmpTblIdxPropIdxMap[tblIdx]; 

		defaultType = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].defaultType; 
		assert(defaultType != MULTIVALUES); 	
		#if STORE_ALL_EXCEPTION_IN_PSO
		if (objType != defaultType){
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			continue; 
		}
		#endif

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

				//Update propTypes
				updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap,tblIdx, subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

				continue; 
			}
			else{ //  
				getTblIdxFromO(*obt,&tmpOidTblIdx);
				if (tmpOidTblIdx != csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblId){
					//printf("Dirty FK at tbl %d | propId " BUNFMT " \n", tblIdx, *pbt);
					insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);

					//Update propTypes
					updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap,tblIdx, subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

					continue; 
				}
			}
		}

		//printf(" Tbl: %d   |   Col: %d \n", tblIdx, tmpColIdx);
		#if COUNT_DISTINCT_REFERRED_S
		isFKCol = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isFKProp; 
		#endif
		
		istmpMVProp = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isMVProp; 
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
				tmpHashBat = BATnew(TYPE_void, TYPE_oid, lastSubjId[tblIdx] + 1, TRANSIENT);
				
				if (tmpHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpHashBat");
				}	
				(void)BAThash(tmpHashBat,0);
				if (!(tmpHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for Bat");
				}

				if (BUNappend(tmpHashBat,obt, TRUE) == GDK_FAIL){		//Insert the first value
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpHashBat");
				}
				isCheckDone = 0; 
				numPKcols++;
			}
			#endif
			#if COUNT_DISTINCT_REFERRED_S
			if (isFKCol){
				initHashBatgz = (csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport > smallHashBatsz)?smallHashBatsz:csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport;
				tmpFKHashBat = BATnew(TYPE_void, TYPE_oid, initHashBatgz + 1, TRANSIENT);

				if (tmpFKHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpFKHashBat");
				}	
				(void)BAThash(tmpFKHashBat,0);
				if (!(tmpFKHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
				}
				if (BUNappend(tmpFKHashBat,obt, TRUE) == GDK_FAIL){		//The first value
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
				tmpFKHashBat = BATnew(TYPE_void, TYPE_oid, initHashBatgz + 1, TRANSIENT);

				if (tmpFKHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpFKHashBat");
				}	
				(void)BAThash(tmpFKHashBat,0);
				if (!(tmpFKHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
				}

				if (BUNappend(tmpFKHashBat,obt, TRUE) == GDK_FAIL){		//The first value
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
				tmpHashBat = BATnew(TYPE_void, TYPE_oid, lastSubjId[tblIdx] + 1, TRANSIENT);

				if (tmpHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpHashBat");
				}	
				(void)BAThash(tmpHashBat,0);
				if (!(tmpHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for Bat");
				}

				csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isPKProp = 1;  /* Assume that the object values are all unique*/

				if (BUNappend(tmpHashBat,obt, TRUE) == GDK_FAIL){		//Insert the first value
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
				tmpObjBun = BUNfnd(tmpHashBat,(ptr) obt);
				if (tmpObjBun == BUN_NONE){
					if (BUNappend(tmpHashBat,obt, TRUE) == GDK_FAIL){		//Insert the first value
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
				assert(tmpFKHashBat != NULL); 
				tmpFKRefBun = BUNfnd(tmpFKHashBat,(ptr) obt);
				if (tmpFKRefBun == BUN_NONE){

				       if (tmpFKHashBat->T->hash && BATcount(tmpFKHashBat) > 4 * tmpFKHashBat->T->hash->mask) {
						HASHdestroy(tmpFKHashBat);
						BAThash(tmpFKHashBat, 2*BATcount(tmpFKHashBat));

						if (!(tmpFKHashBat->T->hash)){
							throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
						}
					}
					if (BUNappend(tmpFKHashBat,obt, TRUE) == GDK_FAIL){		
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
					bunfastapp(tmpmvBat, VALgetExtend_alloid(&vrRealObjValue,objType, &ts, obt));
				}
				else{
					if (i == 0){	//The deafult type column
						//Check whether we can cast the value to the default type value
						if (rdfcast(objType, defaultType, &vrRealObjValue, &vrCastedObjValue) == 1){
							bunfastapp(tmpmvBat,VALgetExtend_alloid(&vrCastedObjValue, defaultType, &ts, obt)); 
							VALclear(&vrCastedObjValue);
						}
						else{
							bunfastapp(tmpmvBat,ATOMnilptr(tmpmvBat->ttype));
						}
					}
					else{
						bunfastapp(tmpmvBat,ATOMnilptr(tmpmvBat->ttype));
					 
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
				bunfastapp(tmpBat, &tmpmvValue);
				//BATprint(tmpBat);
				
				//Insert this "key" to the key column of mv table.
				tmpmvKey = tmpmvValue; 
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].keyBat,&tmpmvKey);

				//Insert the current subject oid of the main table to the subject
				//column of this mvtable
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].subjBat,sbt);
				
				tmplastInsertedS = (int)tmpSoid; 
				
				lastColIdx = tmpColIdx; 
				lastPropIdx = tmpPropIdx; 
				lasttblIdx = tblIdx;
				
				numMultiValues++;
			}
			else{
				//Repeat referred "key" in the key column of mvtable
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].keyBat,&tmpmvKey);

				//Insert the current subject oid of the main table to the subject
				//column of this mvtable
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].subjBat,sbt);
				
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

				//Update propTypes
				updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap, tblIdx,subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,1);

				continue; 
			}
		}


		tmpTableType = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].TableTypes[(int)objType]; 

		//printf("  objType: %d  TblType: %d \n", (int)objType,(int)tmpTableType);
		if (tmpTableType == PSOTBL){			//For infrequent type ---> go to PSO
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			//printf(" ==> To PSO \n");

			//Update propTypes
			updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap, tblIdx,subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

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
				bunfastapp(tmpBat, VALgetExtend_alloid(&vrCastedObjValue, defaultType,&ts, obt));
	
				VALclear(&vrCastedObjValue);
			}
			else{
				bunfastapp(tmpBat, ATOMnilptr(tmpBat->ttype));
			}

		}
		
		bunfastapp(curBat, VALgetExtend_alloid(&vrRealObjValue, objType,&ts, obt));
		
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

	numEmptyBat = 0;
	// Keep the batCacheId
	for (i = 0; i < cstablestat->numTables; i++){
		//printf("----- Table %d ------ \n",i );
		for (j = 0; j < cstablestat->numPropPerTable[i];j++){
			//printf("Column %d \n", j);
			cstablestat->lstbatid[i][j] = cstablestat->lstcstable[i].colBats[j]->batCacheid; 
			tmpBat = cstablestat->lstcstable[i].colBats[j];
			if (BATcount(tmpBat) == 0) {
				printf("Empty Bats at table %d column %d \n",i,j);
				numEmptyBat++;
				fillMissingvalues(tmpBat, (int)BATcount(tmpBat), (int)lastSubjId[i]);
			}
			if (j > 0) 
				if (BATcount(cstablestat->lstcstable[i].colBats[j]) > 0 &&
				    BATcount(cstablestat->lstcstable[i].colBats[j-1]) > 0){			
					assert(BATcount(cstablestat->lstcstable[i].colBats[j]) == BATcount(cstablestat->lstcstable[i].colBats[j-1]));
				}
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
	printf("Number of full empty bats %d \n",numEmptyBat);

	printf("Number of triples in PSO table is "BUNFMT"\n", BATcount(cstablestat->pbat));
	
	BBPunfix(sbat->batCacheid);
	BBPunfix(pbat->batCacheid);
	BBPunfix(obat->batCacheid);
	BBPunfix(mbat->batCacheid);
	BBPunfix(lmap->batCacheid);
	BBPunfix(rmap->batCacheid);

	free(tmpTblIdxPropIdxMap); 

	TKNZRclose(ret);

	return MAL_SUCCEED; 
	
   bunins_failed:
	throw(RDF, "RDFdistTriplesToCSs_alloid","Failed in fast inserting\n");
}

#else
str RDFdistTriplesToCSs(int *ret, bat *sbatid, bat *pbatid, bat *obatid,  bat *mbatid, bat *lmapbatid, bat *rmapbatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId, char *isLotsNullSubj, oid *subjCSMap, int* csTblIdxMapping){
	
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
	int	numEmptyBat = 0; 

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
	timestamp	ts; 
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

	oid	lastRemovedSubj = BUN_NONE; 
	oid	lastRemovedProp = BUN_NONE; 

	(void) isLotsNullSubj;

	maxOrigPbt = ((oid)1 << (sizeof(BUN)*8 - NBITS_FOR_CSID)) - 1; 
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "RDFdistTriplesToCSs",
				"could not open the tokenizer\n");
	}

	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	if ((mbat = BATdescriptor(*mbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}

	if ((lmap = BATdescriptor(*lmapbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		BBPunfix(mbat->batCacheid);
		throw(MAL, "rdf.RDFdistTriplesToCSs", RUNTIME_OBJECT_MISSING);
	}
	
	if ((rmap = BATdescriptor(*rmapbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid);
		BBPunfix(mbat->batCacheid);
		BBPunfix(lmap->batCacheid);
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


		if (tblIdx == -1){	
			#if REMOVE_LOTSOFNULL_SUBJECT
			if (isLotsNullSubj[*sbt] == 0){
				// This is for irregular triples, put them to pso table
				insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
				//printf(" ==> To PSO \n");
				isFKCol = 0;
				continue; 
			}
			#else
				insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
				isFKCol = 0;
				continue;
			#endif
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
			ppos = BUNfnd(propStat->pBat, &origPbt);
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

		#if REMOVE_LOTSOFNULL_SUBJECT
		if (tblIdx == -1 && isLotsNullSubj[*sbt]){	
			// A lots-of-null subject
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			
			//Update propTypes
			updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap, tblIdx, subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

			continue; 
		}
		#endif

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

				//Update propTypes
				updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap,tblIdx, subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

				continue; 
			}
			else{ //  
				getTblIdxFromO(*obt,&tmpOidTblIdx);
				if (tmpOidTblIdx != csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblId){
					//printf("Dirty FK at tbl %d | propId " BUNFMT " \n", tblIdx, *pbt);
					insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);

					//Update propTypes
					updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap,tblIdx, subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

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
				tmpHashBat = BATnew(TYPE_void, TYPE_oid, lastSubjId[tblIdx] + 1, TRANSIENT);
				
				if (tmpHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpHashBat");
				}	
				(void)BAThash(tmpHashBat,0);
				if (!(tmpHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for Bat");
				}

				if (BUNappend(tmpHashBat,obt, TRUE) == GDK_FAIL){		//Insert the first value
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot insert to tmpHashBat");
				}
				isCheckDone = 0; 
				numPKcols++;
			}
			#endif
			#if COUNT_DISTINCT_REFERRED_S
			if (isFKCol){
				initHashBatgz = (csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport > smallHashBatsz)?smallHashBatsz:csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].refTblSupport;
				tmpFKHashBat = BATnew(TYPE_void, TYPE_oid, initHashBatgz + 1, TRANSIENT);

				if (tmpFKHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpFKHashBat");
				}	
				(void)BAThash(tmpFKHashBat,0);
				if (!(tmpFKHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
				}
				if (BUNappend(tmpFKHashBat,obt, TRUE) == GDK_FAIL){		//The first value
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
				tmpFKHashBat = BATnew(TYPE_void, TYPE_oid, initHashBatgz + 1, TRANSIENT);

				if (tmpFKHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpFKHashBat");
				}	
				(void)BAThash(tmpFKHashBat,0);
				if (!(tmpFKHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
				}

				if (BUNappend(tmpFKHashBat,obt, TRUE) == GDK_FAIL){		//The first value
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
				tmpHashBat = BATnew(TYPE_void, TYPE_oid, lastSubjId[tblIdx] + 1, TRANSIENT);

				if (tmpHashBat == NULL){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot create new tmpHashBat");
				}	
				(void)BAThash(tmpHashBat,0);
				if (!(tmpHashBat->T->hash)){
					throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for Bat");
				}

				csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].isPKProp = 1;  /* Assume that the object values are all unique*/

				if (BUNappend(tmpHashBat,obt, TRUE) == GDK_FAIL){		//Insert the first value
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
				tmpObjBun = BUNfnd(tmpHashBat,(ptr) obt);
				if (tmpObjBun == BUN_NONE){
					if (BUNappend(tmpHashBat,obt, TRUE) == GDK_FAIL){		//Insert the first value
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
				assert(tmpFKHashBat != NULL); 
				tmpFKRefBun = BUNfnd(tmpFKHashBat,(ptr) obt);
				if (tmpFKRefBun == BUN_NONE){

				       if (tmpFKHashBat->T->hash && BATcount(tmpFKHashBat) > 4 * tmpFKHashBat->T->hash->mask) {
						HASHdestroy(tmpFKHashBat);
						BAThash(tmpFKHashBat, 2*BATcount(tmpFKHashBat));

						if (!(tmpFKHashBat->T->hash)){
							throw(RDF, "rdf.RDFdistTriplesToCSs", "Cannot allocate the hash for FK Bat");
						}
					}
					if (BUNappend(tmpFKHashBat,obt, TRUE) == GDK_FAIL){		
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
					bunfastapp(tmpmvBat, VALgetExtend(&vrRealObjValue,objType, &ts));
				}
				else{
					if (i == 0){	//The deafult type column
						//Check whether we can cast the value to the default type value
						if (rdfcast(objType, defaultType, &vrRealObjValue, &vrCastedObjValue) == 1){
							bunfastapp(tmpmvBat,VALgetExtend(&vrCastedObjValue, defaultType, &ts));
							VALclear(&vrCastedObjValue);
						}
						else{
							bunfastapp(tmpmvBat,ATOMnilptr(tmpmvBat->ttype));
						}
					}
					else{
						bunfastapp(tmpmvBat,ATOMnilptr(tmpmvBat->ttype));
					 
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
				bunfastapp(tmpBat, &tmpmvValue);
				//BATprint(tmpBat);
				
				//Insert this "key" to the key column of mv table.
				tmpmvKey = tmpmvValue; 
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].keyBat,&tmpmvKey);

				//Insert the current subject oid of the main table to the subject
				//column of this mvtable
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].subjBat,sbt);
				
				tmplastInsertedS = (int)tmpSoid; 
				
				lastColIdx = tmpColIdx; 
				lastPropIdx = tmpPropIdx; 
				lasttblIdx = tblIdx;
				
				numMultiValues++;
			}
			else{
				//Repeat referred "key" in the key column of mvtable
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].keyBat,&tmpmvKey);

				//Insert the current subject oid of the main table to the subject
				//column of this mvtable
				bunfastapp(cstablestat->lstcstable[tblIdx].lstMVTables[tmpColIdx].subjBat,sbt);
				
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

				//Update propTypes
				updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap, tblIdx,subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,1);

				continue; 
			}
		}


		tmpTableType = csPropTypes[tblIdx].lstPropTypes[tmpPropIdx].TableTypes[(int)objType]; 

		//printf("  objType: %d  TblType: %d \n", (int)objType,(int)tmpTableType);
		if (tmpTableType == PSOTBL){			//For infrequent type ---> go to PSO
			insToPSO(cstablestat->pbat,cstablestat->sbat, cstablestat->obat, pbt, sbt, obt);
			//printf(" ==> To PSO \n");

			//Update propTypes
			updatePropTypeForRemovedTriple(csPropTypes, tmpTblIdxPropIdxMap, tblIdx,subjCSMap, csTblIdxMapping, *sbt, *pbt, &lastRemovedProp, &lastRemovedSubj,0);

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
				bunfastapp(tmpBat, VALgetExtend(&vrCastedObjValue, defaultType,&ts));
	
				VALclear(&vrCastedObjValue);
			}
			else{
				bunfastapp(tmpBat, ATOMnilptr(tmpBat->ttype));
			}

		}
		
		bunfastapp(curBat, VALgetExtend(&vrRealObjValue, objType,&ts));
		
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

	numEmptyBat = 0;
	// Keep the batCacheId
	for (i = 0; i < cstablestat->numTables; i++){
		//printf("----- Table %d ------ \n",i );
		for (j = 0; j < cstablestat->numPropPerTable[i];j++){
			//printf("Column %d \n", j);
			cstablestat->lstbatid[i][j] = cstablestat->lstcstable[i].colBats[j]->batCacheid; 
			tmpBat = cstablestat->lstcstable[i].colBats[j];
			if (BATcount(tmpBat) == 0) {
				printf("Empty Bats at table %d column %d \n",i,j);
				numEmptyBat++;
				fillMissingvalues(tmpBat, (int)BATcount(tmpBat), (int)lastSubjId[i]);
			}
			if (j > 0) 
				if (BATcount(cstablestat->lstcstable[i].colBats[j]) > 0 &&
				    BATcount(cstablestat->lstcstable[i].colBats[j-1]) > 0){			
					assert(BATcount(cstablestat->lstcstable[i].colBats[j]) == BATcount(cstablestat->lstcstable[i].colBats[j-1]));
				}
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
	printf("Number of full empty bats %d \n",numEmptyBat);

	printf("Number of triples in PSO table is "BUNFMT"\n", BATcount(cstablestat->pbat));
	
	BBPunfix(sbat->batCacheid);
	BBPunfix(pbat->batCacheid);
	BBPunfix(obat->batCacheid);
	BBPunfix(mbat->batCacheid);
	BBPunfix(lmap->batCacheid);
	BBPunfix(rmap->batCacheid);

	free(tmpTblIdxPropIdxMap); 

	TKNZRclose(ret);

	return MAL_SUCCEED; 
	
   bunins_failed:
	throw(RDF, "RDFdistTriplesToCSs","Failed in fast inserting\n");
}
#endif

#if BUILDTOKENZIER_TO_MAPID 
/*
 * Since the of s,p,o oids are converted to table-based oids,
 * a bat that maps original tokenizer oids to the converted oid
 * need to be built. Then, when it comes a string, it can first look
 * at the tokenizer for its original oid. After that, it can
 * use this TKNZRMappingBat for receiving the converted Id. 
 * 
 * Input String --> Original TKNZR oid --> New Oid (TKNRZ_to_new_MapBAT)
 * New oid --> Original TKNZR oid --> Input String (New_to_TKNZR_MapBat)
 * 
 * To convert from mapId to tknz Id, we use the lmap, rmap bats. 
 *
 * */
static 
str buildTKNZRMappingBat(BAT *lmap, BAT *rmap){

	BAT	*tmpBat = NULL; 
	char*   schema = "rdf";
	int 	ret; 
	int	num = 0; 
	bat	mapBatId; 
	BAT	*tmpmapBat = NULL, *pMapBat = NULL, 
		*tmplmapBat = NULL, *tmprmapBat = NULL, 	
		*plmapBat = NULL, *prmapBat = NULL; 	
	
	str	bname = NULL, bnamelBat = NULL, bnamerBat = NULL; 
	bat	*lstCommits = NULL; 

	/* Check if the bat has already built */
	bname = (str) GDKmalloc(50 * sizeof(char));
	snprintf(bname, 50, "tknzr_to_map");

	bnamelBat = (str) GDKmalloc(50 * sizeof(char));
	snprintf(bnamelBat, 50, "map_to_tknz_left");

	bnamerBat = (str) GDKmalloc(50 * sizeof(char));
	snprintf(bnamerBat, 50, "map_to_tknz_right");


	mapBatId = BBPindex(bname); 
	if (mapBatId != 0){
		printf("The tokenizer-mapping-bat %s has been built \n", bname); 
		GDKfree(bname);
		return MAL_SUCCEED;
	}

	printf("No tokenizer-mapping-bat %s has been built yet\n",bname);
	
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
	}
	
	TKNZRgetTotalCount(&num);
	//create a tmpBat from 0 to number of TKNZR items
	tmpBat = BATnew(TYPE_void, TYPE_void , num + 1, TRANSIENT);
	BATsetcount(tmpBat,num);
	BATseqbase(tmpBat, 0);
	BATseqbase(BATmirror(tmpBat), 0);

	tmpBat->T->nonil = 1;
	tmpBat->tkey = 1;
	tmpBat->tsorted = 1;
	tmpBat->trevsorted = 0;
	tmpBat->tdense = 1;

	if (RDFpartialjoin(&mapBatId, &lmap->batCacheid, &rmap->batCacheid, &tmpBat->batCacheid) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in using RDFpartialjoin for tokenizer map bat");
	}

	if ((tmpmapBat = BATdescriptor(mapBatId)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	pMapBat = COLcopy(tmpmapBat, tmpmapBat->ttype, TRUE, PERSISTENT);

	if (BKCsetName(&ret, (int *) &(pMapBat->batCacheid), (const char*const*) &bname) != MAL_SUCCEED)
		throw(MAL, "tokenizer.open", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(pMapBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "tokenizer.open", OPERATION_FAILED);

	BATmode(pMapBat, PERSISTENT); 

	/*Make persistent bats for mappingId_to_tokenizerId lmap->rmap */
	tmplmapBat = COLcopy(rmap, rmap->ttype, TRUE, TRANSIENT); 
	tmprmapBat = COLcopy(lmap, lmap->ttype, TRUE, TRANSIENT); 
	RDFbisubsort(&tmplmapBat, &tmprmapBat); 

	plmapBat = COLcopy(tmplmapBat, tmplmapBat->ttype, TRUE, PERSISTENT);
	prmapBat = COLcopy(tmprmapBat, tmprmapBat->ttype, TRUE, PERSISTENT);

	plmapBat->tkey = 1; 
	plmapBat->tsorted = 1; 


	if (BKCsetName(&ret, (int *) &(plmapBat->batCacheid), (const char*const*) &bnamelBat) != MAL_SUCCEED)
		throw(MAL, "tokenizer.open", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(plmapBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "tokenizer.open", OPERATION_FAILED);

	if (BKCsetName(&ret, (int *) &(prmapBat->batCacheid), (const char*const*) &bnamerBat) != MAL_SUCCEED)
		throw(MAL, "tokenizer.open", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(prmapBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "tokenizer.open", OPERATION_FAILED);

	lstCommits = GDKmalloc(sizeof(bat) * 4); 
	lstCommits[0] = 0;
	lstCommits[1] = pMapBat->batCacheid;
	lstCommits[2] = plmapBat->batCacheid;
	lstCommits[3] = prmapBat->batCacheid;

	TMsubcommit_list(lstCommits,4);

	GDKfree(lstCommits);

	BBPunfix(tmpmapBat->batCacheid); 
	BBPunfix(pMapBat->batCacheid); 

	GDKfree(bname);

	TKNZRclose(&ret);

	
	return MAL_SUCCEED; 
}

#endif /* BUILDTOKENZIER_TO_MAPID*/

str
RDFreorganize(int *ret, CStableStat *cstablestat, CSPropTypes **csPropTypes, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, bat *ontbatid, int *freqThreshold, int *mode){

	CSset		*freqCSset; 	/* Set of frequent CSs */
	oid		*subjCSMap = NULL;  	/* Store the corresponding CS Id for each subject */
	oid 		maxCSoid = 0; 
	BAT		*sbat = NULL, *obat = NULL, *pbat = NULL, *mbat = NULL;
	BATiter		si,pi,oi,mi; 
	BUN		p,q; 
	BAT		*sNewBat, *lmap, *rmap, *oNewBat, *origobat, *pNewBat; 
	BUN		newId; 
	oid		*sbt; 
	oid		*lastSubjId; 	/* Store the last subject Id in each freqCS */
	//oid		*lastSubjIdEx; 	/* Store the last subject Id (of not-default type) in each freqCS */
	int		tblIdx; 
	#if REMOVE_LOTSOFNULL_SUBJECT
	int		freqIdx;		
	int		numSubjRemoved = 0;
	char		*isRefTables = NULL;
	#endif
	char		*isLotsNullSubj = NULL; 

	oid		lastS;
	oid		l,r; 
	bat		oNewBatid, pNewBatid; 
	int		*csTblIdxMapping;	/* Store the mapping from a CS id to an index of a maxCS or mergeCS in freqCSset. */
	int		*mfreqIdxTblIdxMapping;  /* Store the mapping from the idx of a max/merge freqCS to the table Idx */
	int		*mTblIdxFreqIdxMapping;  /* Invert of mfreqIdxTblIdxMapping */
	int		*csFreqCSMapping = NULL; 
	int		numTables = 0; 
	PropStat	*propStat; 
	int		numdistinctMCS = 0; 
	int		maxNumPwithDup = 0;
	//CStableStat	*cstablestat;
	CSlabel		*labels;
	CSrel		*csRelMergeFreqSet = NULL;
	CSrel		*csRelFinalFKs = NULL;   	//Store foreign key relationships 

	clock_t 	curT;
	clock_t		tmpLastT; 
	
	str		returnStr; 

	tmpLastT = clock();
	freqCSset = initCSset();

	//if (1) printListOntology();
	readParamsInput();
	
	printf("Min positive integer-encoded oid is: "BUNFMT"\n", MIN_POSI_INT_OID); 	
	printf("Max positive integer-encoded oid is: "BUNFMT"\n", MAX_POSI_INT_OID); 	
	printf("Min negative integer-encoded oid is: "BUNFMT"\n", MIN_NEGA_INT_OID); 	
	printf("Max negative integer-encoded oid is: "BUNFMT"\n", MAX_NEGA_INT_OID); 	

	if (RDFextractCSwithTypes(ret, sbatid, pbatid, obatid, mapbatid, ontbatid, freqThreshold, freqCSset,&subjCSMap, &maxCSoid, &maxNumPwithDup, &labels, &csRelMergeFreqSet) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in extracting CSs");
	}
	
	
	curT = clock(); 
	printf (" Total schema extraction process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	printf("Start re-organizing triple store for " BUNFMT " CSs \n", maxCSoid + 1);

	csTblIdxMapping = (int *) malloc (sizeof (int) * (maxCSoid + 1)); 
	initIntArray(csTblIdxMapping, (maxCSoid + 1), -1);
	
	csFreqCSMapping = (int *) malloc (sizeof (int) * (maxCSoid + 1));
	initIntArray(csFreqCSMapping, (maxCSoid + 1), -1);

	mfreqIdxTblIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded); 
	initIntArray(mfreqIdxTblIdxMapping , freqCSset->numCSadded, -1);

	mTblIdxFreqIdxMapping = (int *) malloc (sizeof (int) * freqCSset->numCSadded);  // TODO: little bit reduntdant space
	initIntArray(mTblIdxFreqIdxMapping , freqCSset->numCSadded, -1);

	//Mapping from from CSId to TableIdx 
	printf("Init CS tableIdxMapping \n");
	initCSTableIdxMapping(freqCSset, csTblIdxMapping, csFreqCSMapping, mfreqIdxTblIdxMapping, mTblIdxFreqIdxMapping, &numTables, labels);


	if ((sbat = BATdescriptor(*sbatid)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	if ((obat = BATdescriptor(*obatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	if ((pbat = BATdescriptor(*pbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(obat->batCacheid);
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	if ((mbat = BATdescriptor(*mapbatid)) == NULL) {
		BBPunfix(sbat->batCacheid);
		BBPunfix(obat->batCacheid);
		BBPunfix(pbat->batCacheid);
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	si = bat_iterator(sbat); 
	pi = bat_iterator(pbat); 
	oi = bat_iterator(obat); 
	mi = bat_iterator(mbat);

	/* Get possible types of each property in a table (i.e., mergedCS) */
	*csPropTypes = (CSPropTypes*)GDKmalloc(sizeof(CSPropTypes) * numTables); 
	initCSPropTypes(*csPropTypes, freqCSset, numTables, labels);
	
	printf("Extract CSPropTypes \n");
	RDFExtractCSPropTypes(ret, sbat, pbat, obat,  subjCSMap, csTblIdxMapping, *csPropTypes, maxNumPwithDup);
	genCSPropTypesColIdx(*csPropTypes, numTables, freqCSset);

	#if NO_OUTPUTFILE == 0
	printCSPropTypes(*csPropTypes, numTables, freqCSset, *freqThreshold);
	//Collecting the statistic
	printf("Get table statistics by CSPropTypes \n");
	getTableStatisticViaCSPropTypes(*csPropTypes, numTables, freqCSset, *freqThreshold);
	#endif
	
	#if COLORINGPROP
	/* Update list of support for properties in freqCSset */
	updatePropSupport(*csPropTypes, numTables, freqCSset);
	#if NO_OUTPUTFILE == 0
	printFinalTableWithPropSupport(*csPropTypes, numTables, freqCSset, mapbatid, *freqThreshold, labels);
	#endif
	#endif

	curT = clock(); 
	printf (" Preparing process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	csRelFinalFKs = getFKBetweenTableSet(csRelMergeFreqSet, freqCSset, *csPropTypes,mfreqIdxTblIdxMapping,numTables, labels);
	#if NO_OUTPUTFILE == 0
	printFKs(csRelFinalFKs, *freqThreshold, numTables, *csPropTypes); 
	#endif

	// Init CStableStat
	initCStables(cstablestat, freqCSset, *csPropTypes, numTables, labels, mTblIdxFreqIdxMapping);
	
	// Summarize the statistics
	#if NO_OUTPUTFILE == 0
	getStatisticFinalCSs(freqCSset, sbat, *freqThreshold, numTables, mTblIdxFreqIdxMapping, *csPropTypes, labels);
	#endif	

	/* Extract sample data for the evaluation */

	#if NO_OUTPUTFILE == 0 
	getSampleData(ret, mapbatid, numTables, freqCSset, sbat, si, pi, oi, mTblIdxFreqIdxMapping, labels, csTblIdxMapping, maxNumPwithDup, subjCSMap, 2);
	#endif
	
	// print labels
	printf("Start exporting labels \n"); 
	
	#if EXPORT_LABEL
	exportLabels(freqCSset, csRelFinalFKs, *freqThreshold, mi, mbat, cstablestat, *csPropTypes, numTables, mTblIdxFreqIdxMapping, csTblIdxMapping);
	#endif

	curT = clock(); 
	printf (" Export label process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	#if NO_OUTPUTFILE == 0 
	printFinalStructure(cstablestat, *csPropTypes, numTables,*freqThreshold, mapbatid);
	#endif
	
	#if DETECT_INCORRECT_TYPE_SUBJECT
	{
	LabelStat *labelStat;
	BAT       *ontbat;

	if ((ontbat = BATdescriptor(*ontbatid)) == NULL) {
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}
	(void)BAThash(ontbat,0);
	if (!(ontbat->T->hash)){
		throw(MAL, "rdf.RDFextractCSwithTypes", RUNTIME_OBJECT_MISSING);
	}



	labelStat = initLabelStat();

	#if USING_FINALTABLE
	buildLabelStatForTable(labelStat, numTables, cstablestat);
	#else
	buildLabelStatForFinalMergeCS(labelStat, freqCSset, labels); 	
	#endif

	RDFcheckWrongTypeSubject(sbat, si, pi, oi, freqCSset, maxNumPwithDup, numTables, mTblIdxFreqIdxMapping, labelStat, subjCSMap, csFreqCSMapping);
	freeLabelStat(labelStat);
	}
	#endif
	
	#if STORE_PERFORMANCE_METRIC_INFO
	computeMetricsQForRefinedTable(freqCSset, *csPropTypes,mfreqIdxTblIdxMapping,mTblIdxFreqIdxMapping,numTables);
	#endif


	#if DUMP_CSSET
	{
		str schema = "rdf"; 
		if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
			throw(RDF, "rdf.rdfschema",
				"could not open the tokenizer\n");
		}
	
		dumpFreqCSs(cstablestat, freqCSset, mi, mbat); 

		TKNZRclose(ret);
	}
	#endif

	if (*mode == EXPLOREONLY){
		printf("Only explore the schema information \n");
		freeLabels(labels, freqCSset);
		freeCSrelSet(csRelMergeFreqSet,freqCSset->numCSadded);
		freeCSset(freqCSset); 
		free(subjCSMap);
		free(csTblIdxMapping);
		free(csFreqCSMapping);
		free(mfreqIdxTblIdxMapping);
		free(mTblIdxFreqIdxMapping);
		//freeCSPropTypes(*csPropTypes,numTables);
		freeCSrelSet(csRelFinalFKs, numTables);
		printf("Finish & Exit exploring step! \n"); 
		
		return MAL_SUCCEED;
	}




	sNewBat = BATnew(TYPE_void, TYPE_oid, BATcount(sbat), TRANSIENT);
	if (sNewBat== NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	BATseqbase(sNewBat, 0);
	
	lmap = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);

	if (lmap == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	lmap->tsorted = TRUE;

	BATseqbase(lmap, 0);
	
	rmap = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
	if (rmap == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	BATseqbase(rmap, 0);
	
	lastSubjId = (oid *) malloc (sizeof(oid) * cstablestat->numTables); 
	initArray(lastSubjId, cstablestat->numTables, -1); 
	
	#if REMOVE_LOTSOFNULL_SUBJECT
	//TODO: Find the better way than using isLotsNullSubj array to keep
	//the status of subject
	//
	//If the to-be-removed subject is referred to by an FK, it will make the
	//FK violated. Hence, either the FK should be set isDirty or the subject of
	//non-dirty FK is not removed. We follow the latter approach: Do not remove
	//the subject if it is referred to by a NON-dirty FK. (Dirty FK will be re-checked, thus, 
	//the triples refers to removed subjects will go to PSO).
	//
	isRefTables = (char *)malloc(sizeof(char) * cstablestat->numTables);
	isLotsNullSubj = (char *) malloc(sizeof(char) * BATcount(sbat) + 1);
	initCharArray(isLotsNullSubj, BATcount(sbat) + 1,0);
	initCharArray(isRefTables, cstablestat->numTables, 0); 
	getRefTables(*csPropTypes, cstablestat->numTables, isRefTables);
	#else
	(void) isLotsNullSubj;
	#endif

	printf("Re-assigning Subject oids ... \n");
	lastS = -1; 
	BATloop(sbat, p, q){
		sbt = (oid *) BUNtloc(si, p);
		tblIdx = csTblIdxMapping[subjCSMap[*sbt]];
		
		#if REMOVE_LOTSOFNULL_SUBJECT
		//TODO: If the subject is the target 
		// of an FK prop, do not remove that subject. This is hard to check.
		//
		if (tblIdx != -1 && isRefTables[tblIdx] != 1){
			freqIdx = csFreqCSMapping[subjCSMap[*sbt]];
			if (freqCSset->items[freqIdx].numProp < cstablestat->lstcstable[tblIdx].numCol * LOTSOFNULL_SUBJECT_THRESHOLD){
				//printf("Subject " BUNFMT " is removed from table %d with %d cols \n",*sbt,tblIdx, cstablestat->lstcstable[tblIdx].numCol);
				isLotsNullSubj[*sbt] = 1;
				tblIdx = -1;
				numSubjRemoved++;
			}
		}
		#endif			

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

				BUNappend(lmap, &l, TRUE);
				BUNappend(rmap, &r, TRUE);

			}

		}
		else{	// Does not belong to a freqCS. Use original subject Id
			newId = *sbt; 
		}

		BUNappend(sNewBat, &newId, TRUE);	
		//printf("Tbl: %d  || Convert s: " BUNFMT " to " BUNFMT " \n", tblIdx, *sbt, newId); 
		
	}

        #if REMOVE_LOTSOFNULL_SUBJECT
	printf("Number of subject removed is: %d \n", numSubjRemoved);
	#endif
	
	#if BUILDTOKENZIER_TO_MAPID
	buildTKNZRMappingBat(lmap, rmap); 
	#endif
	
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
	if (RDFtriplesubsort(&pNewBat, &sNewBat, &oNewBat) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in sorting PSO");	
	}	
	printf("Done  \n");

	#if TRIPLEBASED_TABLE
	printf("Build triple-based relational BATs .."); 
	cstablestat->resbat = COLcopy(sNewBat, sNewBat->ttype, TRUE, TRANSIENT);	
	cstablestat->repbat = COLcopy(pNewBat, pNewBat->ttype, TRUE, TRANSIENT);	
	cstablestat->reobat = COLcopy(oNewBat, oNewBat->ttype, TRUE, TRANSIENT);	
	if (RDFtriplesubsort(&cstablestat->repbat, &cstablestat->resbat, &cstablestat->reobat) != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in sorting reorganized PSO");
	}
	//Set the property for the BAT
	cstablestat->repbat->tsorted = 1;
	printf("Done\n");
	#endif

	//BATprint(pNewBat);
	//BATprint(sNewBat);

	propStat = getPropStatisticsByTable(numTables, mTblIdxFreqIdxMapping, freqCSset,  &numdistinctMCS); 
	
	//printPropStat(propStat,0); 
	curT = clock(); 
	printf (" Prepare and create sub-sorted PSO took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		
	#if EVERYTHING_AS_OID == 1
	returnStr = RDFdistTriplesToCSs_alloid(ret, &sNewBat->batCacheid, &pNewBat->batCacheid, &oNewBat->batCacheid, mapbatid, 
			&lmap->batCacheid, &rmap->batCacheid, propStat, cstablestat, *csPropTypes, lastSubjId, isLotsNullSubj, subjCSMap, csTblIdxMapping);
	#else
	returnStr = RDFdistTriplesToCSs(ret, &sNewBat->batCacheid, &pNewBat->batCacheid, &oNewBat->batCacheid, mapbatid, 
			&lmap->batCacheid, &rmap->batCacheid, propStat, cstablestat, *csPropTypes, lastSubjId, isLotsNullSubj, subjCSMap, csTblIdxMapping);
	#endif
	printf("Return value from RDFdistTriplesToCSs is %s \n", returnStr);
	if (returnStr != MAL_SUCCEED){
		throw(RDF, "rdf.RDFreorganize", "Problem in distributing triples to BATs using CSs");		
	}
		
	curT = clock(); 
	printf ("RDFdistTriplesToCSs process took  %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		
	
	#if NO_OUTPUTFILE == 0
	printFKMultiplicityFromCSPropTypes(*csPropTypes, numTables, freqCSset, *freqThreshold);
	#endif
	
	#if NO_OUTPUTFILE == 0
	{
	int curNumMergeCS = countNumberMergeCS(freqCSset);
	oid* mergeCSFreqCSMap = (oid*) malloc(sizeof(oid) * curNumMergeCS);
	PropStat *propStat2;
        initMergeCSFreqCSMap(freqCSset, mergeCSFreqCSMap);
	propStat2 = initPropStat();
	getPropStatisticsFromMergeCSs(propStat2, curNumMergeCS, mergeCSFreqCSMap, freqCSset);

	#if EVERYTHING_AS_OID 
	if (0) 
	#endif	
	getFullSampleData(cstablestat, *csPropTypes, mTblIdxFreqIdxMapping, labels, numTables, &lmap->batCacheid, &rmap->batCacheid, freqCSset, mapbatid, propStat2);

	freePropStat(propStat2);
	free(mergeCSFreqCSMap);
	}
	#endif	
	#if REMOVE_LOTSOFNULL_SUBJECT	
	free(isLotsNullSubj);
	#endif
	freeCSrelSet(csRelMergeFreqSet,freqCSset->numCSadded);
	freeCSrelSet(csRelFinalFKs, numTables); 
	//freeCSPropTypes(*csPropTypes,numTables);
	freeLabels(labels, freqCSset);
	freeCSset(freqCSset); 
	free(subjCSMap); 
	free(csTblIdxMapping);
	free(csFreqCSMapping);
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
	BBPunfix(mbat->batCacheid);

	return MAL_SUCCEED; 
}
