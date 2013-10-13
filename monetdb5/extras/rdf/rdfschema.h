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

#ifndef _RDFSCHEMA_H_
#define _RDFSCHEMA_H_

#include <sql_catalog.h>

rdf_export str
RDFSchemaExplore(int *ret, str *tbname, str *clname);

rdf_export str
RDFextractCS(int *ret, bat *sbatid, bat *pbatid, int *freqThreshold); 

rdf_export str
RDFextractPfromPSO(int *ret, bat *pbatid, bat *sbatid); 

struct CSmergeRel; // forward declaration


typedef enum{
	EXPLOREONLY, 
	REORGANIZE
} ExpMode; 

typedef enum{
	MAINTBL, 
	TYPETBL,
	PSOTBL,
	MVTBL
} TableType; 		

typedef enum {
	NORMALCS, 
	FREQCS, 
	MAXCS, 
	MERGECS
} CStype; 

typedef struct {
	BAT*	hsKeyBat; 
	BAT* 	hsValueBat; 
	BAT* 	freqBat;    	/* Store the frequency of each Characteristic set */	
	BAT* 	coverageBat;	/* Store the exact number of triples coverred by each CS */
	BAT* 	pOffsetBat; 	/* BAT storing the offset for set of properties, refer to fullPBat */
	BAT* 	fullPBat;    	/* Stores all set of properties */	
} CSBats; 	// BATs for storing all information about CSs

typedef struct Postinglist{
	int*	lstIdx; 	/* List of CS containing the property */
	int*	lstInvertIdx; 	/* List of property's index in each CS */
	int	numAdded;
	int	numAllocation; 
} Postinglist; 

/* Statistic about the properties */
typedef struct PropStat {
	BAT*		pBat; 		/* Store the list of properties */
	int* 		freqs; 		/* Store number of CSs containing that property */
	float*		tfidfs; 
	int		numAllocation; 
	int		numAdded; 
	Postinglist* 	plCSidx;	/* Store posting list of CS index */				
	int		maxNumPPerCS;   /* Maximum number of different properties in one CS */
} PropStat; 			

#define INIT_PROP_NUM	10
#define INIT_CS_PER_PROP 10
#define	USINGTFIDF	0

#define STOREFULLCS     1       /* Store full instance of a CS including the a subject and list of predicates, objects. 
                                  Only use this for finding the name of the table corresponding to that CS */

#define NBITS_FOR_CSID	15	/* Use bits from 62th bit --> (62 - NBITS_FOR_CSID) for encoding the CSId in each SubjectId */

#define CSTYPE_TABLE	1

#define FULL_PROP_STAT 1	// Only use for showing the statistic on all properties / all CSs. (Default should be 0)

#define STAT_ANALYZE 1	// Only use for collecting the statistic on the number of multi/null/single-valued prop

#define COLORINGPROP 1	// Only use for coloring property in schema representation. 

#define NEEDSUBCS 0	// We actually do not need to use SubCS as the idea of default subCS is not used. But it is still good
			// for collecting the statistical information (For reporting/writing)

#define USE_LABEL_FINDING_MAXCS	0 	// Use the labels received from labeling process for finding maxCS 
#define USE_LABEL_FOR_MERGING	1 	// Use the labels received from labeling process for finding mergeCS
#define TOPK 1			//Check top 3 candidate
#define MAX_SUB_SUPER_NUMPROP_DIF 3
#define USE_MULTIWAY_MERGING	0

#define MINIMIZE_CONSISTSOF	1	/*Only store the minimize list of consistsof CS's Id. 
					Specifically, the consistsOf list only contains 
					the freqIdx of merged CS from previous rule. */

#define OUTPUT_FREQID_PER_LABEL 1 	/* This is for evaluating the results of merging using S1. TODO: Set it to 0 for default*/

#define IS_MULVALUE_THRESHOLD  1.1	/* The ratio betweeen (the number of triple coverred by Prop P) / (number of Non-NULL object values for P)
					   If this ratio is ~1, only use single value column for that prop
					*/
#define INFREQ_TYPE_THRESHOLD  0.1	/* Threshold that a type is consider as an infrequent type */

typedef struct CS
{
	oid 	csId;		//Id of the CS
	oid*	lstProp;	//List of properties' Ids
	#if	COLORINGPROP	
	int*	lstPropSupport;	 //Number of subjects that the object value for this is not null
	#endif
	int	numProp;
	int	numAllocation;
	//char 	isSubset; 
	int	parentFreqIdx; 	//Index of the parent in freqCSset
	#if STOREFULLCS
	oid     subject;         //A subject
	oid*    lstObj;          //List of sample objects
	#endif
	
	char	type; 
	int 	support; 
	int	coverage; 

	//For mergeCS
	int* 	lstConsistsOf; 	//The list of indexes of freqCS	
	int	numConsistsOf; 
} CS;


typedef struct SubCS {
	//oid 	csId; 
	oid	subCSId;
	oid 	sign;		// Signature generated from subTypes for quick comparison
	char*	subTypes;
	int	numSubTypes; 
	char	isdefault; 
} SubCS; 

/*
typedef struct mergeCS {	// CS formed by merging CS id1 and CS id2	
	oid* 	lstConsistsOf; 	
	int	numConsistsOf; 
	oid*	lstProp; 
	int	numProp; 
	int 	support;
	int	coverage;
	char 	isRemove;

} mergeCS; 

*/

#define INIT_NUM_SUBCS 4

typedef struct SubCSSet{
	oid	csId; 
	SubCS	*subCSs; 
	int	*freq; 
	int	numSubCS; 
	int 	numAllocation; 
} SubCSSet;

#define INIT_NUM_CS 9999 // workaround
#define SIM_THRESHOLD 0.6
#define SIM_TFIDF_THRESHOLD 0.55
#define IMPORTANCE_THRESHOLD 0.01
#define MIN_PERCETAGE_S6 5	// Merge all CS refered by more than 1/MIN_PERCETAGE_S6 percent of a CS via one property
#define MIN_FROMTABLE_SIZE_S6 100  // The minimum size of the "from" table in S6. Meaning that 
				    // the CS's to-be-merged in this rule must cover > MIN_FROMTABLE_SIZE_S6 / MIN_PERCETAGE_S6 triples
#define MINIMUM_TABLE_SIZE 10000   //The minimum number of triples coverred by a table (i.e., a final CS) 
#define SAMPLE_FILTER_THRESHOLD 1  // SAMPLE_FILTER_THRESHOLD/ 100	

typedef struct CSset{
	CS* items;
	int numOrigFreqCS; 
	int numCSadded;
	int numAllocation;
} CSset; 

/*
typedef struct mergeCSset{
	mergeCS* items; 
	int nummergeCSadded; 
	int numAllocation; 
} mergeCSset;
*/

#define INIT_NUM_CSREL 4
typedef struct CSrel{	
	oid	origFreqIdx;
	oid	*lstRefFreqIdx; 
	oid* 	lstPropId; 	// Predicate for a relationship
	int* 	lstCnt; 		// Count per reference
	int* 	lstBlankCnt;	// Count # links to blank node
	int  	numRef; 
	int  	numAllocation; 
} CSrel;

typedef struct CSrelSum{
	oid     origFreqIdx;
	int	numProp;		
	oid 	*lstPropId;
	int	*numPropRef; 	//Number of references per prop
	int	**freqIdList; 			
} CSrelSum;

#define INIT_DISTINCT_LABEL 400
typedef struct LabelStat{	/*Store the list of freqIds having the same label*/
	BAT	*labelBat; 
	int	*lstCount; 	/* Number of items per name */
	int	**freqIdList; 
	int	numLabeladded; 
	int	numAllocation; 
} LabelStat; 

typedef struct CStable {
	BAT**		colBats; 
	ObjectType*	colTypes; 
	BAT** 		mvBats; 	/* One bat for one Muti-values property */
	BAT**		mvExBats; 	/* One more bat for infrequent datatype in multi-valued prop*/	
	int		numCol; 
	oid* 		lstProp;
} CStable; 


typedef struct CStableEx {		/* For non-default-type columns*/
	BAT**		colBats; 
	ObjectType*	colTypes; 
	int		numCol; 
} CStableEx; 

typedef struct CStableStat {
	bat**   	lstbatid;
	int		numTables;
	int*		numPropPerTable; 
	//int* 		freqIdx; 	//Idx of the corresponding freqCS for a table
	oid**		lastInsertedS;	//Last S for each column
	//sql_schema*	schema; 	
	CStable*	lstcstable; 
	#if CSTYPE_TABLE
	CStableEx*        lstcstableEx;
	oid**		lastInsertedSEx; 
	#endif
	BAT*		pbat; 
	BAT*		sbat; 
	BAT*		obat; 
} CStableStat; 

typedef struct PropTypes{
	oid	prop;
	int	numType; 
#if STAT_ANALYZE	
	int	numMVType;	/* Number of subjects having this property a multi-valued prop. */
	int	numNull;	/* Number of subjects that don't have obj value for this prop */
	int	numSingleType;	/* Number of subjects having the */
#endif	
	int	propFreq; 	/* without considering type = Table frequency*/
	int	propCover; 	/* = coverage of that property */	
	char*	lstTypes; 
	int*	lstFreq; 
	int* 	lstFreqWithMV; 	/* Frequency of each type considering types in MV property*/
	int*	colIdxes; 
	char*	TableTypes;
	char	defaultType; 
	char	isMVProp; 	/* = 1 if this prop is a multi-valued prop*/
} PropTypes; 

typedef struct CSPropTypes {
	int		freqCSId; 
	int		numProp; 
	int 		numNonDefTypes; 
	PropTypes*	lstPropTypes; 
} CSPropTypes; 

#define NUM_SAMPLETABLE 20
#define	NUM_SAMPLE_INSTANCE 10
#define NUM_SAMPLE_CANDIDATE 3
typedef struct CSSample{
	int	freqIdx;
	int 	tblIdx; 
	oid	*candidates;
	oid	candidateCount; 
	int	numProp; 
	oid	*lstProp; 
	oid	*lstSubjOid;
	oid	**lstObj; 
	int	numInstances; 
	oid	name;
} CSSample;

rdf_export str
RDFdistTriplesToCSs(int *ret, bat *sbatid, bat *pbatid, bat *obatid,bat *mbatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId);

rdf_export str 
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold, void *freqCSset, oid **subjCSMap, oid *maxCSoid, int *maxNumPwithDup, CSlabel** labels, CSrel **csRelBetweenMergeFreqSet);

rdf_export str
RDFreorganize(int *ret, CStableStat *cstablestat, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold, int *mode);

rdf_export void
freeCStableStat(CStableStat *cstablestat); 

rdf_export void
printPropStat(PropStat *propstat, int isPrintToFile); 

rdf_export void 
createTreeForCSset(CSset *freqCSset);

#endif /* _RDFSCHEMA_H_ */
