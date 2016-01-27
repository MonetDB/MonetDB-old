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
#include "rdftypes.h"

rdf_export str
RDFSchemaExplore(int *ret, str *tbname, str *clname);

rdf_export str
RDFextractCS(int *ret, bat *sbatid, bat *pbatid, int *freqThreshold); 

rdf_export str
RDFextractPfromPSO(int *ret, bat *pbatid, bat *sbatid); 

struct CSmergeRel; // forward declaration


typedef enum{
	EXPLOREONLY, 
	REORGANIZEONLY,
	BUILDTABLE
} ExpMode; 

typedef enum{
	MAINTBL, 
	TYPETBL,
	PSOTBL,
	MVTBL,
	NOTBL		//No data for this 
} TableType; 		

typedef enum {
	NORMALCS, 
	FREQCS, 
	MAXCS, 
	MERGECS,
	DIMENSIONCS
} CStype; 


#define EXTRAINFO_FROM_RDFTYPE 1 	//Using rdf:type value which is an ontology class as an extra info

typedef struct {
	BAT*	hsKeyBat; 
	BAT* 	hsValueBat; 
	BAT* 	freqBat;    	/* Store the frequency of each Characteristic set */	
	BAT* 	coverageBat;	/* Store the exact number of triples coverred by each CS */
	BAT* 	pOffsetBat; 	/* BAT storing the offset for set of properties, refer to fullPBat */
	BAT* 	fullPBat;    	/* Stores all set of properties */	
	//#if EXTRAINFO_FROM_RDFTYPE
	BAT* 	typeOffsetBat;	
	BAT*	fullTypeBat;	/* Stores the type values of each CS */
	//#endif
} CSBats; 	// BATs for storing all information about CSs


typedef struct Postinglist{
	int*	lstIdx; 	/* List of CS containing the property */
	int*	lstInvertIdx; 	/* List of property's index in each CS */
	oid*	lstOnt;		/* List of ontology */
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

typedef struct PsoPropStat {
	BAT*		pBat; 		/* Store the list of properties */
	BAT*		offsetBat; 	/* Store the first pos where the prop starts in PSO table */
} PsoPropStat; 

#define INIT_PROP_NUM	10
#define INIT_CS_PER_PROP 10
#define	USINGTFIDF	1
#define COMBINE_S2_S4	0

#define STOREFULLCS     1       /* Store full instance of a CS including the a subject and list of predicates, objects. 
                                  Only use this for finding the name of the table corresponding to that CS */

#define NBITS_FOR_CSID	20	/* Use bits from 62th bit --> (62 - NBITS_FOR_CSID) for encoding the CSId in each SubjectId */

#define CSTYPE_TABLE	1

#define FULL_PROP_STAT 1	// Only use for showing the statistic on all properties / all CSs. (Default should be 0)

#define STAT_ANALYZE 1	// Only use for collecting the statistic on the number of multi/null/single-valued prop

#define COLORINGPROP 1	// Only use for coloring property in schema representation. 

#define NEEDSUBCS 0	// We actually do not need to use SubCS as the idea of default subCS is not used. But it is still good
			// for collecting the statistical information (For reporting/writing)
#define GETSUBCS_FORALL 1 //Get subCS info for all CS, not only frequent one

#define USE_ONTLABEL_FOR_NAME	1	// If the ontology label of a class is available, use it for the name
#define USE_LABEL_FINDING_MAXCS	0 	// Use the labels received from labeling process for finding maxCS 
#define TOPK 1			//Check top 3 candidate
#define MAX_SUB_SUPER_NUMPROP_DIF 3
#define USE_MULTIWAY_MERGING	0

#define MINIMIZE_CONSISTSOF	1	/*Only store the minimize list of consistsof CS's Id. 
					Specifically, the consistsOf list only contains 
					the freqIdx of merged CS from previous rule. */

#define OUTPUT_FREQID_PER_LABEL 1 	/* This is for evaluating the results of merging using S1. TODO: Set it to 0 for default*/
//#define	MERGING_CONSIDER_NAMEORIGINALITY 0	/*Merging in rule S1, considering where the name comes from (e.g., from Ontology, from rdf:type, or from FK) */	 

//#define IS_MULVALUE_THRESHOLD  1.1	//The ratio betweeen (the number of triple coverred by Prop P) / (number of Non-NULL object values for P)
					//   If this ratio is ~1, only use single value column for that prop
					// Replaced by ( 1 + //INFREQ_TYPE_THRESHOLD) as multi-prop can be considered as the type of the props
//#define INFREQ_TYPE_THRESHOLD  0.1	/* Threshold that a type is consider as an infrequent type */



/* ---- For detecting dimension table */
#define	NUM_ITERATION_FOR_IR 	3	/* Number of iteration for indirect referrences to a CS (table) */
#define ONLY_SMALLTBL_DIMENSIONTBL 1 	/* Only small tables are considered to be dimension table 
					Small table is the one that have support < MINIMUM_TABLE_SIZE */
//#define IR_DIMENSION_THRESHOLD_PERCENTAGE     0.10	//DO NOT USE ANYMORE
//#define IR_DIMENSION_THRESHOLD_PERCENTAGE	0.02	//  Score of indirect references that the CS can be considered as a dimension CS 
							//   IR_DIMENSION_THRESHOLD_PERCENTAGE * totalFrequency 
							//   Number of IR references should be several times larger than the CS frequency 
#define	IR_DIMENSION_FACTOR	1000	//A table is a dimension table if the # of references to it is an order of magnitude (IR_DIMENSION_FACTOR) compared to # of its tuples   
					//
#define MAX_ITERATION_NO	6	//Max number of iteration run
							
//#define IR_DIMENSION_THRESHOLD_PERCENTAGE	0.2	//Value 0.2 is for example data only

#define NOT_MERGE_DIMENSIONCS	1		/* Default: 1, 0: Is for example data */
#define NOT_MERGE_DIMENSIONCS_IN_S1 0		/* Whether we should merge dimension CSs in S1 */
#define	ONLY_MERGE_ONTOLOGYBASEDNAME_CS_S1 0	/* Only merge CS's whose name comes from an ontology class*/
						//If only merge CS from an ontology class, some classes
						//from BSBM (i.e., REVIEW), SP2B (i.e., BOOK),...
						//URI should be ok.
#define	ONLY_MERGE_URINAME_CS_S1 1		/* Only merge CS's whose name is an URI */

#define MERGE_SAME_PROP_CS 1
#define SIM_SAME_PROP_THRESHOLD 0.9999		/* It should exactly be 1.0, however, the float multiplication may loss the precision */

#define FILTER_INFREQ_FK_FOR_IR	1		/* We filter out all the dirty references from a CS */
//#define FILTER_THRESHOLD_FK_FOR_IR	0.1	/* The FK that their frequency < FILTER_THRESHOLD_FK_FOR_IR * FreqCS's frequency */ 	
//						//Replaced by INFREQ_TYPE_THRESHOLD as a reference can be considered as a type of the object value

/*------------------------------------*/

#define STORE_PERFORMANCE_METRIC_INFO	1

#define NO_OUTPUTFILE	0		/*Do not write the output to any file */

extern int totalNumberOfTriples; 
extern int acceptableTableSize;

#define	COUNT_NUMTYPES_PERPROP 1

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

	#if EXTRAINFO_FROM_RDFTYPE
	oid*	typevalues;
	int	numTypeValues; 
	#endif

	//For mergeCS
	int* 	lstConsistsOf; 	//The list of indexes of freqCS	
	int	numConsistsOf; 

	#if STORE_PERFORMANCE_METRIC_INFO
	int	numInRef;	//Number of references to this table
	int	numFill;	//Number of triples covered by this CS. 
				//Note that: We count a multi-value prop with serveral obj values as one
				//Max numFill = numProp x support
	#endif

} CS;

typedef struct TFIDFInfo {
	int 	freqId;
	float*	lsttfidfs;	//TFIDF score of each prop in a CS
	float	totalTFIDF; 	// sqrt of (Sum = Total tfidfV*tfidfV of all props in that CS)
} TFIDFInfo; 


typedef struct SubCS {
	//oid 	csId; 
	oid	subCSId;
	oid 	sign;		// Signature generated from subTypes for quick comparison
	char*	subTypes;
	int	numSubTypes; 
	char	isdefault; 
} SubCS; 

#define INIT_NUM_SUBCS 4

typedef struct SubCSSet{
	oid	csId; 
	SubCS	*subCSs; 
	int	*freq; 
	int	numSubCS; 
	int 	numAllocation; 
} SubCSSet;

#define INIT_NUM_CS 1000 
#define SIM_THRESHOLD 0.6
//#define SIM_TFIDF_THRESHOLD 0.75	//Replaced by simTfidfThreshold
//#define IMPORTANCE_THRESHOLD 0.001 //This is used when merging CS's by common ancestor
					// Replace by generalityThreshold = 1/(upperboundNumTables)
#define COMMON_ANCESTOR_LOWEST_SPECIFIC_LEVEL 2 

//#define MIN_PERCETAGE_S5 5	// Merge all CS refered by more than 1/MIN_PERCETAGE_S6 percent of a CS via one property
				// Replaced by using INFREQ_TYPE_THRESHOLD
				//
#define MIN_FROMTABLE_SIZE_S5 100  // The minimum size of the "from" table in S6. Meaning that 
				    // the CS's to-be-merged in this rule must cover > MIN_FROMTABLE_SIZE_S6 / MIN_PERCETAGE_S6 triples
#define MIN_TO_PERCETAGE_S5 10	// Threshold for the number of instances in the target CS refered by the property
				// Number of references > (Frequency of referredCS / MIN_TO_PERCETAGE_S5)
#define MIN_TFIDF_PROP_S5 3	// The prop for FK in S5 must not be a common prop, it should be a discriminating one
				// This is for preventing the case of webpageID link in dbpedia 
#define MIN_TFIDF_PROP_S4 3.5	//  When we merge two CS's based on the tf-idf/consine similarity score, we want 
				// to make sure that we do not merge two CS's that may have same set of really common properties
				// such as type, description. They should have at least one discriminating prop in common. 
#define MIN_TFIDF_PROP_FINALTABLE 2.5 //Discriminating prop is prop that appears in less than 10% of the table	

#define UPDATE_NAME_BASEDON_POPULARTABLE 1//Update table name from merging multiple freqCS by using the most popular one

//#define MIN_FROMTABLE_SIZE_S5 1		/* For example data */
//#define MINIMUM_TABLE_SIZE 1000   //The minimum number of triples coverred by a table (i.e., a final CS)
//				    //Use the variable minTableSize read from param.ini
//#define MINIMUM_TABLE_SIZE 1   // For example dataset only 
#define HIGH_REFER_THRESHOLD 5

#define       INFREQ_PROP_THRESHOLD   0.05

#define REMOVE_INFREQ_PROP	1
#define REMOVE_LOTSOFNULL_SUBJECT	1
#define	LOTSOFNULL_SUBJECT_THRESHOLD	0.1

#define COUNT_PERCENTAGE_ONTO_PROP_USED	1	//Calculate the percentage of properties of ontology class
						//used in final schema
#define DETECT_INCORRECT_TYPE_SUBJECT	0	//Detect subjects that are assigned wrong type. (Default value 0)
#define USING_FINALTABLE		0	//Using the final table for collecting label stat or using set of 
						//final merged CS. The set of merged CS will be larged as it may
						//contain small table
#define STRANGE_PROP_FREQUENCY	10		//If the prop appears in less than 3 instances, it may be the black sheep

//#define	MIN_FK_FREQUENCY 	0.1	// The frequency of a FK should be > MIN_FK_FREQUENCY * The frequency of a mergedCS (or the number of tuples in one table)	
						// Replaced by INFREQ_TYPE_THRESHOLD
//#define MIN_FK_PROPCOVERAGE	0.9	// The FK needs to happen in MIN_FK_PROPCOVERAGE of all instances of the particular property
						// Replaced by (1 - INFREQ_TYPE_THRESHOLD)

#define EXPORT_LABEL		1	/* Export labels: TODO:   */


#define DETECT_PKCOL		1	/* Detect whether a col can be a primary key col while reorganizing triples table*/
#define ONLY_URI_PK		1	/* Only URI can be considered for PK */

#define COUNT_DISTINCT_REFERRED_S	1 	/* Count the number of distinct subject referred by the object values of certain 
						This is to detect whether the FK is ONE-to-MANY or MANY-to-MANY ....
						*/

#define REMOVE_SMALL_TABLE	1	/* Remove SMALL but NOT dimension table*/

#define APPENDSUBJECTCOLUMN	1	// The subject column actually doesn't need to be included into the relational table
					// However, for creating the foreign key relationship, we add this column and 
					// markt it as a primary key

#define BUILDTOKENZIER_TO_MAPID	1	/* Build the BAT storing the mapping between the
					original tokenizer oid of a URI and converted oid */

#define DUMP_CSSET		1	/* Dump all recognized CSs to BATs*/


typedef struct CSset{
	CS* items;
	int numOrigFreqCS; 
	int numCSadded;
	int numAllocation;
	#if STORE_PERFORMANCE_METRIC_INFO
	int totalInRef;
	#endif
} CSset; 

/*
typedef struct mergeCSset{
	mergeCS* items; 
	int nummergeCSadded; 
	int numAllocation; 
} mergeCSset;
*/

#define INIT_NUM_CSREL 10
typedef struct CSrel{	
	oid	origFreqIdx;
	oid	*lstRefFreqIdx; 
	oid* 	lstPropId; 	// Predicate for a relationship
	int* 	lstCnt; 		// Count per reference
	int* 	lstBlankCnt;	// Count # links to blank node
	int  	numRef; 
	int  	numAllocation; 
} CSrel;

#define NUMTHEAD_CSREL	4
#define USEMULTITHREAD 0
typedef struct csRelThreadArg {
	int	tid; 
	int	nthreads; 
	int	first; 
	int 	last; 
	BAT   	*sbat;
	BAT 	*pbat; 
	BAT 	*obat;
	oid 	*subjCSMap; 
	#if NEEDSUBCS	
	oid 	*subjSubCSMap;
	SubCSSet *csSubCSSet;
	#endif
	CSrel 	*csrelSet;
	BUN 	maxSoid;	
	int 	maxNumPwithDup;
	int 	*csIdFreqIdxMap;
} csRelThreadArg; 

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

typedef struct CSMVtableEx {		/* For non-default-type columns*/
	BAT*		subjBat; 	/* Subject bat refers to the subject ID in the main table */
	BAT*		keyBat; 	/* Key bat refers to corresponding column in the main table*/
	BAT**		mvBats;		/* The first is the default col, other is ex-type-specific cols*/
	ObjectType*	colTypes; 
	int		numCol; 
} CSMVtableEx; 

typedef struct CStable {
	BAT**		colBats; 
	ObjectType*	colTypes; 
	//BAT**		mvBats; 	/* One bat for one Muti-values property */
	//BAT**		mvExBats; 	/* One more bat for infrequent datatype in multi-valued prop*/	
	CSMVtableEx	*lstMVTables; 
	int		numCol; 
	oid* 		lstProp;
	oid		tblname;	/* Label of the table */
} CStable; 


typedef struct CStableEx {		/* For non-default-type columns*/
	BAT**		colBats; 
	ObjectType*	colTypes; 
	int		numCol; 
	int*		mainTblColIdx;	
	oid		tblname; 	/* Label of the table */
} CStableEx; 

#define TRIPLEBASED_TABLE 1

typedef struct CStableStat {
	bat**   	lstbatid;
	int		numTables;
	int*		numPropPerTable; 
	int* 		lstfreqId; 	//Idx of the corresponding freqCS for a table
	oid**		lastInsertedS;	//Last S for each column
	//sql_schema*	schema; 	
	CStable*	lstcstable; 
	#if CSTYPE_TABLE
	CStableEx*      lstcstableEx;
	oid**		lastInsertedSEx; 
	#endif
	BAT*		pbat; 
	BAT*		sbat; 
	BAT*		obat; 
#if TRIPLEBASED_TABLE
	BAT*		resbat; /* Re-organized sbat */
	BAT*		repbat; /* Re-organized pbat */
	BAT*		reobat; /* Re-organized obat */
#endif	
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
	int*	colIdxes; 	/* colIdxes for each type */
	int	defColIdx; 	/* Column index in the main table */
	char*	TableTypes;
	ObjectType	defaultType; 
	char	isMVProp; 	/* = 1 if this prop is a multi-valued prop*/
	char	isPKProp; 	/* = 1 if all the values in this columns is unique */
	char	numMvTypes; 	/* Number of extype BAT for this MV col */
	char	isFKProp; 
	int	refTblId;	/* refTblId != -1 only when isFKProp = 1 */
	int	refTblSupport; 	/* Support of the table referred by this prop */
	int	numReferring; 	/* Number of references from this prop */
	int	numDisRefValues;	/* Number of distinct referred values */
	char 	isDirtyFKProp; 	/* = 1 if not all instances of this prop points to  refTblId*/
} PropTypes; 

typedef struct CSPropTypes {
	int		freqCSId; 
	int		numProp; 
	int		numInfreqProp; 
	int 		numNonDefTypes; 
	PropTypes*	lstPropTypes; 
} CSPropTypes; 

typedef struct Pscore{		//Performance score
	float avgPrec; 		//average precision
	float overallPrec; 	//overall precision
	float Qscore; 		//metric score Q
	float Cscore; 		//metric score C
	int   nTable;		//number of tables
	float avgPrecFinal; 	//Avg precision of expected final tables (after removing small size table)
	float overallPrecFinal; //of expected final tables (after removing small size table)
	float QscoreFinal; 	//of expected final tables (after removing small size table)
	int   nFinalTable; 	//Expected number of final table after removing e.g., small size table	
} Pscore; 

#define NUM_SAMPLETABLE 20
#define	NUM_SAMPLE_INSTANCE 10
#define NUM_SAMPLE_CANDIDATE 3
#define SAMPLE_FILTER_THRESHOLD 10  // SAMPLE_FILTER_THRESHOLD/ 100	
#define GETSAMPLE_BEFOREMERGING 1  // Get the sample data before merging CS's
#define NUM_PROP_SAMPLE 8 // number of properties to be shown in sample data (plus support column, not included)

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


typedef struct CSSampleExtend{
	int	freqIdx;
	int 	tblIdx; 
	oid	*candidates; // shuffled
	oid	candidateCount; 
	oid	*candidatesOrdered; // ordered
	int	candidatesNew;
	int	candidatesOntology;
	int	candidatesType;
	int	candidatesFK;
	int	numProp; 
	oid	*lstProp;
	int	*lstPropSupport; 
	char	*lstIsInfrequentProp;
	char	*lstIsMVCol; 
	oid	*lstSubjOid;
	BAT** 	colBats; 
	int	numInstances; 
	oid	name;
} CSSampleExtend;

rdf_export str
RDFdistTriplesToCSs(int *ret, bat *sbatid, bat *pbatid, bat *obatid,bat *mbatid, bat *lmapbatid, bat *rmapbatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId, char *isLotsNullSubj, oid *subjCSMap, int* csTblIdxMapping);

rdf_export str
RDFdistTriplesToCSs_alloid(int *ret, bat *sbatid, bat *pbatid, bat *obatid,bat *mbatid, bat *lmapbatid, bat *rmapbatid, PropStat* propStat, CStableStat *cstablestat, CSPropTypes *csPropTypes, oid* lastSubjId, char *isLotsNullSubj, oid *subjCSMap, int* csTblIdxMapping);

rdf_export str 
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, bat *ontbatid, int *freqThreshold, void *freqCSset, oid **subjCSMap, oid *maxCSoid, int *maxNumPwithDup, CSlabel** labels, CSrel **csRelBetweenMergeFreqSet);

rdf_export str
RDFreorganize(int *ret, CStableStat *cstablestat, CSPropTypes **csPropTypes, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, bat *ontbatid, int *freqThreshold, int *mode);

rdf_export void
getSqlName(str *name, oid nameId,BATiter mapi, BAT *mbat);

rdf_export ObjectType
getObjType(oid objOid);		/* Return the type of the object value from obj oid*/

rdf_export void
freeCStableStat(CStableStat *cstablestat); 

rdf_export void
freeCSPropTypes(CSPropTypes* csPropTypes, int numCS);

rdf_export void
printPropStat(PropStat *propstat, int isPrintToFile); 

rdf_export void 
createTreeForCSset(CSset *freqCSset);

rdf_export char
isCSTable(CS item, oid name); 

rdf_export str
printTKNZStringFromOid(oid id);

rdf_export BAT*
createEncodedSubjBat(int tblIdx, int num);

rdf_export 
PropStat* initPropStat(void);

rdf_export
void freePropStat(PropStat *propStat);

rdf_export
void addaProp(PropStat* propStat, oid prop, int csIdx, int invertIdx);

rdf_export
void getTblIdxFromS(oid Soid, int *tbidx, oid *baseSoid);

rdf_export
int getTblId_from_S_simple(oid Soid); 


#endif /* _RDFSCHEMA_H_ */
