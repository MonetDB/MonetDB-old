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

#ifndef _RDFSCHEMA_H_
#define _RDFSCHEMA_H_

rdf_export str
RDFSchemaExplore(int *ret, str *tbname, str *clname);

rdf_export str
RDFextractCS(int *ret, bat *sbatid, bat *pbatid, int *freqThreshold); 

rdf_export str
RDFextractPfromPSO(int *ret, bat *pbatid, bat *sbatid); 

rdf_export str 
RDFextractCSwithTypes(int *ret, bat *sbatid, bat *pbatid, bat *obatid, bat *mapbatid, int *freqThreshold);

typedef struct {
	BAT*	hsKeyBat; 
	BAT* 	hsValueBat; 
	BAT* 	freqBat;    	/* Store the frequency of each Characteristic set */	
	BAT* 	coverageBat;	/* Store the exact number of triples coverred by each CS */
	BAT* 	pOffsetBat; 	/* BAT storing the offset for set of properties, refer to fullPBat */
	BAT* 	fullPBat;    	/* Stores all set of properties */	
} CSBats; 	// BATs for storing all information about CSs

/* Statistic about the properties */
typedef struct PropStat {
	BAT*	pBat; 		/* Store the list of properties */
	int* 	freqs; 	/* Store number of CSs containing that property */
	float*	tfidfs; 
	int	numAllocation; 
	int	numAdded; 
} PropStat; 			

#define INIT_PROP_NUM	10
#define	USINGTFIDF	1

#define STOREFULLCS     1       /* Store full instance of a CS including the a subject and list of predicates, objects. 
                                  Only use this for finding the name of the table corresponding to that CS */

typedef struct CS
{
	oid 	csId;		//Id of the CS
	oid*	lstProp;	//List of properties' Ids
	int	numProp;
	int	numAllocation;
	char 	isSubset; 
	#if STOREFULLCS
	oid     subject;        //A subject
	oid*    lstObj;         //List of sample objects
	#endif
} CS;


typedef struct maxCS
{
	oid 	csId;		//Id of the CS
	oid*	lstProp;	//List of properties' Ids
	int	numProp;
	int	numAllocation;
	int	support; 	//Sum of all subCS's frequency
	#if STOREFULLCS
	oid     subject;        //A subject
	oid*    lstObj;         //List of sample objects
	#endif
} maxCS; 

typedef struct SubCS {
	//oid 	csId; 
	oid	subCSId;
	oid 	sign;		// Signature generated from subTypes for quick comparison
	char*	subTypes;
	int	numSubTypes; 
} SubCS; 

typedef struct mergeCS {	// CS formed by merging CS id1 and CS id2	
	oid* 	lstParent; 	
	int	numParent; 
	oid*	lstProp; 
	int	numProp; 
	int 	support;
	int	coverage;
	char 	isRemove;

} mergeCS; 


#define INIT_NUM_SUBCS 4

typedef struct SubCSSet{
	oid	csId; 
	SubCS	*subCSs; 
	int	*freq; 
	int	numSubCS; 
	int 	numAllocation; 
} SubCSSet;

#define INIT_NUM_CS 100
#define SIM_THRESHOLD 0.6
#define SIM_TFIDF_THRESHOLD 0.55

typedef struct CSset{
	CS* items;
	int numCSadded;
	int numAllocation;
} CSset; 

typedef struct mergeCSset{
	mergeCS* items; 
	int nummergeCSadded; 
	int numAllocation; 
} mergeCSset;

#define INIT_NUM_CSREL 4
typedef struct CSrel{	
	oid  origCSoid;	
	oid* lstRefCSoid; 		
	oid* lstPropId; 	// Predicate for a relationship
	int* lstCnt; 		// Count per reference
	int* lstBlankCnt;	// Count # links to blank node
	int  numRef; 
	int  numAllocation; 
} CSrel;

#endif /* _RDFSCHEMA_H_ */
