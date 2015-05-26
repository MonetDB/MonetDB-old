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

#ifndef _RDFLABELS_H_
#define _RDFLABELS_H_

#include "rdfschema.h"
#include "rdfontologyload.h"

// Counts the occurances of type attribute values
typedef struct TypeAttributesFreq {
	oid		value;
	int		freq;
	int		percent;
	float		rankscore; 	//= percent / global
} TypeAttributesFreq;

// Statistics for a foreign key relationship
typedef struct Relation {
	int		from;		// idx
	int		to;		// idx
	int		freq;
	int		percent;
} Relation;

// Tokenized URI prefix of an ontology
typedef struct ontology {
	char		uri[10][100];	// tokenized namespace/prefix
	int		length;		// number of tokens in this uri
} ontology;

// A foreign key relationship
typedef struct IncidentFK {
	oid		prop;		// name (oid) of the incident fk
	int		freq;		// frequency
	int		num;		//  how often the name (oid) occurs in the list of incident fks
} IncidentFK;

// All incident foreign keys of a CS
typedef struct IncidentFKs {
	int		num;
	IncidentFK	*fks;
} IncidentFKs;

// Statistics for an ontology class
typedef struct ClassStat {
	oid		ontoClass;	// URI of the ontology class
	float		tfidfs;		// summarized tfidf score of all properties that accur in the ontology class
	float		totaltfidfs; 	// The total tfidf score for all properties of this ontology classa
	int		numMatchedProp; // Number of matched prop in a ontology class
} ClassStat;

// Statistics for a type attribute value
typedef struct TypeStat {
	oid		value;		// type value
	int		freq;		// number of CS's the value occurs in
} TypeStat;

// Tree node to store the number of tuples per ontology class
typedef struct OntoUsageNode {
	struct OntoUsageNode	*parent;		// parent == NULL <=> artificial root
	struct OntoUsageNode	**lstChildren;
	oid			uri;
	int			numChildren;
	int			numOccurances; // TODO overflow 2,000,000?
	int			numOccurancesSum;
	float			percentage; // TODO rename, range [0..1]
} OntoUsageNode;

enum {
	S1, 
	S2, 
	S3,
	S4, 
	S5,
	S6
} RULE; 

#define FK_FREQ_THRESHOLD 25		// X % of the targeted subjects have to be in this table
#define FK_MIN_REFER_PERCENTAGE 25	// To be consider as the name of a CS, the FK have to point to at least FK_MIN_REFER_PERCENTAGE of all CS's instances 
#define TYPE_FREQ_THRESHOLD 80		// X % of the type values have to be this value
#define GOOD_TYPE_FREQ_THRESHOLD 95	// If a type appears really frequent in that CS, it should be choosen
//#define ONTOLOGY_FREQ_THRESHOLD 0.8	// similarity threshold for tfidf simularity for ontology classes

#define USE_SHORT_NAMES 1		// use getPropNameShort()
#define USE_TYPE_NAMES 1		// use type attribute values for labeling
#define USE_FK_NAMES 1			// use incident fk names for labeling
#define USE_ONTOLOGY_NAMES 1		// use ontology classes for labeling
#define USE_TABLE_NAME 1		// calculate and store the final labels
#define SHOW_CANDIDATES 0		// inserts a row in UML diagrams to show all candidate names
#define	ONLY_USE_ONTOLOGYBASED_TYPE 0
#define USE_BEST_TYPEVALUE_INSTEADOF_DUMMY 1  //Use the most frequent type value instead of a dummy for the label name	
#define MIN_POSSIBLE_TYPE_FREQ_THRESHOLD  20  //However, that type must still appears in more than a minimum threshold
#define TYPE_TFIDF_RANKING 1	//Rank value of type property by using (percent in a CS) / (percent in all subjects)

rdf_export void
getPropNameShort(char** name, char* propStr);

rdf_export void
getStringBetweenQuotes(str* out, str in);

rdf_export CSlabel*
createLabels(CSset* freqCSset, CSrel* csrelSet, int num, BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, int *csIdFreqIdxMap, oid** ontattributes, int ontattributesCount, oid** ontmetadata, 
		int ontmetadataCount, OntoUsageNode** ontoUsageTree, BAT *ontmetaBat, OntClass *ontclassSet);

rdf_export void
exportLabels(CSset* freqCSset, CSrel* csRelFinalFKs, int freqThreshold, BATiter mapi, BAT *mbat, CStableStat* cstablestat, CSPropTypes *csPropTypes, int numTables, int* mTblIdxFreqIdxMapping, int* csTblIdxMapping);

rdf_export str
updateLabel(int ruleNumber, CSset *freqCSset, CSlabel **labels, int newCS, int mergeCSFreqId, int freqCS1, int freqCS2, oid name, int isType, int isOnto, int isFK, oid **ontmetadata, int ontmetadataCount, int *lstFreqId, int numIds);

rdf_export void
freeLabels(CSlabel* labels, CSset* freqCSset);

rdf_export void
freeOntoUsageTree(OntoUsageNode* tree);

rdf_export void
printListOntology(void);

#if USE_TYPE_NAMES
extern char*	typeAttributes[];
extern int typeAttributesCount; 
#endif

#endif /* _RDFLABELS_H_ */
