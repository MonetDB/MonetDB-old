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

// Counts the occurances of type attribute values
typedef struct TypeAttributesFreq {
	char		value[10000];	// TODO find better solution
	int		freq;
	int		percent;
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
	char		uri[100][100];	// tokenized namespace/prefix
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

// Final data structure that stores the labels for tables and attributes
typedef struct Labels {
	str		name;		// table name
	int		numProp;	// number of properties, copied from freqCSset->items[x].numProp
	char		**lstProp;	// attribute names (same order as in freqCSset->items[x].lstProp)
} Labels;

// Statistics for an ontology class
typedef struct ClassStat {
	char		*ontoClass;	// URI of the ontology class
	float		tfidfs;		// summarized tfidf score of all properties that accur in the ontology class
} ClassStat;

// Statistics for a type attribute value
typedef struct TypeStat {
	char		*value;		// type value
	int		freq;		// number of CS's the value occurs in
} TypeStat;

#define FK_FREQ_THRESHOLD 10		// X % of the targeted subjects have to be in this table
#define TYPE_FREQ_THRESHOLD 10		// X % of the type values have to be this value
#define ONTOLOGY_FREQ_THRESHOLD 0.5	// similarity threshold for tfidf simularity for ontology classes

#define USE_SHORT_NAMES 1		// use getPropNameShort()
#define USE_TYPE_NAMES 1		// use type attribute values for labeling
#define USE_FK_NAMES 1			// use incident fk names for labeling
#define USE_ONTOLOGY_NAMES 1		// use ontology classes for labeling
#define USE_TABLE_NAME 1		// calculate and store the final labels
#define SHOW_CANDIDATES 0		// inserts a row in UML diagrams to show all candidate names

Labels*
createLabels(CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet, BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, BAT* mapbat, int *csIdFreqIdxMap, int freqThreshold, str** ontattributes, int ontattributesCount, str** ontmetadata, int ontmetadataCount);

void
freeLabels(Labels* labels, CSset* freqCSset);

#endif /* _RDFLABELS_H_ */
