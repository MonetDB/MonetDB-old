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
 * Copyright August 2008-2015 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * @f rdf
 * @a L.Sidirourgos, Minh-Duc Pham
 *
 * @* The RDF module For MonetDB5 (aka. MonetDB/RDF)
 *
 */
#ifndef _RDF_H_
#define _RDF_H_

#include <gdk.h>
#include "tokenizer.h"
#include <time.h>

#ifdef WIN32
#ifndef LIBRDF
#define rdf_export extern __declspec(dllimport)
#else
#define rdf_export extern __declspec(dllexport)
#endif
#else
#define rdf_export extern
#endif

/* internal debug messages */
#define _RDF_DEBUG

rdf_export str
RDFParser(BAT **graph, str *location, str *graphname, str *schemam, bat *ontbatid);

rdf_export str 
RDFleftfetchjoin_sorted(bat *result, const bat* lid, const bat *rid);

rdf_export str
RDFmultiway_merge_outerjoins(int np, BAT **sbats, BAT **obats, BAT **r_sbat, BAT **r_obats); 

rdf_export str 
TKNZRrdf2str (bat *res, const bat *bid, const bat *map);

rdf_export str
RDFpartialjoin (bat *res, bat *lmap, bat *rmap, bat *input); 

rdf_export str
RDFtriplesubsort(BAT **sbat, BAT **pbat, BAT **obat); 

rdf_export str
RDFbisubsort(BAT **lbat, BAT **rbat); 

rdf_export str
RDFexception_join(bat *ret1, bat *ret2, bat *sdense, bat *o1, bat *s2, bat *o2, bat *scand); 

rdf_export str
RDFmerge_join(bat *ret1, bat *ret2, bat *s1id, bat *o1id, bat *scandid); 

#define RDF_MIN_LITERAL (((oid) 1) << ((sizeof(oid)==8)?59:27))

#define IS_DUPLICATE_FREE 0		/* 0: Duplications have not been removed, otherwise 1 */
#define IS_COMPACT_TRIPLESTORE 1	/* 1: Only keep SPO for triple store */
#define TRIPLE_STORE 1
#define MLA_STORE    2
#define NOT_IGNORE_ERROR_TRIPLE 0
#define USE_MULTIPLICITY 1		/* Properties having >= 2 values are being considered as having 
					the same type, i.e., MULTIVALUES */

#define STORE TRIPLE_STORE /* this should become a compile time option */

#define EVERYTHING_AS_OID 1	/*We do not store type-specific column but oid only*/
#define STORE_ALL_EXCEPTION_IN_PSO 1	/* All the exceptions such as non-default type values are stored in 
					PSO table.*/

#define batsz 10000000
#define smallbatsz 100000
#define smallHashBatsz 10000

#if STORE == TRIPLE_STORE
 typedef enum {
	S_sort, P_sort, O_sort, /* sorted */
	P_PO, O_PO, /* spo */
	P_OP, O_OP, /* sop */
	S_SO, O_SO, /* pso */
	S_OS, O_OS, /* pos */
	S_SP, P_SP, /* osp */
	S_PS, P_PS, /* ops */
	MAP_LEX
 } graphBATType;
#elif STORE == MLA_STORE
 typedef enum {
	S_sort, P_sort, O_sort,
	MAP_LEX
 } graphBATType;
#endif /* STORE */

#define N_GRAPH_BAT (MAP_LEX+1)

#define INFO_WHERE_NAME_FROM 1
#define INFO_NAME_FREQUENCY 1	//Store the frequency of the name or its ontology TF-IDF score
				//This is used for analyzing the result
#define TOP_GENERAL_NAME 2	//Level of hierrachy in which a name is considered to be a general name
				//For example, PERSON, THING is at level 1	
#define	USE_ALTERNATIVE_NAME 1	//Use different but may be better name for a general name
#define USE_NAME_INSTEADOF_CANDIDATE_IN_S1 1	//Use name instead of candidate for merging CS's in S1

// Final data structure that stores the labels for tables and attributes
typedef struct CSlabel {
	oid		name;		// table name
	oid		*candidates;	// list of table name candidates, candidates[0] == name
	int		candidatesCount;// number of entries in the candidates list
	int		candidatesNew;		// number of candidates that are created during merging (e.g. ancestor name)
	int		candidatesOntology;	// number of ontology candidates (first category)
	int		candidatesType;		// number of type candidates (second category)
	int		candidatesFK;		// number of fk candidates (third category)
	oid		*hierarchy;     // hierarchy "bottom to top"
	int		hierarchyCount; // number of entries in the hierarchy list
	int		numProp;	// number of properties, copied from freqCSset->items[x].numProp
	oid		*lstProp;	// attribute names (same order as in freqCSset->items[x].lstProp)
	#if 	INFO_WHERE_NAME_FROM	
	char 		isOntology; 	// First name is decided by ontology
	char		isType; 	// First name is decided based on Type
	char		isFK; 	
	#endif
	#if	INFO_NAME_FREQUENCY
	int		nameFreq;		//Name frequency
	float		ontologySimScore; 	//ontology similarity score
	#endif 	
	
} CSlabel;

#endif /* _RDF_H_ */
