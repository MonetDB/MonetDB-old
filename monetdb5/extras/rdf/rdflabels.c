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

#include "monetdb_config.h"
#include "rdf.h"
#include "rdflabels.h"
#include "rdfschema.h"
#include "tokenizer.h"
#include <math.h>
#include "rdfparams.h"

// list of known ontologies
int ontologyCount = 74;
ontology ontologies[] = {
{{"<http:", "www.facebook.com", "2008"}, 3},
{{"<http:", "facebook.com", "2008"}, 3},
{{"<http:", "developers.facebook.com", "schema"}, 3},
{{"<https:", "www.facebook.com", "2008"}, 3},
{{"<http:", "purl.org", "dc", "elements", "1.1"}, 5}, // dc DublinCore
{{"<http:", "purl.org", "dc", "terms"}, 4}, // DublinCore
{{"<http:", "purl.org", "goodrelations", "v1"}, 4}, // GoodRelations
{{"<http:", "purl.org", "rss", "1.0", "modules"}, 5},
{{"<http:", "purl.org", "stuff"}, 3},
{{"<http:", "www.purl.org", "stuff"}, 3},
{{"<http:", "ogp.me", "ns"}, 3},
{{"<https:", "ogp.me", "ns"}, 3},
{{"<http:", "ogp.mc", "ns"}, 3},
{{"<http:", "www.w3.org", "1999", "02", "22-rdf-syntax-ns"}, 5}, // rdf
{{"<http:", "www.w3.org", "2000", "01", "rdf-schema"}, 5}, // rdfs
{{"<http:", "www.w3.org", "2004", "02", "skos", "core"}, 6}, // skos (Simple Knowledge Organization System)
{{"<http:", "www.w3.org", "2002", "07", "owl"}, 5},
{{"<http:", "www.w3.org", "2006", "vcard", "ns"}, 5}, // vcard
{{"<http:", "www.w3.org", "2001", "vcard-rdf", "3.0"}, 5},
{{"<http:", "www.w3.org", "2003", "01", "geo", "wgs84_pos"}, 6}, // geo
{{"<http:", "www.w3.org", "1999", "xhtml", "vocab"}, 5}, // xhtml
{{"<http:", "search.yahoo.com", "searchmonkey"}, 3},
{{"<https:", "search.yahoo.com", "searchmonkey"}, 3},
{{"<http:", "search.yahoo.co.jp", "searchmonkey"}, 3},
{{"<http:", "g.yahoo.com", "searchmonkey"}, 3},
{{"<http:", "opengraphprotocol.org", "schema"}, 3},
{{"<https:", "opengraphprotocol.org", "schema"}, 3},
{{"<http:", "opengraph.org", "schema"}, 3},
{{"<https:", "opengraph.org", "schema"}, 3},
{{"<http:", "creativecommons.org", "ns"}, 3}, // cc
{{"<http:", "rdf.data-vocabulary.org"}, 2}, // by google
{{"<http:", "rdfs.org", "sioc", "ns"}, 4}, // sioc (pronounced "shock", Semantically-Interlinked Online Communities Project)
{{"<http:", "xmlns.com", "foaf", "0.1"}, 4}, // foaf (Friend of a Friend)
{{"<http:", "mixi-platform.com", "ns"}, 3}, // japanese social graph
{{"<http:", "commontag.org", "ns"}, 3},
{{"<http:", "semsl.org", "ontology"}, 3}, // semantic web for second life
{{"<http:", "schema.org"}, 2},
{{"<http:", "openelectiondata.org", "0.1"}, 3},
{{"<http:", "search.aol.com", "rdf"}, 3},
{{"<http:", "www.loc.gov", "loc.terms", "relators"}, 4}, // library of congress
{{"<http:", "dbpedia.org", "ontology"}, 3}, // dbo
{{"<http:", "dbpedia.org", "resource"}, 3}, // dbpedia
{{"<http:", "dbpedia.org", "property"}, 3}, // dbp
{{"<http:", "www.aktors.org", "ontology", "portal"}, 4}, // akt (research, publications, ...)
{{"<http:", "purl.org", "ontology", "bibo"}, 4}, // bibo (bibliography)
{{"<http:", "purl.org", "ontology", "mo"}, 4}, // mo (music)
{{"<http:", "www.geonames.org", "ontology"}, 3}, // geonames
{{"<http:", "purl.org", "vocab", "frbr", "core"}, 5}, // frbr (Functional Requirements for Bibliographic Records)
{{"<http:", "www.w3.org", "2001", "XMLSchema"}, 4}, // xsd
{{"<http:", "www.w3.org", "2006", "time"}, 4}, // time
{{"<http:", "purl.org", "NET", "c4dm", "event.owl"}, 5}, // event
{{"<http:", "www.openarchives.org", "ore", "terms"}, 4}, // ore (Open Archive)
{{"<http:", "purl.org", "vocab", "bio", "0.1"}, 5}, // bio (biographical data)
{{"<http:", "www.holygoat.co.uk", "owl", "redwood", "0.1", "tags"}, 6}, // tag
{{"<http:", "rdfs.org", "ns", "void"}, 4}, // void (Vocabulary of Interlinked Datasets)
{{"<http:", "www.w3.org", "2006", "http"}, 4}, // http
{{"<http:", "purl.uniprot.org", "core"}, 3}, // uniprot (protein annotation)
{{"<http:", "umbel.org", "umbel"}, 3}, // umbel (Upper Mapping and Binding Exchange Layer)
{{"<http:", "purl.org", "stuff", "rev"}, 4}, // rev (review)
{{"<http:", "purl.org", "linked-data", "cube"}, 4}, // qb (data cube)
{{"<http:", "www.w3.org", "ns", "org"}, 4}, // org (organizations)
{{"<http:", "purl.org", "vocab", "vann"}, 4}, // vann (vocabulary for annotating vocabulary descriptions)
{{"<http:", "data.ordnancesurvey.co.uk", "ontology", "admingeo"}, 4}, // admingeo (administrative geography and civil voting area)
{{"<http:", "www.w3.org", "2007", "05", "powder-s"}, 5}, // wdrs (Web Description Resources)
{{"<http:", "usefulinc.com", "ns", "doap"}, 4}, // doap (Description of a Project)
{{"<http:", "lod.taxonconcept.org", "ontology", "txn.owl"}, 4}, // txn (TaxonConcept, species)
{{"<http:", "xmlns.com", "wot", "0.1"}, 4}, // wot (Web Of Trust)
{{"<http:", "purl.org", "net", "compass"}, 4}, // compass
{{"<http:", "www.w3.org", "2004", "03", "trix", "rdfg-1"}, 6}, // rdfg (RDF graph)
{{"<http:", "purl.org", "NET", "c4dm", "timeline.owl"}, 5}, // tl (timeline)
{{"<http:", "purl.org", "dc", "dcam"}, 4}, // dcam (DublinCore metadata)
{{"<http:", "swrc.ontoware.org", "ontology"}, 3}, // swrc (university, research)
{{"<http:", "zeitkunst.org", "bibtex", "0.1", "bibtex.owl"}, 5}, // bib (bibTeX entries)
{{"<http:", "purl.org", "ontology", "po"}, 4} // po (tv and radio programmes)
};

void printListOntology(void){
	int i,j; 
	for (i = 0; i < ontologyCount; ++i) {
		printf("%s/",ontologies[i].uri[0]);
		for (j = 1; j < ontologies[i].length; j++){
			printf("/%s",ontologies[i].uri[j]);
		}
		printf("\n"); 
	}
}
/* Extracts the "human-readable" part of an URI (usually the last token). */
void getPropNameShort(char** name, char* propStr) {
	char		*token;
	char		*uri;
	char		*uriPtr;
	int		length = 0;		// number of tokens
	char		**tokenizedUri = NULL;	// list of tokens
	int		i, j;
	int		fit;

	// tokenize uri
	uri = (char *) GDKmalloc(sizeof(char) * (strlen(propStr) + 1));
	if (!uri) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	strcpy(uri, propStr); // uri will be modified during tokenization
	uriPtr = uri; // uri will be modified, uriPtr keeps original pointer
	token = strtok(uri, "/#");
	while (token != NULL) {
		tokenizedUri = GDKrealloc(tokenizedUri, sizeof(char*) * ++length);
		if (!tokenizedUri) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		tokenizedUri[length - 1] = token;
		token = strtok(NULL, "/#");
	}

	// match with ontologies
	for (j = 0; j < ontologyCount; ++j) {
		if (length > ontologies[j].length) {
			fit = 1;
			for (i = 0; fit && i < ontologies[j].length; ++i) {
				if (strcmp(ontologies[j].uri[i], tokenizedUri[i]) != 0) {
					fit = 0;
				}
			}
			if (fit) {
				// found matching ontology, create label
				int totalLength = 0;
				for (i = ontologies[j].length; i < length; ++i) {
					totalLength += (strlen(tokenizedUri[i]) + 1); // additional char for underscore
				}
				(*name) = (char *) GDKmalloc(sizeof(char) * (totalLength + 1));
				if (!(*name)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				strcpy(*name, "\0");

				for (i = ontologies[j].length; i < length; ++i) {
					strcat(*name, tokenizedUri[i]);
					strcat(*name, "_"); // if label consists of >=2 tokens, use underscores
				}
				// remove trailing underscore
				(*name)[strlen(*name) - 1] = '\0';

				if ((*name)[strlen(*name) - 1] == '>') (*name)[strlen(*name) - 1] = '\0'; // remove >

				GDKfree(tokenizedUri);
				GDKfree(uriPtr);
				return;
			}
		}
	}

	// no matching ontology found, return content of last token

	if (length <= 1) {
		// value
		(*name) = (char *) GDKmalloc(sizeof(char) * (strlen(propStr) + 1));
		if (!(*name)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(*name, propStr);
	} else {
		(*name) = (char *) GDKmalloc(sizeof(char) * (strlen(tokenizedUri[length - 1]) + 1));
		if (!(*name)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(*name, tokenizedUri[length - 1]);
	}

	if ((*name)[strlen(*name) - 1] == '>') (*name)[strlen(*name) - 1] = '\0'; // remove >

	GDKfree(tokenizedUri);
	GDKfree(uriPtr);
	return;
}

static
int** initTypeAttributesHistogramCount(int typeAttributesCount, int num) {
	int		i, j;
	int**		typeAttributesHistogramCount;

	typeAttributesHistogramCount = (int **) malloc(sizeof(int *) * num);
	if (!typeAttributesHistogramCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < num; ++i) {
		typeAttributesHistogramCount[i] = (int *) malloc(sizeof(int) * typeAttributesCount);
		if (!typeAttributesHistogramCount[i]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < typeAttributesCount; ++j) {
			typeAttributesHistogramCount[i][j] = 0;
		}
	}

	return typeAttributesHistogramCount;
}

static
TypeAttributesFreq*** initTypeAttributesHistogram(int typeAttributesCount, int num) {
	int			i, j;
	TypeAttributesFreq***	typeAttributesHistogram;

	typeAttributesHistogram = (TypeAttributesFreq ***) malloc(sizeof(TypeAttributesFreq **) * num);
	if (!typeAttributesHistogram) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < num; ++i) {
		typeAttributesHistogram[i] = (TypeAttributesFreq **) malloc (sizeof(TypeAttributesFreq *) * typeAttributesCount);
		if (!typeAttributesHistogram[i]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < typeAttributesCount; ++j) {
			typeAttributesHistogram[i][j] = NULL;
		}
	}

	return typeAttributesHistogram;
}

static
int** initRelationMetadataCount(CSset* freqCSset) {
	int		i, j;
	int**		relationMetadataCount;

	relationMetadataCount = (int **) malloc(sizeof(int *) * freqCSset->numCSadded);
	if (!relationMetadataCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		relationMetadataCount[i] = NULL;
		relationMetadataCount[i] = (int *) malloc(sizeof(int) * freqCSset->items[i].numProp);
		if (!relationMetadataCount[i]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < freqCSset->items[i].numProp; ++j) {
			relationMetadataCount[i][j] = 0;
		}
	}

	return relationMetadataCount;
}

/* Calculate frequency per foreign key relationship. */
static
Relation*** initRelationMetadata(int** relationMetadataCount, CSrel* csrelSet, int num, CSset* freqCSset) {
	int		i, j, k;
	Relation***	relationMetadata;

	relationMetadata = (Relation ***) malloc(sizeof(Relation **) * freqCSset->numCSadded);
	if (!relationMetadata) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < num; ++i) { // CS
		CS cs;
		int csId = i;
		cs = (CS) freqCSset->items[csId];
		relationMetadata[csId] = (Relation **) malloc (sizeof(Relation *) * cs.numProp);
		if (!relationMetadata[csId]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < cs.numProp; ++j) { // propNo in CS order
			int sum = 0;
			relationMetadataCount[csId][j] = 0;
			relationMetadata[csId][j] = NULL;
			for (k = 0; k < csrelSet[i].numRef; ++k) { // propNo in CSrel

				if (csrelSet[i].lstPropId[k] == cs.lstProp[j]) {
					int toId = csrelSet[i].lstRefFreqIdx[k];
					relationMetadataCount[csId][j] += 1;

					// alloc/realloc
					if (relationMetadataCount[csId][j] == 1) {
						// alloc
						relationMetadata[csId][j] = (Relation *) malloc (sizeof(Relation));
						if (!relationMetadata[csId][j]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
						relationMetadata[csId][j][0].to = toId;
						relationMetadata[csId][j][0].from = csId;
						relationMetadata[csId][j][0].freq = csrelSet[i].lstCnt[k];
						relationMetadata[csId][j][0].percent = -1;
					} else {
						// realloc
						relationMetadata[csId][j] = (Relation *) realloc(relationMetadata[csId][j], sizeof(Relation) * relationMetadataCount[csId][j]);
						if (!relationMetadata[csId][j]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						relationMetadata[csId][j][relationMetadataCount[csId][j] - 1].to = toId;
						relationMetadata[csId][j][relationMetadataCount[csId][j] - 1].from = csId;
						relationMetadata[csId][j][relationMetadataCount[csId][j] - 1].freq = csrelSet[i].lstCnt[k];
						relationMetadata[csId][j][relationMetadataCount[csId][j] - 1].percent = -1;
					}
				}
			}

			// get total count of values
			for (k = 0; k < relationMetadataCount[csId][j]; ++k) {
				sum += relationMetadata[csId][j][k].freq;
			}
			// assign percentage values for every value
			for (k = 0; k < relationMetadataCount[csId][j]; ++k) {
				relationMetadata[csId][j][k].percent = (int) (100.0 * relationMetadata[csId][j][k].freq / sum + 0.5);
			}
		}
	}

	return relationMetadata;
}

static
IncidentFKs* initLinks(int csCount) {
	int		i;
	IncidentFKs*	links;

	links = (IncidentFKs *) malloc(sizeof(IncidentFKs) * csCount);
	if (!links) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < csCount; ++i) {
		links[i].num = 0;
		links[i].fks = NULL;
	}
	return links;
}

/* from:   <URI>/ or <URI/> or <URI> or URI/   to:   URI */
static
str removeBrackets(char* s) {
	str retStr;

	if (s[0] == '<' && s[strlen(s) - 2] == '>' && s[strlen(s) - 1] == '/') {
		// case <URI>/
		retStr = (str) GDKmalloc(strlen(s) - 2);
		if (!retStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strncpy(retStr, s + 1, strlen(s) - 3);
		retStr[strlen(s) - 3] = '\0';
		return retStr;
	} else if (s[0] == '<' && s[strlen(s) - 2] == '/' && s[strlen(s) - 1] == '>') {
		// case <URI/>
		retStr = (str) GDKmalloc(strlen(s) - 2);
		if (!retStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strncpy(retStr, s + 1, strlen(s) - 3);
		retStr[strlen(s) - 3] = '\0';
		return retStr;
	} else if (s[0] == '<' && s[strlen(s) - 1] == '>') {
		// case <URI>
		retStr = (str) GDKmalloc(strlen(s) - 1);
		if (!retStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strncpy(retStr, s + 1, strlen(s) - 2);
		retStr[strlen(s) - 2] = '\0';
		return retStr;
	} else if (s[strlen(s) - 1] == '/') {
		// case URI/
		retStr = (str) GDKmalloc(strlen(s));
		if (!retStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strncpy(retStr, s + 1, strlen(s) - 1);
		retStr[strlen(s) - 1] = '\0';
		return retStr;
	} else {
		// copy
		retStr = (str) GDKmalloc(strlen(s) + 1);
		if (!retStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(retStr, s);
		return retStr;
	}
}

/* Modifies the parameter! */
/* Replaces colons and quotes with undescores. */
static
void escapeURI(char* s) {
	int		i;

	for (i = 0; i < (int) strlen(s); ++i) {
		if (s[i] == ':' || s[i] == '"') s[i] = '_';
	}
}

void
getStringBetweenQuotes(str* out, str in) {
	int open = -1, close = -1;
	int i;

	for (i = 0; i < (int) strlen(in); ++i) {
		if (in[i] == '"') {
			if (open == -1) {
				open = i;
			} else {
				close = i;
				break;
			}
		}
	}
	if (close != -1) {
		// found pair of quotes
		(*out) = (str) GDKmalloc(close - open);
		if (!(*out)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strncpy((*out), &(in[open + 1]), (close - open - 1));
		(*out)[close - open - 1] = '\0';
	} else {
		// copy whole string
		(*out) = (str) GDKmalloc(strlen(in) + 1);
		if (!(*out)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strncpy((*out), in, strlen(in));
		(*out)[strlen(in)] = '\0';
	}
}

#if TYPE_TFIDF_RANKING
/*
 * Init the BATs for storing all type oids and their frequency
 * */
static 
void initGlobalTypeBATs(BAT **glTypeValueBat, BAT **glTypeFreqBat){

	*glTypeValueBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, TRANSIENT);
	BATseqbase(*glTypeValueBat, 0);
	if (*glTypeValueBat == NULL) {
		fprintf(stderr, "ERROR: Couldn't create BAT!\n");
	}

	(void)BAThash(*glTypeValueBat,0);
	if (!((*glTypeValueBat)->T->hash)){
		fprintf(stderr, "ERROR: Couldn't create Hash for BAT!\n");	
	}

	*glTypeFreqBat = BATnew(TYPE_void, TYPE_int, smallbatsz, TRANSIENT);
	if (*glTypeFreqBat == NULL) {
		fprintf(stderr, "ERROR: Couldn't create BAT!\n");	
	}

}

static
void freeGlobalTypeBATs(BAT *glTypeValueBat, BAT *glTypeFreqBat){
	BBPunfix(glTypeValueBat->batCacheid);
	BBPunfix(glTypeFreqBat->batCacheid);
}

static 
void addGlobalType(oid typevalue, BAT *glTypeValueBat, BAT *glTypeFreqBat){
	oid tmp;
	BUN bun; 
	int freq; 

	tmp = typevalue; 
	bun = BUNfnd(glTypeValueBat,(ptr) &tmp);
	if (bun == BUN_NONE){	//New type value
		if (glTypeValueBat->T->hash && BATcount(glTypeValueBat) > 4 * glTypeValueBat->T->hash->mask) {
			HASHdestroy(glTypeValueBat);
			BAThash(glTypeValueBat, 2*BATcount(glTypeValueBat));
		}
		BUNappend(glTypeValueBat,&tmp, TRUE);	
		freq = 1; 
		BUNappend(glTypeFreqBat, &freq, TRUE);
	} else{
		int *curfreq = (int *)Tloc(glTypeFreqBat, bun);	
		(*curfreq)++;  
	}
}

static
int getTypeGlobalFrequency(oid typevalue, BAT *glTypeValueBat, BAT *glTypeFreqBat){
	
	oid tmp;
	BUN bun; 
	int ret = -1; 

	tmp = typevalue; 
	bun = BUNfnd(glTypeValueBat,(ptr) &tmp);
	if (bun == BUN_NONE){	//New type value
		fprintf(stderr, "ERROR: This typevalue must be there!\n");	
	} else{
		int *freq = (int *)Tloc(glTypeFreqBat, bun);	
		ret = *freq;  
		return ret; 
	}	
	return ret; 
}
#endif


#if USE_TYPE_NAMES
static
int compareTypeAttributesFreqs (const void * a, const void * b) {
  return ( (*(TypeAttributesFreq*)b).freq - (*(TypeAttributesFreq*)a).freq ); // sort descending
}
#endif

#if USE_TYPE_NAMES
/* Add type values to the histogram. Values that are not present in the hierarchy tree built from the ontologies are NOT added to the histogram. */
static
void insertValuesIntoTypeAttributesHistogram(oid* typeList, int typeListLength, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, int csFreqIdx, int type, BAT *ontmetaBat, BAT *glTypeValueBat, BAT *glTypeFreqBat) {
	int		i, j;
	int		fit;
	(void) ontmetaBat;

	for (i = 0; i < typeListLength; ++i) {
		#if ONLY_USE_ONTOLOGYBASED_TYPE
		BUN pos = BUNfnd(ontmetaBat, &typeList[i]);
		if (pos == BUN_NONE) continue; // no ontology information, ignore
		#endif
		// add to histogram
		fit = 0;
		for (j = 0; j < typeAttributesHistogramCount[csFreqIdx][type]; ++j) {
			if (typeAttributesHistogram[csFreqIdx][type][j].value == typeList[i]) {
				// bucket exists
				typeAttributesHistogram[csFreqIdx][type][j].freq += 1;
				fit = 1;
				break;
			}
		}
		if (!fit) {
			// bucket does not exist
			// realloc
			typeAttributesHistogramCount[csFreqIdx][type] += 1;
			typeAttributesHistogram[csFreqIdx][type] = (TypeAttributesFreq *) realloc(typeAttributesHistogram[csFreqIdx][type], sizeof(TypeAttributesFreq) * typeAttributesHistogramCount[csFreqIdx][type]);
			if (!typeAttributesHistogram[csFreqIdx][type]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

			// insert value
			typeAttributesHistogram[csFreqIdx][type][typeAttributesHistogramCount[csFreqIdx][type] - 1].value = typeList[i];
			typeAttributesHistogram[csFreqIdx][type][typeAttributesHistogramCount[csFreqIdx][type] - 1].freq = 1;
		}

		//Add to global types
		#if TYPE_TFIDF_RANKING
		addGlobalType(typeList[i], glTypeValueBat, glTypeFreqBat); 
		#else
		(void) glTypeValueBat; 
		(void) glTypeFreqBat; 
		#endif
	}
}



/* Loop through all subjects to collect frequency statistics for type attribute values. */
static
void createTypeAttributesHistogram(BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, CSset *freqCSset, int *csIdFreqIdxMap, int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, char** typeAttributes, BAT *ontmetaBat) {
	// looping, extracting
	BUN		p, q;
	oid 		*sbt, *obt, *pbt;
	char 		objType;
	oid 		objOid;
	int		csFreqIdx;
	oid		curS; // last subject
	int		curT; // last type (index in 'typeAttributes' array)
	oid		*typeValues; // list of type values per subject and type
	int		typeValuesSize;
	int		typeValuesMaxSize = 10;
	int		numS = 0; 

	// histogram
	int		i, j, k;

	oid 		*typeAttributesOids = malloc(sizeof(oid) * typeAttributesCount);

	BAT		*glTypeValueBat = NULL;	//Store the oid of each type value 
	BAT		*glTypeFreqBat = NULL; 	//Store the global frequency (#of subjects) of a type value 

	#if TYPE_TFIDF_RANKING	
	int		tmpgl_freq = 0; 
	initGlobalTypeBATs(&glTypeValueBat, &glTypeFreqBat); 
	#endif

	if (BATcount(sbat) == 0) {
		fprintf(stderr, "sbat must not be empty");
		/* otherwise, variable sbt is not initialized and thus
		 * cannot be dereferenced after the BATloop below */
	}

	// get oids for typeAttributes[]
	for (i = 0; i < typeAttributesCount; ++i) {
		TKNZRappend(&typeAttributesOids[i], &typeAttributes[i]);
	}

	curS = BUN_NONE;
	curT = -1;
	typeValues = GDKmalloc(sizeof(oid) * typeValuesMaxSize);
	if (!typeValues) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	typeValuesSize = 0;
	BATloop(sbat, p, q) {
		// Get data
		sbt = (oid *) BUNtloc(si, p);
		pbt = (oid *) BUNtloc(pi, p);

		csFreqIdx = csIdFreqIdxMap[subjCSMap[*sbt]];
		if (csFreqIdx == -1) {
			// subject does not belong to a freqCS
			continue;
		}

		// check if property (*pbt) is a type
		for (i = 0; i < typeAttributesCount; ++i) {
			if (*pbt == typeAttributesOids[i]) {
				// prop is a type!

				// get object
				obt = (oid *) BUNtloc(oi, p);
				objOid = *obt;
				objType = (char) ((*obt) >> (sizeof(BUN)*8 - 4))  &  7;

				if (objType == URI || objType == BLANKNODE) {
					objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
				} else {
					objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */
				}

				// if finished looping over one subject or type, the list of type values is analyzed and added to the histogram
				if (curS != *sbt || curT != i) {
					if (curS == BUN_NONE || typeValuesSize == 0) {
						// nothing to add to histogram
					} else {
						// analyze values and add to histogram
						csFreqIdx = csIdFreqIdxMap[subjCSMap[curS]]; // get csFreqIdx of last subject
						insertValuesIntoTypeAttributesHistogram(typeValues, typeValuesSize, typeAttributesHistogram, typeAttributesHistogramCount, csFreqIdx, curT, ontmetaBat, glTypeValueBat,glTypeFreqBat);
						typeValuesSize = 0; // reset
					}
					curS = *sbt;
					numS++; 
					curT = i;
				}
				// add value to list of type values
				if (typeValuesSize == typeValuesMaxSize) {
					// resize
					typeValuesMaxSize *= 2;
					typeValues = GDKrealloc(typeValues, sizeof(oid) * typeValuesMaxSize);
					if (!typeValues) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				typeValues[typeValuesSize++] = *obt;
				break;
			}
		}
	}

	// analyze and add last set of typeValues
	if (curS != BUN_NONE && typeValuesSize != 0) {
		csFreqIdx = csIdFreqIdxMap[subjCSMap[curS]]; // get csFreqIdx of last subject
		insertValuesIntoTypeAttributesHistogram(typeValues, typeValuesSize, typeAttributesHistogram, typeAttributesHistogramCount, csFreqIdx, curT, ontmetaBat, glTypeValueBat,glTypeFreqBat);
	}

	GDKfree(typeValues);

	// sort descending by frequency
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			qsort(typeAttributesHistogram[i][j], typeAttributesHistogramCount[i][j], sizeof(TypeAttributesFreq), compareTypeAttributesFreqs);
		}
	}

	(void) numS; 
	// assign percentage and tf-idf ranking score
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			// assign percentage values for every value
			for (k = 0; k < typeAttributesHistogramCount[i][j]; ++k) {
				typeAttributesHistogram[i][j][k].percent = (int) (100.0 * typeAttributesHistogram[i][j][k].freq / freqCSset->items[i].support + 0.5);
				#if TYPE_TFIDF_RANKING
				tmpgl_freq = getTypeGlobalFrequency(typeAttributesHistogram[i][j][k].value, glTypeValueBat, glTypeFreqBat); 
				typeAttributesHistogram[i][j][k].rankscore = ((float) typeAttributesHistogram[i][j][k].percent * numS) / (float) tmpgl_freq; 
				//printf("numS = %d, oid "BUNFMT", typeAttributesHistogram[i][j][k].freq = %d, tmpgl_freq = %d, percent = %d , rankscore = %f\n",
				//		numS,  typeAttributesHistogram[i][j][k].value, typeAttributesHistogram[i][j][k].freq, tmpgl_freq, typeAttributesHistogram[i][j][k].percent, typeAttributesHistogram[i][j][k].rankscore);
				#endif
			}
		}
	}

	free(typeAttributesOids);
	#if TYPE_TFIDF_RANKING
	freeGlobalTypeBATs(glTypeValueBat, glTypeFreqBat);
	#endif
}
#endif

#if USE_TYPE_NAMES
static
int compareTypeStats(const void * a, const void * b) {
	return (*(TypeStat*)a).freq - (*(TypeStat*)b).freq; // sort ascending
}
#endif

#if USE_TYPE_NAMES
/* Creates frequency statistics for type value attributes. ("In how many CS's is this value used?") */
static
TypeStat* getTypeStats(int* typeStatCount, int csCount, int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount) {
	TypeStat	*typeStat;
	int		i, j, k, l;

	typeStat = NULL;
	*typeStatCount = 0;
	for (i = 0; i < csCount; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			for (k = 0; k < typeAttributesHistogramCount[i][j]; ++k) {
				int found = 0;
				if (typeAttributesHistogram[i][j][k].percent < TYPE_FREQ_THRESHOLD) break;
				// search for this value
				for (l = 0; l < *typeStatCount; ++l) {
					if (typeAttributesHistogram[i][j][k].value == typeStat[l].value) {
						// found
						typeStat[l].freq += 1;
						found = 1;
						break;
					}
				}
				if (found == 0) {
					// create new entry
					typeStat = (TypeStat *) realloc(typeStat, sizeof(TypeStat) * (*typeStatCount + 1));
					if (!typeStat) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					typeStat[*typeStatCount].value = typeAttributesHistogram[i][j][k].value;
					typeStat[*typeStatCount].freq = 1;
					*typeStatCount += 1;
				}
			}
		}
	}

	qsort(typeStat, *typeStatCount, sizeof(TypeStat), compareTypeStats); // lowest freq first

	return typeStat;
}
#endif

#if USE_ONTOLOGY_NAMES
/* Group the attributes by the ontologies they belong to. */
static
str findOntologies(CS cs, int *propOntologiesCount, oid*** propOntologiesOids) {
	int		i, j, k;

	(*propOntologiesOids) = (oid **) malloc(sizeof(oid *) * ontologyCount);
	if (!(*propOntologiesOids)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < ontologyCount; ++i) {
		(*propOntologiesOids)[i] = NULL;
	}


	for (j = 0; j < cs.numProp; ++j) {
		int		fit;
		int		length = 0;
		char		**tokenizedUri = NULL;
		char		*token;			// token, modified during tokenization
		char		*uri;			// uri, modified during tokenization

		str		tmpStr;

		takeOid(cs.lstProp[j], &tmpStr);
		uri = (char *) malloc(sizeof(char) * (strlen(tmpStr) + 1));
		if (!uri) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(uri, tmpStr);

		// tokenize uri
		token = strtok(uri, "/#");
		while (token != NULL) {
			tokenizedUri = realloc(tokenizedUri, sizeof(char*) * ++length);
			if (!tokenizedUri) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			tokenizedUri[length -1] = (char *) malloc(sizeof(char *) * (strlen(token) + 1));
			if (!tokenizedUri[length - 1]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tokenizedUri[length - 1], token);
			token = strtok(NULL, "/#");
		}
		free(uri);
			
		//DUC: TODO: Do not need to tokenize the URI.
		//Use BAT with hash table to check the available of that URI 
		
		for (i = 0; i < ontologyCount; ++i) {
			// check for match with ontology
			if (length > ontologies[i].length) {
				fit = 1;
				for (k = 0; fit && k < ontologies[i].length; ++k) {
					if (strcmp(ontologies[i].uri[k], tokenizedUri[k]) != 0) {
						fit = 0;
					}
				}
				if (fit) {
					// found matching ontology, store property
					(*propOntologiesOids)[i] = realloc((*propOntologiesOids)[i], sizeof(oid) * (propOntologiesCount[i] + 1));
					if (!(*propOntologiesOids)[i]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

					(*propOntologiesOids)[i][propOntologiesCount[i]] = cs.lstProp[j];
					propOntologiesCount[i] += 1;
				}
			}
		}

		for (k = 0; k < length; ++k) {
			free(tokenizedUri[k]);
		}
		free(tokenizedUri);

		GDKfree(tmpStr);
	}

	return MAL_SUCCEED; 
}
#endif

#if USE_ONTOLOGY_NAMES
static
int compareOntologyCandidates (const void * a, const void * b) {
	float f1 = (*(ClassStat*)a).tfidfs;
	float f2 = (*(ClassStat*)b).tfidfs;

	if (f1 > f2) return -1;
	if (f2 > f1) return 1;
	
	//f1 = f2
	if ((*(ClassStat*)a).numMatchedProp > (*(ClassStat*)b).numMatchedProp) return -1;
	if ((*(ClassStat*)a).numMatchedProp < (*(ClassStat*)b).numMatchedProp) return 1;

	return 0; // sort descending
}
#endif

#if USE_ONTOLOGY_NAMES
/* For one CS: Calculate the ontology classes that are similar (tfidf) to the list of attributes. */
static
oid* getOntologyCandidates(oid** ontattributes, int ontattributesCount, oid** ontmetadata, int ontmetadataCount, int *resultCount, int** resultMatchedProp, oid **listOids, int *listCount, int listNum, PropStat *propStat, int freqId) {
	int		i, j, k, l;
	oid		*result = NULL;
	
	//if (freqId == 161) printf("listNum = %d\n",listNum);
	//Go through each ontology
	//listNum = 74
	//listCount[i] is the number props of this CS that appears in ontology i
	
	for (i = 0; i < listNum; ++i) {			//listNum = 74 
		int		filledListsCount = 0;
		oid		**candidates = NULL;
		int		*candidatesCount = NULL;
		ClassStat*	classStat = NULL;
		int		num;
		float		totalTfidfs;

		if (listCount[i] == 0) continue;

		candidates = (oid **) malloc(sizeof(oid *) * listCount[i]);
		candidatesCount = (int *) malloc(sizeof(int) * listCount[i]);
		if (!candidates || !candidatesCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < listCount[i]; ++j) {
			candidates[j] = NULL;
			candidatesCount[j] = 0;
		}
		//printf("Number of attribute in corresponding ontology is: %d \n", ontattributesCount);
		/*
		for (j = 0; j < ontattributesCount; ++j) {		//ontattributesCount = 70024
			oid auri = ontattributes[0][j];
			oid aattr = ontattributes[1][j];

			for (k = 0; k < listCount[i]; ++k) {
				if (aattr == listOids[i][k]) {
					// attribute found, store auristr in list
					candidates[k] = realloc(candidates[k], sizeof(oid) * (candidatesCount[k] + 1));
					if (!candidates[k]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					candidates[k][candidatesCount[k]] = auri;
					candidatesCount[k] += 1;
					if (candidatesCount[k] == 1) filledListsCount += 1; // new list
				}
			}
		}
		*/
		(void) ontattributes;
		(void) ontattributesCount;
		for (k = 0; k < listCount[i]; ++k) {
			BUN p, bun; 
			p = listOids[i][k];
			bun = BUNfnd(propStat->pBat, (ptr) &p);
			if (bun == BUN_NONE) continue; 
			else{
				candidates[k] = malloc(sizeof(oid) * (propStat->plCSidx[bun].numAdded));
				for (j = 0; j < propStat->plCSidx[bun].numAdded; j++){
					candidates[k][j] = propStat->plCSidx[bun].lstOnt[j];
				}
				candidatesCount[k] = propStat->plCSidx[bun].numAdded;
				filledListsCount = 1; 

			}
		}

		if (filledListsCount == 0) {
			free(candidatesCount);
			for (k = 0; k < listCount[i]; ++k) {
				if (candidates[k] != NULL) free(candidates[k]);
			}
			free(candidates);

			continue; // no new results will be generated using this ontology
		}
		totalTfidfs = 0.0;
		num = 0;
		for (j = 0; j < listCount[i]; ++j) { // for each list
			BUN p, bun;
			p = listOids[i][j];
			bun = BUNfnd(propStat->pBat, (ptr) &p);
			if (bun == BUN_NONE) continue; // property does not belong to an ontology class and therefore has no tfidfs score
			for (k = 0; k < candidatesCount[j]; ++k) { // for each candidate
				// search for this class
				int found = 0;
				for (l = 0; l < num; ++l) {
					if (candidates[j][k] == classStat[l].ontoClass) {
						// add tdidf^2 to sum
						classStat[l].tfidfs += (propStat->tfidfs[bun] * propStat->tfidfs[bun]);
						classStat[l].numMatchedProp++;
						found = 1;
						break;
					}
				}
				if (found == 0) {
					// create new entry
					classStat = (ClassStat *) realloc(classStat, sizeof(ClassStat) * (num + 1));
					if (!classStat) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					classStat[num].ontoClass = candidates[j][k]; // pointer, no copy
					classStat[num].totaltfidfs = 0.0;
					classStat[num].numMatchedProp = 1;
					classStat[num].tfidfs = (propStat->tfidfs[bun] * propStat->tfidfs[bun]);
					num += 1;
				}
			}
		}

		// calculate optimal tfidf score (all properties) & normalize tfidf sums
		totalTfidfs = 0.0;
		for (j = 0; j < listCount[i]; ++j) {
			BUN bun;
			if (candidatesCount[j] == 0) continue; // ignore properties without classes (dbpedia-specific issue)
			bun = BUNfnd(propStat->pBat, (ptr) &listOids[i][j]);
			totalTfidfs += (propStat->tfidfs[bun] * propStat->tfidfs[bun]);
		}
		for (j = 0; j < num; ++j) {
			classStat[j].tfidfs /= totalTfidfs;
		}

		// sort by tfidf desc
		qsort(classStat, num, sizeof(ClassStat), compareOntologyCandidates);

		// remove subclass if superclass is in list
		for (k = 0; k < num; ++k) {
			int found = 0;
			//if (freqId == 161) printf("   TFIDF score at %d ("BUNFMT") is: %f | Number of matched Prop %d \n",k, classStat[k].ontoClass, classStat[k].tfidfs,classStat[k].numMatchedProp);
			if (classStat[k].tfidfs < simTfidfThreshold) break; // values not frequent enough (list is sorted by tfidfs)
			for (j = 0; j < ontmetadataCount && (found == 0); ++j) {
				oid muri = ontmetadata[0][j];
				oid msuper = ontmetadata[1][j];
				if (classStat[k].ontoClass == muri) {
					if (msuper == BUN_NONE) {
						// muristr is a candidate! (because it has no superclass)
						result = realloc(result, sizeof(oid) * ((*resultCount) + 1));
						if (!result) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						result[*resultCount] = muri;
							
						resultMatchedProp[freqId] = realloc(resultMatchedProp[freqId], sizeof(int) * ((*resultCount) + 1));
						resultMatchedProp[freqId][*resultCount] = classStat[k].numMatchedProp;

						*resultCount += 1;
						found = 1;
						break;
					}

					for (l = 0; l < num; ++l) {
						if (classStat[k].tfidfs > classStat[l].tfidfs) break; // superclass has to have a higher/equal tfidf to be concidered
						if (msuper == classStat[l].ontoClass) {
							// superclass is in list, therefore do not copy subclass
							found = 1;
							break;
						}
					}
				}
			}
			if (found == 0) {
				// superclass not in list, classStat[k].ontoClass is a candidate!
				result = realloc(result, sizeof(oid) * ((*resultCount) + 1));
				if (!result) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				result[*resultCount] = classStat[k].ontoClass;

				resultMatchedProp[freqId] = realloc(resultMatchedProp[freqId], sizeof(int) * ((*resultCount) + 1));
				resultMatchedProp[freqId][*resultCount] = classStat[k].numMatchedProp;

				*resultCount += 1;
				break;
			}
		}

		for (k = 0; k < listCount[i]; ++k) {
			free(candidates[k]);
		}
		free(candidates);
		//free(candidatesIntersect);
		free(classStat);
		free(candidatesCount);
	}

	return result;
}
#endif

static
int* initOntologyLookupResultCount(int csCount) {
	int		*resultCount;
	int		i;

	resultCount = (int *) malloc(sizeof(int) * csCount);
	if (!resultCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < csCount; ++i) {
		resultCount[i] = 0;
	}
	return resultCount;
}

static
oid** initOntologyLookupResult(int csCount) {
	oid		**result; // result[cs][index] (list of class names per cs)
	int i;

	result = (oid **) malloc(sizeof(oid *) * csCount);
	if (!result) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < csCount; ++i) {
		result[i] = NULL;
	}
	return result;
}

static
int** initOntologyLookupResultMatchedProp(int csCount) {
	int		**resultMatchedProp; // resultMatchedProp[cs][index] Number of props matched between a CS and an ontology class
	int i;

	resultMatchedProp = (int **) malloc(sizeof(int *) * csCount);
	if (!resultMatchedProp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < csCount; ++i) {
		resultMatchedProp[i] = NULL;
	}
	return resultMatchedProp;
}

#if USE_ONTOLOGY_NAMES
//[DUC] Create propstat for ontology only 
static
void createPropStatistics(PropStat* propStat, oid** ontattributes, int ontattributesCount, int ontmetadataCount) {
	int		i;
	int		numProps = 0;

	for (i = 0; i < ontattributesCount; ++i) {
		oid attr = ontattributes[1][i];
		oid uri = ontattributes[0][i];
		// add prop to propStat
		BUN	bun = BUNfnd(propStat->pBat, (ptr) &attr);
		if (bun == BUN_NONE) {
			numProps++;
			if (propStat->pBat->T->hash && BATcount(propStat->pBat) > 4 * propStat->pBat->T->hash->mask) {
				HASHdestroy(propStat->pBat);
				BAThash(propStat->pBat, 2*BATcount(propStat->pBat));
			}

			BUNappend(propStat->pBat, &attr, TRUE);

			if (propStat->numAdded == propStat->numAllocation) {
				propStat->numAllocation += INIT_PROP_NUM;

				propStat->freqs = realloc(propStat->freqs, ((propStat->numAllocation) * sizeof(int)));
				propStat->tfidfs = realloc(propStat->tfidfs, ((propStat->numAllocation) * sizeof(float)));
				propStat->plCSidx = realloc(propStat->plCSidx, ((propStat->numAllocation) * sizeof(Postinglist)));
				if (!propStat->freqs || !propStat->tfidfs || !propStat->plCSidx) {fprintf(stderr, "ERROR: Couldn't realloc memory!\n");}

			}
			propStat->freqs[propStat->numAdded] = 1;

			//Store the list of ontology URI for each prop
			propStat->plCSidx[propStat->numAdded].lstIdx = NULL; 
			propStat->plCSidx[propStat->numAdded].lstInvertIdx = NULL; 
			propStat->plCSidx[propStat->numAdded].lstOnt = (oid *) malloc(sizeof(oid) * INIT_CS_PER_PROP);
			propStat->plCSidx[propStat->numAdded].lstOnt[0] = uri;
			propStat->plCSidx[propStat->numAdded].numAdded = 1;
			propStat->plCSidx[propStat->numAdded].numAllocation = INIT_CS_PER_PROP;

			propStat->numAdded++;
		} else {
			propStat->freqs[bun]++;
			if (propStat->plCSidx[bun].numAdded == propStat->plCSidx[bun].numAllocation){
				propStat->plCSidx[bun].numAllocation += INIT_CS_PER_PROP;
				propStat->plCSidx[bun].lstOnt = realloc(propStat->plCSidx[bun].lstOnt, ((propStat->plCSidx[bun].numAllocation) * sizeof(oid)));
			}
			propStat->plCSidx[bun].lstOnt[propStat->plCSidx[bun].numAdded] = uri;
			propStat->plCSidx[bun].numAdded++;
		}
	}

	for (i = 0; i < propStat->numAdded; ++i) {
		propStat->tfidfs[i] = log(((float)ontmetadataCount) / (1 + propStat->freqs[i]));
	}
}

//... [DUC]
#endif

#if USE_ONTOLOGY_NAMES
/* For all CS: Calculate the ontology classes that are similar (tfidf) to the list of attributes. */
static
void createOntologyLookupResult(oid** result, int** resultMatchedProp, CSset* freqCSset, int* resultCount, oid** ontattributes, int ontattributesCount, oid** ontmetadata, int ontmetadataCount) {
	int		i, j;
	PropStat	*propStat;

	propStat = initPropStat();

	//[DUC] Change the function for getting propStat. Use ontattributes for the propStat. 
	// Not the properties from freqCS
	createPropStatistics(propStat, ontattributes, ontattributesCount, ontmetadataCount);

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS		cs;
		oid		**propOntologiesOids = NULL;
		int		*propOntologiesCount = NULL;

		cs = (CS) freqCSset->items[i];

		// order properties by ontologies
		propOntologiesCount = (int *) malloc(sizeof(int) * ontologyCount);	//ontologyCount = 74
		if (!propOntologiesCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < ontologyCount; ++j) {
			propOntologiesCount[j] = 0;
		}
		
		//printf("Get ontology for FreqId %d. Orignal numProp = %d \n", i, cs.numProp);

		findOntologies(cs, propOntologiesCount, &propOntologiesOids);

		// get class names
		resultCount[i] = 0;
		
		result[i] = getOntologyCandidates(ontattributes, ontattributesCount, ontmetadata, ontmetadataCount, &(resultCount[i]), resultMatchedProp, propOntologiesOids, propOntologiesCount, ontologyCount, propStat, i);

		for (j = 0; j < ontologyCount; ++j) {
			free(propOntologiesOids[j]);
		}
		free(propOntologiesOids);
		free(propOntologiesCount);
	}
	freePropStat(propStat);
}
#endif

/*
 * Graphical UML-like schema representation.
 * Table width encodes table coverage (wider = more coverage), column colors encode column support (darker = more filled)
 * Call GraphViz to create the graphic: "dot -Tpdf -O UMLxxx.dot" to create "UMLxxx.dot.pdf"
 */
static
void printUML2(CStableStat *cstablestat, CSPropTypes* csPropTypes, int freqThreshold, CSrel *csRelFinalFKs, BATiter mapi, BAT *mbat, int numTables, int* mTblIdxFreqIdxMapping, int* csTblIdxMapping, CSset* freqCSset) {
	int 		i, j, k;
	FILE 		*fout;
	char 		filename[20], tmp[10];

	int		smallest = -1, biggest = -1;

	(void) csTblIdxMapping;

	strcpy(filename, "UML");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".dot");

	fout = fopen(filename, "wt");

	// header
	fprintf(fout, "digraph g {\n");
	fprintf(fout, "graph[ratio=\"compress\"];\n");
	fprintf(fout, "node [shape=\"none\"];\n\n");

	// find biggest and smallest table
	for (i = 0; i < numTables; ++i) {
		int csIdx = mTblIdxFreqIdxMapping[i];

		// set first values
		if (smallest == -1) smallest = csIdx;
		if (biggest == -1) biggest = csIdx;

		if (freqCSset->items[csIdx].coverage < freqCSset->items[smallest].coverage) smallest = csIdx;
		if (freqCSset->items[csIdx].coverage > freqCSset->items[biggest].coverage) biggest = csIdx;
	}

	// for each table
	for (i = 0; i < numTables; ++i) {
		int csIdx = mTblIdxFreqIdxMapping[i];
		int width;
		str labelStrEscaped = NULL;

		if(!isCSTable(freqCSset->items[csIdx], cstablestat->lstcstable[i].tblname)) continue; // ignore small tables

		// print table header
		// set table width between 300 (smallest coverage) and 600 (biggest coverage) px, using log10 logarithm
		width = (int) ((300 + 300 * (log10(freqCSset->items[csIdx].coverage) - log10(freqCSset->items[smallest].coverage)) / (log10(freqCSset->items[biggest].coverage) - log10(freqCSset->items[smallest].coverage))) + 0.5);
		fprintf(fout, "\"%d\" [\n", i);
		fprintf(fout, "label = <<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n");

		getSqlName(&labelStrEscaped, cstablestat->lstcstable[i].tblname, mapi, mbat);
		fprintf(fout, "<TR><TD WIDTH=\"%d\"><B>%s (#triples: %d, #tuples: %d)</B></TD></TR>\n", width, labelStrEscaped, freqCSset->items[csIdx].coverage, freqCSset->items[csIdx].support);
		GDKfree(labelStrEscaped);

		// print columns
		for (j = 0; j < csPropTypes[i].numProp; ++j) {
			str		propStr;
			str		tmpStr;
			char    *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char    *propStrShort = NULL;
#endif
			str color;

#if REMOVE_INFREQ_PROP
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1) continue; // ignore infrequent props
#endif

			takeOid(freqCSset->items[csIdx].lstProp[j], &tmpStr);

			// assign color (the more tuples the property occurs in, the darker)
			if ((1.0 * freqCSset->items[csIdx].lstPropSupport[j])/freqCSset->items[csIdx].support > 0.8) {
				color = "#5555FF";
			} else if ((1.0 * freqCSset->items[csIdx].lstPropSupport[j])/freqCSset->items[csIdx].support > 0.6) {
				color = "#7777FF";
			} else if ((1.0 * freqCSset->items[csIdx].lstPropSupport[j])/freqCSset->items[csIdx].support > 0.4) {
				color = "#9999FF";
			} else if ((1.0 * freqCSset->items[csIdx].lstPropSupport[j])/freqCSset->items[csIdx].support > 0.2) {
				color = "#BBBBFF";
			} else {
				color = "#DDDDFF";
			}

			// escape column names
			propStr = removeBrackets(tmpStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			fprintf(fout, "<TR><TD BGCOLOR=\"%s\" PORT=\"%s\">%s (%d%%)</TD></TR>\n", color, propStrEscaped, propStrShort, (100 * freqCSset->items[csIdx].lstPropSupport[j])/freqCSset->items[csIdx].support);
			GDKfree(propStrShort);
#else
			fprintf(fout, "<TR><TD BGCOLOR=\"%s\" PORT=\"%s\">%s (%d%%)</TD></TR>\n", color, propStrEscaped, propStrEscaped, (100 * freqCSset->items[csIdx].lstPropSupport[j])/freqCSset->items[csIdx].support);
#endif

			GDKfree(propStr);
			free(propStrEscaped);
			GDKfree(tmpStr); 

		}
		fprintf(fout, "</TABLE>>\n");
		fprintf(fout, "];\n\n");
	}

	// for each foreign key relationship
	for (i = 0; i < numTables; ++i) {
		int from = i;
		CSrel rel = csRelFinalFKs[from];

		for (j = 0; j < rel.numRef; ++j) {
			int to = rel.lstRefFreqIdx[j];
			oid prop = rel.lstPropId[j];
			str tmpStr;
			str propStr;
			char *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char *propStrShort = NULL;
#endif

#if REMOVE_INFREQ_PROP
			// find prop
			k = 0;
			while (freqCSset->items[mTblIdxFreqIdxMapping[from]].lstProp[k] != prop) ++k;
			if (csPropTypes[from].lstPropTypes[k].defColIdx == -1) continue; // ignore infrequent props
#endif

			takeOid(prop, &tmpStr);

			// escape column names
			propStr = removeBrackets(tmpStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);

#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			fprintf(fout, "\"%d\":\"%s\" -> \"%d\" [label=\"%s\"];\n", from, propStrEscaped, to, propStrShort); // print foreign keys to dot file
			GDKfree(propStrShort);
#else
			fprintf(fout, "\"%d\":\"%s\" -> \"%d\" [label=\"%s\"];\n", from, propStrEscaped, to, propStrEscaped); // print foreign keys to dot file
#endif

			GDKfree(propStr);
			free(propStrEscaped);
			GDKfree(tmpStr); 
		}
	}

	fprintf(fout, "}\n"); // footer

	fclose(fout);
}

static
oid* getOntoHierarchy(oid ontology, int* hierarchyCount, oid** ontmetadata, int ontmetadataCount) {
	int		i;
	oid		*hierarchy;
	int		foundTop;

	// add 'ontology' to hierarchy
	(*hierarchyCount) = 1;
	hierarchy = (oid *) GDKmalloc(sizeof(oid) * (*hierarchyCount));
	if (!hierarchy)
		fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	hierarchy[(*hierarchyCount) -1] = ontology;

	// follow the hierarchy from bottom to top
	foundTop = 0;
	while (!foundTop) {
		// lookup superclass
		int foundTuple = 0;
		for (i = 0; i < ontmetadataCount; ++i) {
			oid muri = ontmetadata[0][i];
			oid msuper = ontmetadata[1][i];

			if (hierarchy[(*hierarchyCount) - 1] == muri) {
				// found entry
				foundTuple = 1;
				if (msuper == BUN_NONE) {
					// no superclass
					foundTop = 1;
					break;
				} else {
					// superclass
					// add 'msuperstr' to hierarchy
					(*hierarchyCount) += 1;
					hierarchy = GDKrealloc(hierarchy, sizeof(oid) * (*hierarchyCount));
					if (!hierarchy)
						fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					hierarchy[(*hierarchyCount) -1] = msuper;
					break;
				}
			}
		}
		if (!foundTuple) {
			// value not found in 'ontmetadata' --> end of hierarchy
			foundTop = 1;
		}
	}
	return hierarchy;
}

/* Remove duplicated candidate values and remove DUMMY values if better candidates exist
 */
static
void removeDuplicatedCandidates(CSlabel *label) {
	int i, j;
	int cNew = label->candidatesNew, cType = label->candidatesType, cOnto = label->candidatesOntology, cFK = label->candidatesFK;

	if (label->candidatesCount < 2) return; // no duplicates

	// loop through all candidates
	for (i = 0; i < label->candidatesCount - 1; ++i) {
		// search (direction: right) whether this value occurs again
		int moveLeft = 0;
		for (j = i + 1; j < label->candidatesCount; ++j) {
			// find out which category (new, onto, type, fk) we are in
			int *cPtr = NULL;
			if (j < label->candidatesNew) cPtr = &cNew;
			else if (j < label->candidatesNew + label->candidatesType) cPtr = &cType;
			else if (j < label->candidatesNew + label->candidatesType + label->candidatesOntology) cPtr = &cOnto;
			else cPtr = &cFK;

			if (label->candidates[i] == label->candidates[j] || label->candidates[j] == BUN_NONE) {
				// DUMMY value will be overwritten
				// OR:
				// value occurs again, will be overwritten
				moveLeft++;
				(*cPtr)--;
			} else {
				// different value, keep it
				label->candidates[j - moveLeft] = label->candidates[j];
			}
		}
		// value 'i' is unique now
		// update counts
		label->candidatesCount -= moveLeft;
		label->candidatesNew = cNew;
		label->candidatesType = cType;
		label->candidatesOntology = cOnto;
		label->candidatesFK = cFK;
	}

	// remove DUMMY value on position 0
	if (label->candidates[0] == BUN_NONE && label->candidatesCount > 1) {
		for (i = 1; i < label->candidatesCount; ++i) {
			label->candidates[i - 1] = label->candidates[i];
		}
		label->candidatesCount--;

		// update value in category;
		if (label->candidatesNew > 0) {
			label->candidatesNew--;
		} else if (label->candidatesType > 0) {
			label->candidatesType--;
		} else if (label->candidatesOntology > 0) {
			label->candidatesOntology--;
		} else {
			label->candidatesFK--;
		}
	}
}

#if USE_TABLE_NAME
/* For one CS: Choose the best table name out of all collected candidates (ontology, type, fk). */
/**
 * The priority is:
 * Ontology-based type values >  Ontology-based name > Type value > FK name > Non frequent type value
 * 
 */
static
void getTableName(CSlabel* label, CSset* freqCSset, int csIdx,  int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, TypeStat* typeStat, int typeStatCount, oid** result,int** resultMatchedProp, int* resultCount, IncidentFKs* links, oid** ontmetadata, int ontmetadataCount, BAT *ontmetaBat, OntClass *ontclassSet) {
	int		i, j;
	oid		*tmpList;
	int		tmpListCount;
	char		nameFound = 0;
	oid		maxDepthOid;
	int		maxFreq = 0;
	
	#if TYPE_TFIDF_RANKING	
	oid		maxRankscoreOid; 
	float		maxRankscore = 0.0; 
	float 		tmprankscore = 0.0; 
	int		maxRankscoreFreq; 
	#endif

	//for choosing the right type values
	BUN		ontClassPos;
	oid		typeOid;
	int 		depth, maxDepth;
	int 		freq;
	
	char		foundOntologyTypeValue = 0; 	
	oid		choosenOntologyTypeValue = BUN_NONE;
	int		choosenFreq = 0;

	int		bestOntCandIdx = -1;
	int		isGoodTypeExist = 0; 

	(void) ontmetaBat;
	// --- TYPE ---
	// get most frequent type value per type attribute
	tmpList = NULL;
	tmpListCount = 0;
	#if INFO_NAME_FREQUENCY
	label->nameFreq = 0;
	label->ontologySimScore = 0.0;
	#endif
	
	for (i = 0; i < typeAttributesCount; ++i) {
		foundOntologyTypeValue = 0;
		if (typeAttributesHistogramCount[csIdx][i] == 0) continue;
		/*   //TODO: Uncomment this path
		for (j = 0; j < typeAttributesHistogramCount[csIdx][i]; j++){
			str typelabel; 
			BUN		ontClassPos; 	//Position of ontology in the ontmetaBat
			oid		typeOid; 	

			typeOid = typeAttributesHistogram[csIdx][i][j].value;
			printf("FreqCS %d : Type[%d][%d][oid] = " BUNFMT, csIdx, i,j, typeOid);
			ontClassPos = BUNfnd(ontmetaBat, &typeOid); 
			if (ontClassPos != BUN_NONE){
				takeOid(typeOid,&typelabel);
				assert(ontclassSet[ontClassPos].cOid == typeOid); 
				printf(" --> class %s | Index = %d |Specific level: %d \n", typelabel, (int)ontClassPos, ontclassSet[ontClassPos].hierDepth);
				GDKfree(typelabel);
			}
			else{
				printf(" --> No class \n");	
			}
		}
		*/
		
		if (typeAttributesHistogram[csIdx][i][0].percent < TYPE_FREQ_THRESHOLD) continue; // sorted
		if (typeAttributesHistogram[csIdx][i][0].percent > GOOD_TYPE_FREQ_THRESHOLD) isGoodTypeExist = 1;

		tmpList = (oid *) realloc(tmpList, sizeof(oid) * (tmpListCount + 1));
		if (!tmpList) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

		// of all values that are >= TYPE_FREQ_THRESHOLD, choose the value with the highest hierarchy level ("deepest" value)
		maxDepthOid = typeAttributesHistogram[csIdx][i][0].value;
		maxFreq = typeAttributesHistogram[csIdx][i][0].freq;
		#if TYPE_TFIDF_RANKING
		maxRankscore = typeAttributesHistogram[csIdx][i][0].rankscore; 
		maxRankscoreOid = typeAttributesHistogram[csIdx][i][0].value;
		maxRankscoreFreq = typeAttributesHistogram[csIdx][i][0].freq;
		#endif

		ontClassPos = BUNfnd(ontmetaBat, &maxDepthOid);
		if ( ontClassPos != BUN_NONE){
			foundOntologyTypeValue = 1;
			maxDepth = ontclassSet[ontClassPos].hierDepth;
		}	
		else{
			maxDepth = -1;
		}


		for (j = 1; j < typeAttributesHistogramCount[csIdx][i]; ++j) {

			if (typeAttributesHistogram[csIdx][i][j].percent < TYPE_FREQ_THRESHOLD) break;
			
			typeOid = typeAttributesHistogram[csIdx][i][j].value;
			ontClassPos = BUNfnd(ontmetaBat, &typeOid);
			if (ontClassPos != BUN_NONE){
				foundOntologyTypeValue = 1;
				depth = ontclassSet[ontClassPos].hierDepth;
				freq = typeAttributesHistogram[csIdx][i][j].freq;

				if (depth > maxDepth) {
					// choose value with higher hierarchy level
					maxDepthOid = typeAttributesHistogram[csIdx][i][j].value;
					maxFreq = freq;
					maxDepth = depth;
				} else if (depth == maxDepth && freq > maxFreq) {
					// if both values are on the same level, choose the value with higher frequency
					maxDepthOid = typeAttributesHistogram[csIdx][i][j].value;
					maxFreq = freq;
				}

				#if TYPE_TFIDF_RANKING
				tmprankscore = typeAttributesHistogram[csIdx][i][j].rankscore; 
				if (tmprankscore > maxRankscore){
					maxRankscore = tmprankscore; 
					maxRankscoreOid = typeAttributesHistogram[csIdx][i][j].value;
					maxRankscoreFreq = typeAttributesHistogram[csIdx][i][j].freq; 	
				}
				#endif
			}
		}

		tmpList[tmpListCount] = maxDepthOid;
		tmpListCount += 1;

		if (foundOntologyTypeValue){
			choosenOntologyTypeValue = maxDepthOid;
			choosenFreq = maxFreq;
			#if TYPE_TFIDF_RANKING
			choosenOntologyTypeValue = maxRankscoreOid; 
			choosenFreq = maxRankscoreFreq; 
			#endif
		}
	}

	// add all most frequent type values to list of candidates
	if (tmpListCount >= 1) {
		int counter = 0;
		label->candidatesType = tmpListCount;
		label->candidates = GDKrealloc(label->candidates, sizeof(oid) * (label->candidatesCount + tmpListCount));
		if (!label->candidates) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		for (i = 0; i < typeStatCount; ++i) {
			for (j = 0; j < tmpListCount; ++j) {
				if (typeStat[i].value == tmpList[j]) {
					label->candidates[label->candidatesCount + counter] = tmpList[j];
					counter++;
				}
			}
		}
		assert(counter == tmpListCount);
		label->candidatesCount += tmpListCount;
	}

	if (!nameFound) {
		// one type attribute --> use most frequent one
		if (tmpListCount == 1) {
			// only one type attribute, use most frequent value (sorted)
			label->name = tmpList[0];
			nameFound = 1;
			#if INFO_WHERE_NAME_FROM
			label->isType = 1; 
			#endif

			#if INFO_NAME_FREQUENCY
			label->nameFreq = maxFreq;
			label->ontologySimScore = 0.0;
			#endif
		}
	}

	//If there is any ontology-based type value, use it for the name
	if (!nameFound){
		if (choosenOntologyTypeValue != BUN_NONE){
			label->name = choosenOntologyTypeValue;
			nameFound = 1;
			
			#if INFO_WHERE_NAME_FROM
			label->isType = 1; 
			#endif
			
			#if INFO_NAME_FREQUENCY
			label->nameFreq = choosenFreq;
			label->ontologySimScore = 0.0;
			#endif
		}
	}

	if (!nameFound) {
		// multiple type attributes --> use the one with fewest occurances in other CS's
		if (tmpListCount > 1) {
			for (i = 0; i < typeStatCount && !nameFound; ++i) {
				for (j = 0; j < tmpListCount && !nameFound; ++j) {
					if (typeStat[i].value == tmpList[j]) {
						label->name = tmpList[j];
						nameFound = 1;

						#if INFO_WHERE_NAME_FROM
						label->isType = 1; 
						#endif
						
						#if INFO_NAME_FREQUENCY
						label->nameFreq = maxFreq;	//This is not really true. The name freq can be smaller
						label->ontologySimScore = 0.0;
						#endif
					}
				}
			}
		}
	}
		
	// --- ONTOLOGY ---
	// add all ontology candidates to list of candidates
	// Find the best candidate by looking at the number of matched prop
	// between the CS and the ontology candidate
	
	// If the name found previously (based on the type values) is not 
	// an ontology-based value (e.g., simply a string), and not a really good (so frequent) type value 
	// we will choose the ontology name for the CS's name. 
	
	// chose the best ontology candidate based on number of matched props as label 
	// TODO: Improve this score a bit, by choosing the higher tfidf score, than number of matched prop
	
	if (choosenOntologyTypeValue == BUN_NONE && isGoodTypeExist == 0 && resultCount[csIdx] >= 1){

		// Only put ontology-based class to the candidate if it is choosen as the class name
		{
		int maxNumMatchedProp = -1;
		bestOntCandIdx = 0;
		label->candidatesOntology = resultCount[csIdx];
		label->candidates = GDKrealloc(label->candidates, sizeof(oid) * (label->candidatesCount + resultCount[csIdx]));
		if (!label->candidates) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		for (i = 0; i < resultCount[csIdx]; ++i) {
			label->candidates[label->candidatesCount + i] = result[csIdx][i];
			if (resultMatchedProp[csIdx][i] > maxNumMatchedProp){
				maxNumMatchedProp = resultMatchedProp[csIdx][i];
				bestOntCandIdx = i;
			}
		}
		label->candidatesCount += resultCount[csIdx];
		}
		

		label->name = result[csIdx][bestOntCandIdx];
		nameFound = 1;
		#if INFO_WHERE_NAME_FROM
		label->isOntology = 1; 
		#endif
		
	}


	// --- FK ---
	// add top3 fk values to list of candidates
	if (links[csIdx].num > 0) {
		//Only add the FK name, if its number of references is large enought
		if ((links[csIdx].fks[0].freq * 100) > (FK_MIN_REFER_PERCENTAGE * freqCSset->items[csIdx].support)){
			label->candidatesFK = MIN(3, links[csIdx].num);
			label->candidates = GDKrealloc(label->candidates, sizeof(oid) * (label->candidatesCount + MIN(3, links[csIdx].num)));
			if (!label->candidates) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			for (i = 0; i < MIN(3, links[csIdx].num); ++i) {
				label->candidates[label->candidatesCount + i] = links[csIdx].fks[0].prop;
			}
			label->candidatesCount += MIN(3, links[csIdx].num);
		}
	}

	if (!nameFound) {
		// incident foreign keys --> use the one with the most occurances (num and freq)
		if (links[csIdx].num > 0) {
			if ((links[csIdx].fks[0].freq * 100) > (FK_MIN_REFER_PERCENTAGE * freqCSset->items[csIdx].support)){
				label->name = links[csIdx].fks[0].prop; // sorted
				nameFound = 1;

				#if INFO_WHERE_NAME_FROM
				label->isFK = 1; 
				#endif
				
				#if INFO_NAME_FREQUENCY
				label->nameFreq = links[csIdx].fks[0].freq;
				label->ontologySimScore = 0.0;
				#endif
			}
		}
	}
	
	
	//Add hierarchy information for ontology-based name
	if (nameFound){
		ontClassPos = BUNfnd(ontmetaBat, &(label->name));
		if ( ontClassPos != BUN_NONE){
			label->hierarchy = getOntoHierarchy(label->name, &(label->hierarchyCount), ontmetadata, ontmetadataCount);
		}
	}

	//if no name is found, check again the typecount to assign a name
	#if USE_BEST_TYPEVALUE_INSTEADOF_DUMMY
	if (!nameFound){
		for (i = 0; i < typeAttributesCount; ++i){
			if (typeAttributesHistogramCount[csIdx][i] == 0) continue;
			
			if (typeAttributesHistogram[csIdx][i][0].percent < MIN_POSSIBLE_TYPE_FREQ_THRESHOLD) continue; 

			//printf("Current candidate count = %d",label->candidatesCount);
			label->candidatesType = 1;
			label->candidates = GDKrealloc(label->candidates, sizeof(oid));
			label->candidatesCount = 1;
			label->name = typeAttributesHistogram[csIdx][i][0].value;
			label->candidates[0] = typeAttributesHistogram[csIdx][i][0].value;
			nameFound = 1;
			#if INFO_WHERE_NAME_FROM
			label->isType = 1; 
			#endif

			#if INFO_NAME_FREQUENCY
			label->nameFreq = typeAttributesHistogram[csIdx][i][0].percent;
			label->ontologySimScore = 0.0;
			#endif
			break; 
		}
	}
	#endif

	// --- NOTHING ---
	if (label->candidatesCount == 0) {
		label->candidatesNew = 1;
		label->candidates = GDKrealloc(label->candidates, sizeof(oid));
		if (!label->candidates) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		label->candidates[0] = BUN_NONE;
		label->candidatesCount = 1;
	}

	if (!nameFound) {
		label->name = BUN_NONE;
		nameFound = 1;
	}
	
	// de-duplicate
	removeDuplicatedCandidates(label);

	if(tmpList != NULL) free(tmpList);
	return;
}
#endif

static
CSlabel* initLabels(CSset *freqCSset) {
	CSlabel		*labels;
	int		i;

	labels = (CSlabel *) GDKmalloc(sizeof(CSlabel) * freqCSset->numCSadded);
	if (!labels) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		labels[i].name = BUN_NONE; 
		labels[i].candidates = NULL;
		labels[i].candidatesCount = 0;
		labels[i].candidatesNew = 0;
		labels[i].candidatesType = 0;
		labels[i].candidatesOntology = 0;
		labels[i].candidatesFK = 0;
		labels[i].hierarchy = NULL;
		labels[i].hierarchyCount = 0;
		labels[i].numProp = 0;
		labels[i].lstProp = NULL;
		#if INFO_WHERE_NAME_FROM
		labels[i].isOntology = 0; 
		labels[i].isType = 0; 
		labels[i].isFK = 0; 
		#endif
	}
	return labels;
}

#if USE_TABLE_NAME
/* Creates the final result of the labeling: table name and attribute names. */
static
void getAllLabels(CSlabel* labels, CSset* freqCSset,  int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, TypeStat* typeStat, int typeStatCount, oid** result, int** resultMatchedProp, int* resultCount, IncidentFKs* links, oid** ontmetadata, int ontmetadataCount, BAT *ontmetaBat, OntClass *ontclassSet) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];

		// get table name
		getTableName(&labels[i], freqCSset, i,  typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeStat, typeStatCount, result, resultMatchedProp, resultCount, links, ontmetadata, ontmetadataCount, ontmetaBat, ontclassSet);

		// copy attribute oids (names)
		labels[i].numProp = cs.numProp;
		labels[i].lstProp = (oid *) GDKmalloc(sizeof(oid) * cs.numProp);
		if (!labels[i].lstProp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < cs.numProp; ++j) {
			labels[i].lstProp[j] = cs.lstProp[j];
		}
	}
}
#endif

#if USE_FK_NAMES
static
int compareIncidentFKs (const void * a, const void * b) {
	IncidentFK c = *(IncidentFK*) a;
	IncidentFK d = *(IncidentFK*) b;

	// higher num
	if (c.num > d.num) return -1;
	if (c.num < d.num) return 1;

	// higher freq
	if (c.freq > d.freq) return -1;
	if (c.freq < d.freq) return 1;

	return 0;
}
#endif

#if USE_FK_NAMES
/* Calculate frequency statistics for incident foreign keys. */
static
void createLinks(CSset* freqCSset, Relation*** relationMetadata, int** relationMetadataCount, IncidentFKs* links) {
	int		i, j, k, l;
	int		found;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];
		for (j = 0; j < cs.numProp; ++j) {
			for (k = 0; k < relationMetadataCount[i][j]; ++k) {
				int to;
				if (relationMetadata[i][j][k].percent < FK_FREQ_THRESHOLD) continue; // foreign key is not frequent enough
				to = relationMetadata[i][j][k].to;

				found = 0;
				for (l = 0; l < links[to].num; ++l) {
					if (links[to].fks[l].prop == cs.lstProp[j]) {
						// update frequency and num
						found = 1;
						links[to].fks[l].freq += relationMetadata[i][j][k].freq;
						links[to].fks[l].num += 1;
						break;
					}
				}
				if (found == 0) {
					// new
					links[to].fks = realloc(links[to].fks, sizeof(IncidentFK) * (links[to].num + 1));
					if (!links[to].fks) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					links[to].fks[links[to].num].prop = cs.lstProp[j];
					links[to].fks[links[to].num].freq = relationMetadata[i][j][k].freq;
					links[to].fks[links[to].num].num = 1;
					links[to].num += 1;
				}
			}
		}
	}

	// sort links
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		qsort(links[i].fks, links[i].num, sizeof(IncidentFK), compareIncidentFKs);
	}
}
#endif

static
void createOntoUsageTreeStatistics(OntoUsageNode* tree, int numTuples) {
	int i;

	if (tree->numChildren == 0) {
		// leaf node
		tree->numOccurancesSum = tree->numOccurances;
		tree->percentage = (1.0 * tree->numOccurancesSum) / numTuples;
	} else {
		// inner node
		tree->numOccurancesSum = tree->numOccurances;
		for (i = 0; i < tree->numChildren; ++i) {
			createOntoUsageTreeStatistics(tree->lstChildren[i], numTuples);
			// sum up data
			tree->numOccurancesSum += tree->lstChildren[i]->numOccurancesSum;
		}
		tree->percentage = (1.0 * tree->numOccurancesSum) / numTuples;
	}
}

static
void addToOntoUsageTree(OntoUsageNode* tree, oid* hierarchy, int hierarchyCount, int numTuples) {
	int		i;
	oid		uri;
	OntoUsageNode	*leaf;

	if (hierarchyCount == 0) {
		// found position in tree
//		tree->numOccurances += numTuples; // TODO cs.support not yet available
		tree->numOccurances += 1;
		return;
	}

	// search through children
	uri  = hierarchy[hierarchyCount - 1];
	hierarchyCount--;
	for (i = 0; i < tree->numChildren; ++i) {
		if (tree->lstChildren[i]->uri == uri) {
			// found
			addToOntoUsageTree(tree->lstChildren[i], hierarchy, hierarchyCount, numTuples);
			return;
		}
	}

	// child not found
	// create leaf
	leaf = (OntoUsageNode *) malloc(sizeof(OntoUsageNode));
	if (!leaf)
		fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	leaf->parent = tree;
	leaf->uri = uri;
	leaf->lstChildren = NULL;
	leaf->numChildren = 0;
	leaf->numOccurances = 0;
	leaf->numOccurancesSum = 0;
	leaf->percentage = 0.0;
	// add to tree
	tree->numChildren++;
	tree->lstChildren = realloc(tree->lstChildren, sizeof(OntoUsageNode *) * tree->numChildren);
	if (!tree->lstChildren)
		fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
	tree->lstChildren[tree->numChildren - 1] = leaf;
	// call
	addToOntoUsageTree(leaf, hierarchy, hierarchyCount, numTuples);
}

static
void printTreePrivate(OntoUsageNode* tree, int level, FILE* fout) {
	int i;
	str uriStr, uriStrShort;

	if (tree->parent) {
		takeOid(tree->uri, &uriStr);
		getPropNameShort(&uriStrShort, uriStr);
		fprintf(fout, BUNFMT" [label = \"%s (%.1f%%)\"];\n", tree->uri, uriStrShort, tree->percentage * 100);
		fprintf(fout, BUNFMT"--"BUNFMT";\n", tree->uri, tree->parent->uri);
		GDKfree(uriStrShort);
		GDKfree(uriStr);
	} else {
		// artifical root, has no name
		fprintf(fout, BUNFMT" [label = \"ROOT (%.1f%%)\"];\n", tree->uri, tree->percentage * 100);
	}
	for (i = 0; i < tree->numChildren; ++i) {
		printTreePrivate(tree->lstChildren[i], level+1, fout);
	}
}

/*
 * Print ontology tree to file, dot code
 */
static
void printTree(OntoUsageNode* tree) {
	FILE *fout = fopen("ontoUsageTree.txt", "wt");

	// header
	fprintf(fout, "graph g {\n");
	fprintf(fout, "graph [ratio = \"compress\", rankdir = \"RL\"];\n");
	fprintf(fout, "node [shape = \"box\"];\n\n");
	// body
	printTreePrivate(tree, 0, fout);
	// footer

	fprintf(fout, "}\n");
	fclose(fout);
}

static
void createOntoUsageTree(OntoUsageNode** tree, CSset* freqCSset, oid** ontmetadata, int ontmetadataCount, BAT *ontmetaBat,CSlabel* labels) {
	int 		i;
	int 		numTuples = 0;
	BUN		pos; 

	// init tree with an artifical root node
	(*tree) = (OntoUsageNode *) malloc(sizeof(OntoUsageNode));
	if (!(*tree))
		fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	(*tree)->parent = NULL; // artificial root
	(*tree)->lstChildren = NULL;
	(*tree)->numChildren = 0;
	(*tree)->numOccurances = 0;
	(*tree)->numOccurancesSum = 0;
	(*tree)->percentage = 0.0;

	// loop through data
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		oid		uri;
		int		hierarchyCount = 0;
		oid*		hierarchy;

		uri = labels[i].name; 	
		if (uri == BUN_NONE) continue; 	//No name freqCS
	
		//Check if the name is ontology name	
             	pos = BUNfnd(ontmetaBat, &uri);
	        if (pos == BUN_NONE) continue; // no ontology information, ignore

		// get ontology hierarchy
		hierarchy = getOntoHierarchy(uri, &hierarchyCount, ontmetadata, ontmetadataCount);

		// search class in tree and add CS to statistics
		addToOntoUsageTree(*tree, hierarchy, hierarchyCount, freqCSset->items[i].support);
		GDKfree(hierarchy);
//		numTuples += freqCSset->items[i].support; // update total number of tuples in dataset // TODO cs.support not yet available
		numTuples += 1;
	}

	// calculate summed parameters
	createOntoUsageTreeStatistics(*tree, numTuples);

	// print
	if(0){
	printf("Ontology tree:\n");
	printTree(*tree);
	}
}

static
void freeTypeAttributesHistogram(TypeAttributesFreq*** typeAttributesHistogram, int csCount, int typeAttributesCount) {
	int		i, j;

	for (i = 0; i < csCount; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			free(typeAttributesHistogram[i][j]);
		}
		free(typeAttributesHistogram[i]);
	}
	free(typeAttributesHistogram);
}

static
void freeTypeAttributesHistogramCount(int** typeAttributesHistogramCount, int csCount) {
	int		i;

	for (i = 0; i < csCount; ++i) {
		free(typeAttributesHistogramCount[i]);
	}
	free(typeAttributesHistogramCount);
}

static
void freeRelationMetadataCount(int** relationMetadataCount, int csCount) {
	int		i;

	for (i = 0; i < csCount; ++i) {
		free(relationMetadataCount[i]);
	}
	free(relationMetadataCount);
}

static
void freeRelationMetadata(Relation*** relationMetadata, CSset* freqCSset) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) { // CS
		CS cs = (CS) freqCSset->items[i];
		for (j = 0; j < cs.numProp; ++j) {
			if (relationMetadata[i][j])
				free(relationMetadata[i][j]);
		}
		if (relationMetadata[i])
			free(relationMetadata[i]);
	}
	free(relationMetadata);
}

static
void freeLinks(IncidentFKs* links, int csCount) {
	int		i;

	for (i = 0; i < csCount; ++i) {
		if (links[i].num > 0)
			free(links[i].fks);
	}
	free(links);
}

//static
//void freeOntattributes(str** ontattributes) {
//	free(ontattributes[0]);
//	free(ontattributes[1]);
//	free(ontattributes);
//}
//
//static
//void freeOntmetadata(str** ontmetadata) {
//	free(ontmetadata[0]);
//	free(ontmetadata[1]);
//	free(ontmetadata);
//}

static
void freeOntologyLookupResult(oid** ontologyLookupResult, int** ontologyLookupResutMatchedProp, int csCount) {
	int		i;

	for (i = 0; i < csCount; ++i) {
		if (ontologyLookupResult[i])
			free(ontologyLookupResult[i]);
		if (ontologyLookupResutMatchedProp[i])
			free(ontologyLookupResutMatchedProp[i]);
	}
	free(ontologyLookupResult);
	free(ontologyLookupResutMatchedProp);
}

#if USE_TYPE_NAMES
char*		typeAttributes[] = {
			"<http://ogp.me/ns#type>",
			"<https://ogp.me/ns#type>",
			"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
			"<http://purl.org/dc/elements/1.1/type>",
			"<http://mixi-platform.com/ns#type>",
			"<http://ogp.me/ns/fb#type>",
			"<http://opengraph.org/schema/type>",
			"<http://opengraphprotocol.org/schema/type>",
			"<http://purl.org/dc/terms/type>",
			"<http://purl.org/goodrelations/v1#typeOfGood>",
			"<http://search.yahoo.com/searchmonkey/media/type>",
			"<https://opengraphprotocol.org/schema/type>",
			"<https://search.yahoo.com/searchmonkey/media/type>",
			"<http://www.w3.org/1999/xhtmltype>",
			"<http://dbpedia.org/ontology/longtype>",
			"<http://dbpedia.org/ontology/type>",
			"<http://dbpedia.org/ontology/typeOfElectrification>",
			"<http://dbpedia.org/property/type>"}; // <...> necessary to get the correct oids

int			typeAttributesCount = 18;
#endif

/* Creates labels for all CS (without a parent). */
CSlabel* createLabels(CSset* freqCSset, CSrel* csrelSet, int num, BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, int *csIdFreqIdxMap, oid** ontattributes, int ontattributesCount, oid** ontmetadata, int ontmetadataCount, OntoUsageNode** ontoUsageTree, BAT *ontmetaBat, OntClass *ontclassSet) {

	int			**typeAttributesHistogramCount;
	TypeAttributesFreq	***typeAttributesHistogram;
	TypeStat		*typeStat = NULL;
	int			typeStatCount = 0;
	int			**relationMetadataCount;
	Relation		***relationMetadata;
	oid			**ontologyLookupResult;
	int			*ontologyLookupResultCount;
	int			**ontologyLookupResutMatchedProp;

	IncidentFKs		*links;
	CSlabel			*labels;
        clock_t  	  	curT;
        clock_t         	tmpLastT;



	str		schema = "rdf";
	int		ret;

	tmpLastT = clock();
	TKNZRopen (NULL, &schema);

	// Type
	typeAttributesHistogramCount = initTypeAttributesHistogramCount(typeAttributesCount, freqCSset->numCSadded);
	typeAttributesHistogram = initTypeAttributesHistogram(typeAttributesCount, freqCSset->numCSadded);
#if USE_TYPE_NAMES
	createTypeAttributesHistogram(sbat, si, pi, oi, subjCSMap, freqCSset, csIdFreqIdxMap, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeAttributes, ontmetaBat);
	typeStat = getTypeStats(&typeStatCount, freqCSset->numCSadded, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount);
#else
	(void) sbat;
	(void) si;
	(void) pi;
	(void) oi;
	(void) subjCSMap;
	(void) csIdFreqIdxMap;
#endif

	curT = clock(); 
	//printf (" Labeling: Collecting type attributes histogram took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	// Relation (FK)
	relationMetadataCount = initRelationMetadataCount(freqCSset);
	relationMetadata = initRelationMetadata(relationMetadataCount, csrelSet, num, freqCSset);
	links = initLinks(freqCSset->numCSadded);
#if USE_FK_NAMES
	createLinks(freqCSset, relationMetadata, relationMetadataCount, links);
#endif

	curT = clock(); 
	//printf (" Labeling: Collecting relationship metatdata count took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	// Ontologies
	ontologyLookupResultCount = initOntologyLookupResultCount(freqCSset->numCSadded);
	ontologyLookupResult = initOntologyLookupResult(freqCSset->numCSadded);
	ontologyLookupResutMatchedProp = initOntologyLookupResultMatchedProp(freqCSset->numCSadded);
#if USE_ONTOLOGY_NAMES
	createOntologyLookupResult(ontologyLookupResult, ontologyLookupResutMatchedProp, freqCSset, ontologyLookupResultCount, ontattributes, ontattributesCount, ontmetadata, ontmetadataCount);
	// TODO ont-data have to be freed on shutdown of the database
	// freeOntattributes(ontattributes);
	// freeOntmetadata(ontmetadata);
#else
	(void) ontattributesCount;
	(void) ontattributes;
#endif

	curT = clock(); 
	//printf (" Labeling: Collecting ontology lookup results took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	// Assigning Names
	labels = initLabels(freqCSset);
#if USE_TABLE_NAME
	getAllLabels(labels, freqCSset, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeStat, typeStatCount, ontologyLookupResult, ontologyLookupResutMatchedProp, ontologyLookupResultCount, links, ontmetadata, ontmetadataCount, ontmetaBat, ontclassSet);

	curT = clock(); 
	printf (" Labeling: Get all labels took %f seconds.\n", ((float)(curT - tmpLastT))/CLOCKS_PER_SEC);
	tmpLastT = curT; 		

	if (typeStatCount > 0) free(typeStat);
#endif

	// Collect ontology statistics (tree)
	createOntoUsageTree(ontoUsageTree, freqCSset, ontmetadata, ontmetadataCount, ontmetaBat, labels);

	free(ontologyLookupResultCount);
	freeOntologyLookupResult(ontologyLookupResult, ontologyLookupResutMatchedProp, freqCSset->numCSadded);
	freeTypeAttributesHistogram(typeAttributesHistogram, freqCSset->numCSadded, typeAttributesCount);
	freeTypeAttributesHistogramCount(typeAttributesHistogramCount, freqCSset->numCSadded);
	freeLinks(links, freqCSset->numCSadded);
	freeRelationMetadata(relationMetadata, freqCSset);
	freeRelationMetadataCount(relationMetadataCount, freqCSset->numCSadded);

	TKNZRclose(&ret);

	return labels;
}

/* Merge two lists of candidates.
 * Result: <common name> <ontology candidates CS1> <ontology candidates CS2> <type candidates CS1> <type candidates CS2> <FK candidates CS1> <FK candidates CS2>
 */
static
oid* mergeCandidates(int *candidatesCount, int *candidatesNew, int *candidatesType, int *candidatesOntology, int *candidatesFK, CSlabel cs1, CSlabel cs2, oid commonName) {
	oid	*candidates;
	int	counter = 0;
	int	i;

	(*candidatesCount) = cs1.candidatesCount + cs2.candidatesCount + 1; // +1 for common name
	candidates = GDKmalloc(sizeof(oid) * (*candidatesCount));

	candidates[counter] = commonName;
	counter++;

	// copy "new"
	for (i = 0; i < cs1.candidatesNew; ++i) {
		candidates[counter] = cs1.candidates[i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesNew; ++i) {
		candidates[counter] = cs2.candidates[i];
		counter++;
	}
	(*candidatesNew) = counter;

	// copy "type"
	for (i = 0; i < cs1.candidatesType; ++i) {
		candidates[counter] = cs1.candidates[cs1.candidatesNew + i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesType; ++i) {
		candidates[counter] = cs2.candidates[cs2.candidatesNew + i];
		counter++;
	}
	(*candidatesType) = counter - (*candidatesNew);

	// copy "ontology"
	for (i = 0; i < cs1.candidatesOntology; ++i) {
		candidates[counter] = cs1.candidates[cs1.candidatesNew + cs1.candidatesType + i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesOntology; ++i) {
		candidates[counter] = cs2.candidates[cs2.candidatesNew + cs2.candidatesType + i];
		counter++;
	}
	(*candidatesOntology) = counter - (*candidatesNew) - (*candidatesType);

	// copy "fk"
	for (i = 0; i < cs1.candidatesFK; ++i) {
		candidates[counter] = cs1.candidates[cs1.candidatesNew + cs1.candidatesType + cs1.candidatesOntology + i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesFK; ++i) {
		candidates[counter] = cs2.candidates[cs2.candidatesNew + cs2.candidatesType + cs2.candidatesOntology + i];
		counter++;
	}
	(*candidatesFK) = counter - (*candidatesNew) - (*candidatesType) - (*candidatesOntology);

	return candidates;
}

/* Create labels for merged CS's. Uses rules S1 to S5 (new names!).
 * If no MERGECS is created (subset-superset relation), mergeCSFreqId contains the Id of the superset class.
 * For S1 and S2, parameter 'name' is used to avoid recomputation of CS names
 */
str updateLabel(int ruleNumber, CSset *freqCSset, CSlabel **labels, int newCS, int mergeCSFreqId, int freqCS1, int freqCS2, oid name, int isType, int isOntology, int isFK, oid **ontmetadata, int ontmetadataCount, int *lstFreqId, int numIds){
	int		i;
	int		freqCS1Counter;
	CSlabel		big, small;
	CSlabel		*label;
	CS		cs;	
/*	#if     USE_MULTIWAY_MERGING
	// multiway merging cannot be used here, see below
	int		tmpMaxCoverage; 
	int		tmpFreqId;
	#endif */
	oid		*mergedCandidates = NULL;
	int		candidatesCount, candidatesNew, candidatesType, candidatesOntology, candidatesFK;

	(void) lstFreqId;
	(void) numIds;

	#if	! INFO_WHERE_NAME_FROM
	(void) isType;
	(void) isOntology;
	(void) isFK;
	#endif

	if (newCS) {
		// realloc labels
		*labels = GDKrealloc(*labels, sizeof(CSlabel) * freqCSset->numCSadded);
		if (!(*labels)) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		(*labels)[mergeCSFreqId].name = BUN_NONE; 
		(*labels)[mergeCSFreqId].candidates = NULL;
		(*labels)[mergeCSFreqId].candidatesCount = 0;
		(*labels)[mergeCSFreqId].candidatesNew = 0;
		(*labels)[mergeCSFreqId].candidatesType = 0;
		(*labels)[mergeCSFreqId].candidatesOntology = 0;
		(*labels)[mergeCSFreqId].candidatesFK = 0;
		(*labels)[mergeCSFreqId].hierarchy = NULL;
		(*labels)[mergeCSFreqId].hierarchyCount = 0;
		(*labels)[mergeCSFreqId].numProp = 0;
		(*labels)[mergeCSFreqId].lstProp = NULL;
	}
	label = &(*labels)[mergeCSFreqId];
	cs = freqCSset->items[mergeCSFreqId];

	// copy properties
	if (ruleNumber != S3) {
		if (label->numProp > 0) GDKfree(label->lstProp);
		label->numProp = cs.numProp;
		label->lstProp = (oid *) GDKmalloc(sizeof(oid) * label->numProp);
		if (!label->lstProp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (i = 0; i < label->numProp; ++i) {
			label->lstProp[i] = cs.lstProp[i];
		}
	}

	switch (ruleNumber) {
		case S1: // was: (S1 or S2), now combined
		// use common name
		label->name = name;
		#if	INFO_WHERE_NAME_FROM
		label->isType = isType;
		label->isOntology = isOntology;
		label->isFK = isFK;
		#endif

		#if     USE_MULTIWAY_MERGING
		(void)ontmetadata;
		(void)ontmetadataCount; 
		(void)freqCS2;

		#else
		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesType, &candidatesOntology, &candidatesFK, (*labels)[freqCS1], (*labels)[freqCS2], label->name);
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesType = candidatesType;
		label->candidatesOntology = candidatesOntology;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);

		// hierarchy
		if ((*labels)[freqCS1].name == label->name) {
			// copy hierarchy from CS freqCS1
			label->hierarchyCount = (*labels)[freqCS1].hierarchyCount;
			if (label->hierarchyCount > 0) {
				if (label->hierarchy != NULL) GDKfree(label->hierarchy);
				label->hierarchy = (oid *) GDKmalloc(sizeof(oid) * label->hierarchyCount);
				if (!label->hierarchy) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				for (i = 0; i < label->hierarchyCount; ++i) {
					label->hierarchy[i] = (*labels)[freqCS1].hierarchy[i];
				}
			}
		} else if ((*labels)[freqCS2].name == label->name) {
			// copy hierarchy from CS freqCS2
			label->hierarchyCount = (*labels)[freqCS2].hierarchyCount;
			if (label->hierarchyCount > 0) {
				label->hierarchy = (oid *) GDKmalloc(sizeof(oid) * label->hierarchyCount);
				if (!label->hierarchy) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				for (i = 0; i < label->hierarchyCount; ++i) {
					label->hierarchy[i] = (*labels)[freqCS2].hierarchy[i];
				}
			}
		} else {
			// no top 1 name, no hierarchy available
			label->hierarchy = getOntoHierarchy(name, &(label->hierarchyCount), ontmetadata, ontmetadataCount);
		}
		#endif

		break;

		case S2:
		// use common ancestor
		label->name = name;
		#if	INFO_WHERE_NAME_FROM
		label->isType = isType;
		label->isOntology = isOntology;
		label->isFK = isFK;
		#endif

		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesType, &candidatesOntology, &candidatesFK, (*labels)[freqCS1], (*labels)[freqCS2], label->name);
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesType = candidatesType;
		label->candidatesOntology = candidatesOntology;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);

		// hierarchy
		freqCS1Counter = (*labels)[freqCS1].hierarchyCount - 1;
		while (TRUE) {
			if ((*labels)[freqCS1].hierarchy[freqCS1Counter] == label->name)
				break;
			freqCS1Counter--;
		}
		label->hierarchyCount = (*labels)[freqCS1].hierarchyCount - freqCS1Counter;
		if (label->hierarchyCount > 0) {
			label->hierarchy = (oid *) GDKmalloc(sizeof(oid) * label->hierarchyCount);
			if (!label->hierarchy) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			for (i = 0; i < label->hierarchyCount; ++i) {
				label->hierarchy[i] = (*labels)[freqCS1].hierarchy[freqCS1Counter + i];
			}
		}

		break;

		case S3:
		// subset-superset relation

		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesType, &candidatesOntology, &candidatesFK, (*labels)[freqCS1], (*labels)[freqCS2], label->name); // freqCS1 is superCS, freqCS2 is subCS
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesType = candidatesType;
		label->candidatesOntology = candidatesOntology;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);
		if (label->name == BUN_NONE && label->candidates[0] != BUN_NONE) {
			// superCS had no name before, but subCS adds candidates
			label->name = label->candidates[0];
			#if	INFO_WHERE_NAME_FROM
			label->isType = (*labels)[freqCS2].isType;
			label->isOntology = (*labels)[freqCS2].isOntology;
			label->isFK = (*labels)[freqCS2].isFK;
			#endif
		} // else: old name and isType/isOntology/isFK remain valid

		// hierarchy already set
		// properties already set

		break;

		case S4: // FALLTHROUGH
		case S5:
/*		#if	USE_MULTIWAY_MERGING
		// multiwaymerging cannot be used because 'small' is not set, but needed for mergeCandidates()
		tmpMaxCoverage = 0; 
		tmpFreqId = 0;
		for (i = 0; i < numIds; i++){
			if (freqCSset->items[lstFreqId[i]].coverage > tmpMaxCoverage){
				tmpFreqId = lstFreqId[i];
				tmpMaxCoverage = freqCSset->items[lstFreqId[i]].coverage;
			}
		}
		big = (*labels)[tmpFreqId];

		#else */
		// use label of biggest CS (higher coverage value)
		if (freqCSset->items[freqCS1].coverage > freqCSset->items[freqCS2].coverage) {
			big = (*labels)[freqCS1];
			small = (*labels)[freqCS2];
		} else {
			big = (*labels)[freqCS2];
			small = (*labels)[freqCS1];
		}
//		#endif
		label->name = big.name;
		#if	INFO_WHERE_NAME_FROM
		label->isType = big.isType;
		label->isOntology = big.isOntology;
		label->isFK = big.isFK;
		#endif

		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesType, &candidatesOntology, &candidatesFK, big, small, label->name);
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesType = candidatesType;
		label->candidatesOntology = candidatesOntology;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);
		if (label->name == BUN_NONE && label->candidates[0] != BUN_NONE) {
			// no name yet, use name of small table
			label->name = label->candidates[0];
			#if	INFO_WHERE_NAME_FROM
			label->isType = small.isType;
			label->isOntology = small.isOntology;
			label->isFK = small.isFK;
			#endif
		}

		// hierarchy
		label->hierarchyCount = big.hierarchyCount;
		if (label->hierarchyCount > 0) {
			if (label->hierarchy != NULL) GDKfree(label->hierarchy);
			label->hierarchy = (oid *) GDKmalloc(sizeof(oid) * label->hierarchyCount);
			if (!label->hierarchy) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			for (i = 0; i < label->hierarchyCount; ++i) {
				label->hierarchy[i] = big.hierarchy[i];
			}
		}
		break;

		default:
		// error
		return "ERROR: ruleNumber must be S1, S2, S3, S4, or S5";
	}

	return MAL_SUCCEED;
}

void freeLabels(CSlabel* labels, CSset* freqCSset) {
	int		i;

	//for (i = 0; i < freqCSset->numOrigFreqCS; ++i) { // do not use numCSadded because of additional mergeCS
	for (i = 0; i < freqCSset->numCSadded; ++i) {	
		if (labels[i].numProp > 0)
			GDKfree(labels[i].lstProp);

		if (labels[i].hierarchyCount > 0)
			GDKfree(labels[i].hierarchy);

		if (labels[i].candidatesCount > 0)
			GDKfree(labels[i].candidates);
	}
	GDKfree(labels);
}

void exportLabels(CSset* freqCSset, CSrel* csRelFinalFKs, int freqThreshold, BATiter mapi, BAT *mbat, CStableStat* cstablestat, CSPropTypes *csPropTypes, int numTables, int* mTblIdxFreqIdxMapping, int* csTblIdxMapping) {
	str             schema = "rdf";
	int             ret;
	
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		fprintf(stderr, "ERROR: Couldn't open tokenizer!\n");
	}

	// Print and Export
	printf("exportLabels: printUML \n"); 
	printUML2(cstablestat, csPropTypes, freqThreshold, csRelFinalFKs, mapi, mbat, numTables, mTblIdxFreqIdxMapping, csTblIdxMapping, freqCSset);
	printf("exportLabels: Done \n"); 

	TKNZRclose(&ret);
}

void freeOntoUsageTree(OntoUsageNode* tree) {
	int i;

	if (tree->numChildren == 0) {
		// leaf node
		free(tree);
		return;
	}

	// inner node
	for (i = 0; i < tree->numChildren; ++i) {
		freeOntoUsageTree(tree->lstChildren[i]);
	}
	free(tree->lstChildren);
	free(tree);
}
