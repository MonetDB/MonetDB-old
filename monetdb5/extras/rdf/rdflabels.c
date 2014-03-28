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

// list of known ontologies
int ontologyCount = 73;
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

/* Calculate frequency per foreign key relationship. */
static
Relation*** initRelationMetadata2(int** relationMetadataCount, CSrel* csRelBetweenMergeFreqSet, CSset* freqCSset) {
	int		i, j, k;
	Relation***	relationMetadata;

	relationMetadata = (Relation ***) malloc(sizeof(Relation **) * freqCSset->numCSadded);
	if (!relationMetadata) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < freqCSset->numCSadded; ++i) { // CS
		CS cs;
		if (i == -1) continue; // ignore
		cs = (CS) freqCSset->items[i];
		relationMetadata[i] = (Relation **) malloc (sizeof(Relation *) * cs.numProp);
		if (!relationMetadata[i]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < cs.numProp; ++j) { // propNo in CS order
			int sum = 0;
			relationMetadataCount[i][j] = 0;
			relationMetadata[i][j] = NULL;
			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) { // propNo in CSrel

				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == cs.lstProp[j]) {
					int toId = csRelBetweenMergeFreqSet[i].lstRefFreqIdx[k];
					if (toId == -1) continue; // ignore
					relationMetadataCount[i][j] += 1;

					// alloc/realloc
					if (relationMetadataCount[i][j] == 1) {
						// alloc
						relationMetadata[i][j] = (Relation *) malloc (sizeof(Relation));
						if (!relationMetadata[i][j]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
						relationMetadata[i][j][0].to = toId;
						relationMetadata[i][j][0].from = i;
						relationMetadata[i][j][0].freq = csRelBetweenMergeFreqSet[i].lstCnt[k];
						relationMetadata[i][j][0].percent = -1;
					} else {
						// realloc
						relationMetadata[i][j] = (Relation *) realloc(relationMetadata[i][j], sizeof(Relation) * relationMetadataCount[i][j]);
						if (!relationMetadata[i][j]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						relationMetadata[i][j][relationMetadataCount[i][j] - 1].to = toId;
						relationMetadata[i][j][relationMetadataCount[i][j] - 1].from = i;
						relationMetadata[i][j][relationMetadataCount[i][j] - 1].freq = csRelBetweenMergeFreqSet[i].lstCnt[k];
						relationMetadata[i][j][relationMetadataCount[i][j] - 1].percent = -1;
					}
				}
			}

			// get total count of values
			for (k = 0; k < relationMetadataCount[i][j]; ++k) {
				sum += relationMetadata[i][j][k].freq;
			}
			// assign percentage values for every value
			for (k = 0; k < relationMetadataCount[i][j]; ++k) {
				relationMetadata[i][j][k].percent = (int) (100.0 * relationMetadata[i][j][k].freq / sum + 0.5);
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

/* Modifies the parameter! */
/* Replaces colons, quotes, spaces, and dashes with underscores. All lowercase. */
static
void escapeURIforSQL(char* s) {
	int i;

	for (i = 0; i < (int) strlen(s); ++i) {
		if (s[i] == ':' || s[i] == '"' || s[i] == ' ' || s[i] == '-' || s[i] == '<' || s[i] == '>' || s[i] == '/' || s[i] == '(' || s[i] == ')' || s[i] == '.' || s[i] == '%') s[i] = '_';
		s[i] = tolower(s[i]);
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

/* Create SQL CREATE TABLE statements including foreign keys. */
static
void convertToSQL(CSset *freqCSset, Relation*** relationMetadata, int** relationMetadataCount, CSlabel* labels, int freqThreshold) {
	// file i/o
	FILE		*fout;
	char		filename[20], tmp[10];

	// looping
	int		i, j, k;

	strcpy(filename, "CS");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".sql");

	fout = fopen(filename, "wt");

	// create statement for every table
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		str labelStr, tmpStr;

		if (!isCSTable(freqCSset->items[i])) continue; // ignore

		if (labels[i].name == BUN_NONE) {
			fprintf(fout, "CREATE TABLE %s_"BUNFMT" (\nsubject VARCHAR(10) PRIMARY KEY,\n", "DUMMY", freqCSset->items[i].csId); // TODO underscores?
		} else {
#if USE_SHORT_NAMES
			str labelStrShort = NULL;
#endif
			takeOid(labels[i].name, &labelStr);
#if USE_SHORT_NAMES
			getPropNameShort(&labelStrShort, labelStr);
			tmpStr = (str) malloc(sizeof(char) * (strlen(labelStrShort) + 1));
			if (!tmpStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr, labelStrShort);
			escapeURIforSQL(tmpStr);
			fprintf(fout, "CREATE TABLE %s_"BUNFMT" (\nsubject VARCHAR(10) PRIMARY KEY,\n", tmpStr, freqCSset->items[i].csId); // TODO underscores?
			free(tmpStr);
			GDKfree(labelStrShort);
			GDKfree(labelStr);
#else
			tmpStr = (str) malloc(sizeof(char) * (strlen(labelStr) + 1));
			if (!tmpStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr, labelStr);
			escapeURIforSQL(tmpStr);
			fprintf(fout, "CREATE TABLE %s_"BUNFMT" (\nsubject VARCHAR(10) PRIMARY KEY,\n", tmpStr, freqCSset->items[i].csId); // TODO underscores?
			free(tmpStr);
			GDKfree(labelStr);
#endif
		}
		for (j = 0; j < labels[i].numProp; ++j) {
			str propStr, tmpStr2;
#if USE_SHORT_NAMES
			str propStrShort = NULL;
#endif
			takeOid(labels[i].lstProp[j], &propStr);

#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			tmpStr2 = (char *) malloc(sizeof(char) * (strlen(propStrShort) + 1));
			if (!tmpStr2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr2, propStrShort);
			escapeURIforSQL(tmpStr2);

			if (j + 1 < labels[i].numProp) {
				fprintf(fout, "%s_%d BOOLEAN,\n", tmpStr2, j);
			} else {
				// last column
				fprintf(fout, "%s_%d BOOLEAN\n", tmpStr2, j);
			}
			free(tmpStr2);
			GDKfree(propStrShort);
			GDKfree(propStr); 
#else
			tmpStr2 = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!tmpStr2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr2, propStr);
			escapeURIforSQL(tmpStr2);

			if (j + 1 < labels[i].numProp) {
				fprintf(fout, "%s_%d BOOLEAN,\n", tmpStr2, j);
			} else {
				// last column
				fprintf(fout, "%s_%d BOOLEAN\n", tmpStr2, j);
			}
			free(tmpStr2);
			GDKfree(propStr); 
#endif
		}
		fprintf(fout, ");\n\n");
	}

	// add foreign key columns and add foreign keys
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (!isCSTable(freqCSset->items[i])) continue; // ignore

		for (j = 0; j < labels[i].numProp; ++j) {
			str propStr, tmpStr2;
#if USE_SHORT_NAMES
			str propStrShort = NULL;
#endif
			int refCounter = 0;

			takeOid(labels[i].lstProp[j], &propStr);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			tmpStr2 = (str) malloc(sizeof(char) * (strlen(propStrShort) + 1));
			if (!tmpStr2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr2, propStrShort);
			escapeURIforSQL(tmpStr2);
			GDKfree(propStrShort);
			GDKfree(propStr);
#else
			tmpStr2 = (str) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!tmpStr2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr2, propStr);
			escapeURIforSQL(tmpStr2);
			GDKfree(propStr);
#endif

			for (k = 0; k < relationMetadataCount[i][j]; ++k) {
				int from, to;
				str tmpStrFrom, tmpStrTo;



				if (relationMetadata[i][j][k].percent < FK_FREQ_THRESHOLD) continue; // foreign key is not frequent enough

				from = relationMetadata[i][j][k].from;
				to = relationMetadata[i][j][k].to;

				// get "from" and "to" table names
				if (labels[from].name == BUN_NONE) {
					tmpStrFrom = (str) malloc(sizeof(char) * 6);
					if (!tmpStrFrom) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(tmpStrFrom, "DUMMY");
				} else {
					str labelStrFrom;
#if USE_SHORT_NAMES
					str labelStrFromShort;
#endif
					takeOid(labels[from].name, &labelStrFrom);
#if USE_SHORT_NAMES
					getPropNameShort(&labelStrFromShort, labelStrFrom);
					tmpStrFrom = (str) malloc(sizeof(char) * (strlen(labelStrFromShort) + 1));
					if (!tmpStrFrom) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(tmpStrFrom, labelStrFromShort);
					escapeURIforSQL(tmpStrFrom);
					GDKfree(labelStrFromShort);
					GDKfree(labelStrFrom);
#else
					tmpStrFrom = (str) malloc(sizeof(char) * (strlen(labelStrFrom) + 1));
					if (!tmpStrFrom) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(tmpStrFrom, labelStrFrom);
					escapeURIforSQL(tmpStrFrom);
					GDKfree(labelStrFrom);
#endif
				}

				if (labels[to].name == BUN_NONE) {
					tmpStrTo = (str) malloc(sizeof(char) * 6);
					if (!tmpStrTo) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(tmpStrTo, "DUMMY");
				} else {
					str labelStrTo;
#if USE_SHORT_NAMES
					str labelStrToShort;
#endif
					takeOid(labels[to].name, &labelStrTo);
#if USE_SHORT_NAMES
					getPropNameShort(&labelStrToShort, labelStrTo);
					tmpStrTo = (str) malloc(sizeof(char) * (strlen(labelStrToShort) + 1));
					if (!tmpStrTo) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(tmpStrTo, labelStrToShort);
					escapeURIforSQL(tmpStrTo);
					GDKfree(labelStrToShort);
					GDKfree(labelStrTo);
#else
					tmpStrTo = (str) malloc(sizeof(char) * (strlen(labelStrTo) + 1));
					if (!tmpStrTo) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(tmpStrTo, labelStrTo);
					escapeURIforSQL(tmpStrTo);
					GDKfree(labelStrTo);
#endif
				}

				fprintf(fout, "ALTER TABLE %s_"BUNFMT" ADD COLUMN %s_%d_%d VARCHAR(10);\n", tmpStrFrom, freqCSset->items[from].csId, tmpStr2, j, refCounter);
				fprintf(fout, "ALTER TABLE %s_"BUNFMT" ADD FOREIGN KEY (%s_%d_%d) REFERENCES %s_"BUNFMT"(subject);\n\n", tmpStrFrom, freqCSset->items[from].csId, tmpStr2, j, refCounter, tmpStrTo, freqCSset->items[to].csId);
				refCounter += 1;
				free(tmpStrFrom);
				free(tmpStrTo);
			}
			free(tmpStr2);
		}
	}

	fclose(fout);
}

static
void createSQLMetadata(CSset* freqCSset, CSrel* csRelBetweenMergeFreqSet, CSlabel* labels, int*  mTblIdxFreqIdxMapping,int* mfreqIdxTblIdxMapping,int numTables) {
	int	**matrix = NULL; // matrix[from][to] frequency
	int	i, j, k;
	FILE	*fout;
	int	tblfrom, tblto;

	(void) mTblIdxFreqIdxMapping; 
	(void) mfreqIdxTblIdxMapping; 
	(void) numTables; 
	// init
	matrix = (int **) malloc(sizeof(int *) * numTables);
	if (!matrix) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

 	for (i = 0; i < numTables; ++i) {
		matrix[i] = (int *) malloc(sizeof(int) * numTables);
		if (!matrix) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

		for (j = 0; j < numTables; ++j) {
			matrix[i][j] = 0;
		}
	}

	// set values
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];

		if (!isCSTable(cs)) continue; // ignore
		if (csRelBetweenMergeFreqSet[i].numRef == 0) continue; 

		for (j = 0; j < cs.numProp; ++j) { // propNo in CS order
			// check foreign key frequency
			int sum = 0;
			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) {
				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == cs.lstProp[j]) {
					sum += csRelBetweenMergeFreqSet[i].lstCnt[k];
				}
			}

			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) { // propNo in CSrel
				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == cs.lstProp[j]) {
					int toId = csRelBetweenMergeFreqSet[i].lstRefFreqIdx[k];
					if (toId == -1) continue; // ignore
					if (i == toId) continue; // ignore self references
					if (!isCSTable(freqCSset->items[toId])) continue; 
					if ((int) (100.0 * csRelBetweenMergeFreqSet[i].lstCnt[k] / sum + 0.5) < FK_FREQ_THRESHOLD) continue; // foreign key is not frequent enough
					tblfrom = mfreqIdxTblIdxMapping[i]; 
					tblto = mfreqIdxTblIdxMapping[toId];
					matrix[tblfrom][tblto] += csRelBetweenMergeFreqSet[i].lstCnt[k]; // multiple links from 'i' to 'toId'? add the frequencies
				}
			}
		}
	}

	// store matrix as csv
	fout = fopen("adjacencyList.csv", "wt");
	for (i = 0; i < numTables; ++i) {
		for (j = 0; j < numTables; ++j) {
			if (matrix[i][j]) {
				fprintf(fout, "%d,%d,%d\n", mTblIdxFreqIdxMapping[i], mTblIdxFreqIdxMapping[j], matrix[i][j]);
			}
		}
	}
	fclose(fout);

	// print id -> table name
	fout = fopen("tableIdFreq.csv", "wt");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (!isCSTable(freqCSset->items[i])) continue; // ignore

		if (labels[i].name == BUN_NONE) {
			fprintf(fout, "%d,\"%s_"BUNFMT"\",%d\n", i, "DUMMY", freqCSset->items[i].csId, freqCSset->items[i].support); // TODO underscores?
		} else {
			str labelStr, tmpStr;
#if USE_SHORT_NAMES
			str labelStrShort;
#endif

			takeOid(labels[i].name, &labelStr);

#if USE_SHORT_NAMES
			getPropNameShort(&labelStrShort, labelStr);
			tmpStr = (str) GDKmalloc(sizeof(char) * (strlen(labelStrShort) + 1));
			if (!tmpStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr, labelStrShort);
			escapeURIforSQL(tmpStr);
			GDKfree(labelStrShort);
			GDKfree(labelStr); 
#else
			tmpStr = (str) GDKmalloc(sizeof(char) * (strlen(labelStr) + 1));
			if (!tmpStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(tmpStr, labelStr);
			escapeURIforSQL(tmpStr);
#endif

			fprintf(fout, "%d,\"%s_"BUNFMT"\",%d\n", i, tmpStr, freqCSset->items[i].csId, freqCSset->items[i].support); // TODO underscores?
			GDKfree(tmpStr);
		}
	}
	fclose(fout);

	fout = fopen("CSmetadata.sql", "wt");
	fprintf(fout, "CREATE TABLE table_id_freq (id INTEGER, name VARCHAR(100), frequency INTEGER);\n");
	fprintf(fout, "CREATE TABLE adjacency_list (from_id INTEGER, to_id INTEGER, frequency INTEGER);\n");
	fprintf(fout, "COPY INTO table_id_freq from '/export/scratch2/linnea/dbfarm/test/tableIdFreq.csv' USING DELIMITERS ',','\\n','\"';\n");
	fprintf(fout, "COPY INTO adjacency_list from '/export/scratch2/linnea/dbfarm/test/adjacencyList.csv' USING DELIMITERS ',','\\n','\"';");
	fclose(fout);

	for (i = 0; i < numTables; ++i) {
		free(matrix[i]);
	}
	free(matrix);
}

/* Simple representation of the final labels for tables and attributes. */
static
void printTxt(CSset* freqCSset, CSlabel* labels, int freqThreshold) {
	FILE 		*fout;
	char		filename[20], tmp[10];
	int		i, j;

	strcpy(filename, "labels");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".txt");

	fout = fopen(filename, "wt");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		str labelStr;
#if USE_SHORT_NAMES
		str labelStrShort = NULL;
#endif

		if (!isCSTable(freqCSset->items[i])) continue; // ignore

		if (labels[i].name == BUN_NONE) {
			fprintf(fout, "%s (CS "BUNFMT"): ", "DUMMY", freqCSset->items[i].csId);
		} else {
			takeOid(labels[i].name, &labelStr);
#if USE_SHORT_NAMES
			getPropNameShort(&labelStrShort, labelStr);
			fprintf(fout, "%s (CS "BUNFMT"): ", labelStrShort, freqCSset->items[i].csId);
			GDKfree(labelStrShort);
			GDKfree(labelStr);
#else
			fprintf(fout, "%s (CS "BUNFMT"): ", labelStr, freqCSset->items[i].csId);
			GDKfree(labelStr);
#endif
		}
		for (j = 0; j < labels[i].numProp; ++j) {
			str propStr;
#if USE_SHORT_NAMES
			str propStrShort = NULL;
#endif
			takeOid(labels[i].lstProp[j], &propStr);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			if (j + 1 < labels[i].numProp) fprintf(fout, "%s, ", propStrShort);
			else fprintf(fout, "%s\n", propStrShort);
			GDKfree(propStrShort);
			GDKfree(propStr);
#else
			if (j + 1 < labels[i].numProp) fprintf(fout, "%s, ", propStr);
			else fprintf(fout, "%s\n", propStr);
			GDKfree(propStr);
#endif
		}
	}
	fclose(fout);
}

#if USE_TYPE_NAMES
static
int compareTypeAttributesFreqs (const void * a, const void * b) {
  return ( (*(TypeAttributesFreq*)b).freq - (*(TypeAttributesFreq*)a).freq ); // sort descending
}
#endif

#if USE_TYPE_NAMES
/* Add type values to the histogram. Values that are not present in the hierarchy tree built from the ontologies are NOT added to the histogram. */
static
void insertValuesIntoTypeAttributesHistogram(oid* typeList, int typeListLength, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, int csFreqIdx, int type, BAT *ontmetaBat) {
	int		i, j;
	int		fit;

	for (i = 0; i < typeListLength; ++i) {
		BUN pos = BUNfnd(BATmirror(ontmetaBat), &typeList[i]);
		if (pos == BUN_NONE) continue; // no ontology information, ignore

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

	// histogram
	int		i, j, k;

	oid 		*typeAttributesOids = malloc(sizeof(oid) * typeAttributesCount);

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
						insertValuesIntoTypeAttributesHistogram(typeValues, typeValuesSize, typeAttributesHistogram, typeAttributesHistogramCount, csFreqIdx, curT, ontmetaBat);
						typeValuesSize = 0; // reset
					}
					curS = *sbt;
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
		insertValuesIntoTypeAttributesHistogram(typeValues, typeValuesSize, typeAttributesHistogram, typeAttributesHistogramCount, csFreqIdx, curT, ontmetaBat);
	}

	GDKfree(typeValues);

	// sort descending by frequency
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			qsort(typeAttributesHistogram[i][j], typeAttributesHistogramCount[i][j], sizeof(TypeAttributesFreq), compareTypeAttributesFreqs);
		}
	}

	// assign percentage
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			// assign percentage values for every value
			for (k = 0; k < typeAttributesHistogramCount[i][j]; ++k) {
				typeAttributesHistogram[i][j][k].percent = (int) (100.0 * typeAttributesHistogram[i][j][k].freq / freqCSset->items[i].support + 0.5);

			}
		}
	}

	free(typeAttributesOids);
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
str** findOntologies(CS cs, int *propOntologiesCount, oid*** propOntologiesOids) {
	int		i, j, k;
	str		**propOntologies = NULL;

	propOntologies = (str **) malloc(sizeof(str *) * ontologyCount);
	(*propOntologiesOids) = (oid **) malloc(sizeof(str *) * ontologyCount);
	if (!propOntologies || !(*propOntologiesOids)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < ontologyCount; ++i) {
		propOntologies[i] = NULL;
		(*propOntologiesOids)[i] = NULL;
	}

	for (i = 0; i < ontologyCount; ++i) {
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
					propOntologies[i] = realloc(propOntologies[i], sizeof(str) * (propOntologiesCount[i] + 1));
					(*propOntologiesOids)[i] = realloc((*propOntologiesOids)[i], sizeof(str) * (propOntologiesCount[i] + 1));
					if (!propOntologies[i] || !(*propOntologiesOids)[i]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					propOntologies[i][propOntologiesCount[i]] = tmpStr;
					(*propOntologiesOids)[i][propOntologiesCount[i]] = cs.lstProp[j];
					propOntologiesCount[i] += 1;
				}
			}
			for (k = 0; k < length; ++k) {
				free(tokenizedUri[k]);
			}
			free(tokenizedUri);

			GDKfree(tmpStr);
		}
	}
	return propOntologies;
}
#endif

#if USE_ONTOLOGY_NAMES
static
int compareOntologyCandidates (const void * a, const void * b) {
	float f1 = (*(ClassStat*)a).tfidfs;
	float f2 = (*(ClassStat*)b).tfidfs;

	if (f1 > f2) return -1;
	if (f2 > f1) return 1;
	return 0; // sort descending
}
#endif

#if USE_ONTOLOGY_NAMES
/* For one CS: Calculate the ontology classes that are similar (tfidf) to the list of attributes. */
static
oid* getOntologyCandidates(oid** ontattributes, int ontattributesCount, oid** ontmetadata, int ontmetadataCount, int *resultCount, oid **listOids, int *listCount, int listNum, PropStat *propStat) {
	int		i, j, k, l;
	oid		*result = NULL;

	for (i = 0; i < listNum; ++i) {
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
		for (j = 0; j < ontattributesCount; ++j) {
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
			bun = BUNfnd(BATmirror(propStat->pBat), (ptr) &p);
			if (bun == BUN_NONE) continue; // property does not belong to an ontology class and therefore has no tfidfs score
			for (k = 0; k < candidatesCount[j]; ++k) { // for each candidate
				// search for this class
				int found = 0;
				for (l = 0; l < num; ++l) {
					if (candidates[j][k] == classStat[l].ontoClass) {
						// add tdidf^2 to sum
						classStat[l].tfidfs += (propStat->tfidfs[bun] * propStat->tfidfs[bun]);
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
			bun = BUNfnd(BATmirror(propStat->pBat), (ptr) &listOids[i][j]);
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
			//printf("    TFIDF score at %d is: %f  \n",k, classStat[k].tfidfs);
			if (classStat[k].tfidfs < ONTOLOGY_FREQ_THRESHOLD) break; // values not frequent enough (list is sorted by tfidfs)
			for (j = 0; j < ontmetadataCount && (found == 0); ++j) {
				oid muri = ontmetadata[0][j];
				oid msuper = ontmetadata[1][j];
				if (classStat[k].ontoClass == muri) {
					if (msuper == BUN_NONE) {
						// muristr is a candidate! (because it has no superclass)
						result = realloc(result, sizeof(oid) * ((*resultCount) + 1));
						if (!result) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						result[*resultCount] = muri;
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

#if USE_ONTOLOGY_NAMES
static
PropStat* initPropStat(void) {
	PropStat *propStat = (PropStat *) malloc(sizeof(PropStat));
	propStat->pBat = BATnew(TYPE_void, TYPE_oid, INIT_PROP_NUM);

	BATseqbase(propStat->pBat, 0);

	if (propStat->pBat == NULL) {
		return NULL;
	}

	(void)BATprepareHash(BATmirror(propStat->pBat));
	if (!(propStat->pBat->T->hash)) {
		return NULL;
	}

	propStat->freqs = (int*) malloc(sizeof(int) * INIT_PROP_NUM);
	if (!propStat->freqs) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	propStat->tfidfs = (float*) malloc(sizeof(float) * INIT_PROP_NUM);
	if (!propStat->tfidfs) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	propStat->numAdded = 0;
	propStat->numAllocation = INIT_PROP_NUM;

	return propStat;
}
#endif

#if USE_ONTOLOGY_NAMES
/* Copied from Duc's code. */
/*
static
void createPropStatistics(PropStat* propStat, int numMaxCSs, CSset* freqCSset) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS)freqCSset->items[i];
		for (j = 0; j < cs.numProp; ++j) {
			// add prop to propStat
			BUN	bun = BUNfnd(BATmirror(propStat->pBat), (ptr) &cs.lstProp[j]);
			if (bun == BUN_NONE) {
				   if (propStat->pBat->T->hash && BATcount(propStat->pBat) > 4 * propStat->pBat->T->hash->mask) {
					HASHdestroy(propStat->pBat);
					BAThash(BATmirror(propStat->pBat), 2*BATcount(propStat->pBat));
				}

				propStat->pBat = BUNappend(propStat->pBat, &cs.lstProp[j], TRUE);

				if (propStat->numAdded == propStat->numAllocation) {
					propStat->numAllocation += INIT_PROP_NUM;

					propStat->freqs = realloc(propStat->freqs, ((propStat->numAllocation) * sizeof(int)));
					propStat->tfidfs = realloc(propStat->tfidfs, ((propStat->numAllocation) * sizeof(float)));
					if (!propStat->freqs || !propStat->tfidfs) {fprintf(stderr, "ERROR: Couldn't realloc memory!\n");}
				}
				propStat->freqs[propStat->numAdded] = 1;
				propStat->numAdded++;
			} else {
				propStat->freqs[bun]++;
			}
		}
	}

	for (i = 0; i < propStat->numAdded; ++i) {
		propStat->tfidfs[i] = log(((float)numMaxCSs) / (1 + propStat->freqs[i]));
	}
}
*/
//[DUC] Create propstat for ontology only 
static
void createPropStatistics(PropStat* propStat, oid** ontattributes, int ontattributesCount) {
	int		i;

	for (i = 0; i < ontattributesCount; ++i) {
		oid attr = ontattributes[1][i];
		// add prop to propStat
		BUN	bun = BUNfnd(BATmirror(propStat->pBat), (ptr) &attr);
		if (bun == BUN_NONE) {
			if (propStat->pBat->T->hash && BATcount(propStat->pBat) > 4 * propStat->pBat->T->hash->mask) {
				HASHdestroy(propStat->pBat);
				BAThash(BATmirror(propStat->pBat), 2*BATcount(propStat->pBat));
			}

			propStat->pBat = BUNappend(propStat->pBat, &attr, TRUE);

			if (propStat->numAdded == propStat->numAllocation) {
				propStat->numAllocation += INIT_PROP_NUM;

				propStat->freqs = realloc(propStat->freqs, ((propStat->numAllocation) * sizeof(int)));
				propStat->tfidfs = realloc(propStat->tfidfs, ((propStat->numAllocation) * sizeof(float)));
				if (!propStat->freqs || !propStat->tfidfs) {fprintf(stderr, "ERROR: Couldn't realloc memory!\n");}
			}
			propStat->freqs[propStat->numAdded] = 1;
			propStat->numAdded++;
		} else {
			propStat->freqs[bun]++;
		}
	}

	for (i = 0; i < propStat->numAdded; ++i) {
		propStat->tfidfs[i] = log(((float)ontattributesCount) / (1 + propStat->freqs[i]));
	}
}

//... [DUC]
#endif

#if USE_ONTOLOGY_NAMES
static
void freePropStat(PropStat *propStat) {
	BBPreclaim(propStat->pBat);
	free(propStat->freqs);
	free(propStat->tfidfs);

	free(propStat);
}
#endif

#if USE_ONTOLOGY_NAMES
/* For all CS: Calculate the ontology classes that are similar (tfidf) to the list of attributes. */
static
void createOntologyLookupResult(oid** result, CSset* freqCSset, int* resultCount, oid** ontattributes, int ontattributesCount, oid** ontmetadata, int ontmetadataCount) {
	int		i, j;
	PropStat	*propStat;

	propStat = initPropStat();

	//[DUC] Change the function for getting propStat. Use ontattributes for the propStat. 
	// Not the properties from freqCS
	//createPropStatistics(propStat, freqCSset->numCSadded, freqCSset);
	createPropStatistics(propStat, ontattributes, ontattributesCount);

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS		cs;
		str		**propOntologies = NULL;
		oid		**propOntologiesOids = NULL;
		int		*propOntologiesCount = NULL;

		cs = (CS) freqCSset->items[i];

		// order properties by ontologies
		propOntologiesCount = (int *) malloc(sizeof(int) * ontologyCount);
		if (!propOntologiesCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < ontologyCount; ++j) {
			propOntologiesCount[j] = 0;
		}
		
		//printf("Get ontology for FreqId %d. Orignal numProp = %d \n", i, cs.numProp);

		propOntologies = findOntologies(cs, propOntologiesCount, &propOntologiesOids);

		/*
		printf("Prop ontologies count. \n");
		for (j = 0; j < ontologyCount; ++j) {
			if (propOntologiesCount[j] > 0)
				printf("    %d props in ontology %d \n ", propOntologiesCount[j], j);
		}
		*/

		// get class names
		resultCount[i] = 0;
		
		result[i] = getOntologyCandidates(ontattributes, ontattributesCount, ontmetadata, ontmetadataCount, &(resultCount[i]), propOntologiesOids, propOntologiesCount, ontologyCount, propStat);

		for (j = 0; j < ontologyCount; ++j) {
			free(propOntologies[j]);
			free(propOntologiesOids[j]);
		}
		free(propOntologies);
		free(propOntologiesOids);
		free(propOntologiesCount);
	}
	freePropStat(propStat);
}
#endif

/* Print the dot code to draw an UML-like diagram. Call:   dot -Tpdf -O <filename>   to create <filename>.pdf */
/*
static
void printUML(CSset *freqCSset, int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, str** result, int* resultCount, IncidentFKs* links, CSlabel* labels, Relation*** relationMetadata, int** relationMetadataCount, int freqThreshold) {
	str		propStr, tmpStr;

#if SHOW_CANDIDATES
	char*           resultStr = NULL;
	unsigned int    resultStrSize = 100;
	int		found;
#endif

	int 		i, j, k;
	FILE 		*fout;
	char 		filename[20], tmp[10];

	strcpy(filename, "CSmax");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".dot");

	fout = fopen(filename, "wt");

	// header
	fprintf(fout, "digraph g {\n");
	fprintf(fout, "graph[ratio=\"compress\"];\n");
	fprintf(fout, "node [shape=\"none\"];\n\n");
	fprintf(fout, "legend [label = <<TABLE BGCOLOR=\"lightgray\" BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\"><TR><TD><B>Colors:</B></TD></TR>\n");
	fprintf(fout, "<TR><TD><FONT COLOR=\"red\"><B>Ontology Classes</B></FONT></TD></TR>\n");
	fprintf(fout, "<TR><TD><FONT COLOR=\"blue\"><B>Type Values</B></FONT></TD></TR>\n");
	fprintf(fout, "<TR><TD><FONT COLOR=\"green\"><B>Foreign Keys</B></FONT></TD></TR>\n");
	fprintf(fout, "</TABLE>>];\n\n");

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];

#if SHOW_CANDIDATES
		// DATA SOURCES
		resultStr = (char *) malloc(sizeof(char) * resultStrSize);
		if (!resultStr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(resultStr, "\0");
#endif

#if SHOW_CANDIDATES
		// ontologies (red)
		if (resultCount[i] > 0) {
			// resize resultStr ?
			while (strlen(resultStr) + strlen("<FONT color=\"red\">") + 1 > resultStrSize) { // + 1 for \0
				resultStrSize *= 2;
				resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
				if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			strcat(resultStr, "<FONT color=\"red\">");
			for (j = 0; j < resultCount[i]; ++j) {
#if USE_SHORT_NAMES
				char *resultShort = NULL;
				getPropNameShort(&resultShort, result[i][j]);
				// resize resultStr ?
				while (strlen(resultStr) + strlen(resultShort) + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, resultShort);
#else
				// resize resultStr ?
				while (strlen(resultStr) + strlen(result[i][j]) + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, result[i][j]);
#endif
				// resize resultStr ?
				while (strlen(resultStr) + strlen(", ") + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, ", ");
			}
			// resize resultStr ?
			while (strlen(resultStr) + strlen("</FONT>") + 1 > resultStrSize) { // + 1 for \0
				resultStrSize *= 2;
				resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
				if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			strcat(resultStr, "</FONT>");
		}

		// types (blue)
		found = 0;
		for (j = 0; j < typeAttributesCount; ++j) {
			for (k = 0; k < typeAttributesHistogramCount[i][j]; ++k) {
#if USE_SHORT_NAMES
				char *resultShort = NULL;
#endif
				if (typeAttributesHistogram[i][j][k].percent < TYPE_FREQ_THRESHOLD) break;
				if (found == 0) {
					// first value found
					found = 1;
					// resize resultStr ?
					while (strlen(resultStr) + strlen("<FONT color=\"blue\">") + 1 > resultStrSize) { // + 1 for \0
						resultStrSize *= 2;
						resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
						if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					}
					strcat(resultStr, "<FONT color=\"blue\">");
				}
#if USE_SHORT_NAMES
				getPropNameShort(&resultShort, typeAttributesHistogram[i][j][k].value);
				// resize resultStr ?
				while (strlen(resultStr) + strlen(resultShort) + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, resultShort);
#else
				// resize resultStr ?
				while (strlen(resultStr) + strlen(typeAttributesHistogram[i][j][k].value) + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, typeAttributesHistogram[i][j][k].value);
#endif
				// resize resultStr ?
				while (strlen(resultStr) + strlen(", ") + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, ", ");
			}
		}
		if (found == 1) {
			// there was a type value
			// resize resultStr ?
			while (strlen(resultStr) + strlen("</FONT>") + 1 > resultStrSize) { // + 1 for \0
				resultStrSize *= 2;
				resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
				if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			strcat(resultStr, "</FONT>");
		}

		// incoming fks (green)
		if (links[i].num > 0) {
			// resize resultStr ?
			while (strlen(resultStr) + strlen("<FONT color=\"green\">") + 1 > resultStrSize) { // + 1 for \0
				resultStrSize *= 2;
				resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
				if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			strcat(resultStr, "<FONT color=\"green\">");
			for (j = 0; j < links[i].num; ++j) {
				char *temp = NULL;
#if USE_SHORT_NAMES
				char *resultShort = NULL;
#endif

				takeOid(links[i].fks[j].prop, &tmpStr);
				propStr = removeBrackets(tmpStr);
#if USE_SHORT_NAMES
				getPropNameShort(&resultShort, propStr);
				temp = (char *) malloc(sizeof(char) * (strlen(resultShort) + 3));
				if (!temp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				sprintf(temp, "%s, ", resultShort);
#else
				temp = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
				if (!temp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				sprintf(temp, "%s, ", propStr);
#endif

				// resize resultStr ?
				while (strlen(resultStr) + strlen(temp) + 1 > resultStrSize) { // + 1 for \0
					resultStrSize *= 2;
					resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
					if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				}
				strcat(resultStr, temp);
				free(temp);
			}
			// resize resultStr ?
			while (strlen(resultStr) + strlen("</FONT>") + 1 > resultStrSize) { // + 1 for \0
				resultStrSize *= 2;
				resultStr = realloc(resultStr, sizeof(char) * resultStrSize);
				if (!resultStr) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			}
			strcat(resultStr, "</FONT>");
		}

		if (strlen(resultStr) > 0) {
			// remove last comma
			strcpy((resultStr + (strlen(resultStr) - 9)), "</FONT>");
		} else {
			strcpy(resultStr, "---");
		}
#else
		(void) typeAttributesCount;
		(void) typeAttributesHistogram;
		(void) typeAttributesHistogramCount;
		(void) result;
		(void) resultCount;
		(void) links;
#endif

		// print header
		fprintf(fout, "\"" BUNFMT "\" [\n", cs.csId);
		fprintf(fout, "label = <<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n");
		fprintf(fout, "<TR><TD><B>%s (CS "BUNFMT", Freq: %d)</B></TD></TR>\n", labels[i].name, cs.csId, cs.support);
#if SHOW_CANDIDATES
		fprintf(fout, "<TR><TD><B>%s</B></TD></TR>\n", resultStr);
		free(resultStr);
#endif

		for (j = 0; j < cs.numProp; ++j) {
			char    *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char    *propStrShort = NULL;
#endif

			takeOid(cs.lstProp[j], &tmpStr);

			// copy propStr to propStrEscaped because .dot-PORTs cannot contain colons and quotes
			propStr = removeBrackets(tmpStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
#endif

			// if it is a type, include top-3 values
#if USE_SHORT_NAMES
			fprintf(fout, "<TR><TD PORT=\"%s\">%s</TD></TR>\n", propStrEscaped, propStrShort);
#else
			fprintf(fout, "<TR><TD PORT=\"%s\">%s</TD></TR>\n", propStrEscaped, propStr);
#endif
			free(propStrEscaped);

		}
		fprintf(fout, "</TABLE>>\n");
		fprintf(fout, "];\n\n");
	}

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];
		for (j = 0; j < cs.numProp; ++j) {
			char    *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char    *propStrShort = NULL;
#endif

			takeOid(cs.lstProp[j], &tmpStr);

			// copy propStr to propStrEscaped because .dot-PORTs cannot contain colons and quotes
			propStr = removeBrackets(tmpStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
#endif

			for (k = 0; k < relationMetadataCount[i][j]; ++k) {

				if (relationMetadata[i][j][k].percent >= FK_FREQ_THRESHOLD) {
					// target of links is frequent enough, not an outlier
					int from = relationMetadata[i][j][k].from;
					int to = relationMetadata[i][j][k].to;
#if USE_SHORT_NAMES
					fprintf(fout, "\""BUNFMT"\":\"%s\" -> \""BUNFMT"\" [label=\"%s\"];\n", freqCSset->items[from].csId, propStrEscaped, freqCSset->items[to].csId, propStrShort); // print foreign keys to dot file
#else
					fprintf(fout, "\""BUNFMT"\":\"%s\" -> \""BUNFMT"\" [label=\"%s\"];\n", freqCSset->items[from].csId, propStrEscaped, freqCSset->items[to].csId, propStr); // print foreign keys to dot file
#endif
				}
			}
			free(propStrEscaped);
		}
	}

	fprintf(fout, "}\n"); // footer

	fclose(fout);
}
*/

static
void printUML2(CSset *freqCSset, CSlabel* labels, Relation*** relationMetadata, int** relationMetadataCount, int freqThreshold) {
	int 		i, j, k;
	FILE 		*fout;
	char 		filename[20], tmp[10];

	int		smallest = -1, biggest = -1;

	strcpy(filename, "CS2max");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".dot");

	fout = fopen(filename, "wt");

	// header
	fprintf(fout, "digraph g {\n");
	fprintf(fout, "graph[ratio=\"compress\"];\n");
	fprintf(fout, "node [shape=\"none\"];\n\n");

	// find biggest and smallest table
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];
		if (!isCSTable(cs)) continue; // ignore

		// first values
		if (smallest == -1) smallest = i;
		if (biggest == -1) biggest = i;

		if (cs.coverage < freqCSset->items[smallest].coverage) smallest = i;
		if (cs.coverage > freqCSset->items[biggest].coverage) biggest = i;
	}

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		int width;
		str labelStr;
		str tmpStr;
		str labelStrEscaped = NULL;
#if USE_SHORT_NAMES
		str labelStrShort = NULL;
#endif

		CS cs = (CS) freqCSset->items[i];
		if (!isCSTable(cs)) continue; // ignore

		// print header
		width = (int) ((300 + 300 * (log10(freqCSset->items[i].coverage) - log10(freqCSset->items[smallest].coverage)) / (log10(freqCSset->items[biggest].coverage) - log10(freqCSset->items[smallest].coverage))) + 0.5); // width between 300 and 600 px, using logarithm
		fprintf(fout, "\"" BUNFMT "\" [\n", cs.csId);
		fprintf(fout, "label = <<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n");

		if (labels[i].name == BUN_NONE) {
			labelStrEscaped = (str) GDKmalloc(sizeof(char) * 6);
			if (!labelStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(labelStrEscaped, "DUMMY");
		} else {
			takeOid(labels[i].name, &tmpStr);
			labelStr = removeBrackets(tmpStr);
#if USE_SHORT_NAMES
			getPropNameShort(&labelStrShort, labelStr);
			labelStrEscaped = (str) GDKmalloc(sizeof(char) * (strlen(labelStrShort) + 1));
			if (!labelStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(labelStrEscaped, labelStrShort, (strlen(labelStrShort) + 1));
			escapeURI(labelStrEscaped);
			GDKfree(labelStrShort);
#else
			labelStrEscaped = (str) GDKmalloc(sizeof(char) * (strlen(labelStr) + 1));
			if (!labelStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(labelStrEscaped, labelStr, (strlen(labelStr) + 1));
			escapeURI(labelStrEscaped);
#endif
			GDKfree(tmpStr);
			GDKfree(labelStr);
		}

		fprintf(fout, "<TR><TD WIDTH=\"%d\"><B>%s (#triples: %d)</B></TD></TR>\n", width, labelStrEscaped, cs.coverage);
		GDKfree(labelStrEscaped);

		for (j = 0; j < cs.numProp; ++j) {
			str		propStr;
			str		tmpStr;
			char    *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char    *propStrShort = NULL;
#endif
			str color;

			takeOid(cs.lstProp[j], &tmpStr);

			// assign color (the more tuples the property occurs in, the darker
			if ((1.0 * cs.lstPropSupport[j])/cs.support > 0.8) {
				color = "#5555FF";
			} else if ((1.0 * cs.lstPropSupport[j])/cs.support > 0.6) {
				color = "#7777FF";
			} else if ((1.0 * cs.lstPropSupport[j])/cs.support > 0.4) {
				color = "#9999FF";
			} else if ((1.0 * cs.lstPropSupport[j])/cs.support > 0.2) {
				color = "#BBBBFF";
			} else {
				color = "#DDDDFF";
			}

			// copy propStr to propStrEscaped because .dot-PORTs cannot contain colons and quotes
			propStr = removeBrackets(tmpStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			fprintf(fout, "<TR><TD BGCOLOR=\"%s\" PORT=\"%s\">%s (%d%%)</TD></TR>\n", color, propStrEscaped, propStrShort, (100 * cs.lstPropSupport[j])/cs.support);
			GDKfree(propStrShort);
#else
			fprintf(fout, "<TR><TD BGCOLOR=\"%s\" PORT=\"%s\">%s (%d%%)</TD></TR>\n", color, propStrEscaped, propStrEscaped, (100 * cs.lstPropSupport[j])/cs.support);
#endif

			GDKfree(propStr);
			free(propStrEscaped);
			GDKfree(tmpStr); 

		}
		fprintf(fout, "</TABLE>>\n");
		fprintf(fout, "];\n\n");
	}

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];
		if (!isCSTable(cs)) continue; // ignore

		for (j = 0; j < cs.numProp; ++j) {
			str	tmpStr;
			str	propStr;
			char    *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char    *propStrShort = NULL;
#endif

			takeOid(cs.lstProp[j], &tmpStr);

			// copy propStr to propStrEscaped because .dot-PORTs cannot contain colons and quotes
			propStr = removeBrackets(tmpStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);

#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			for (k = 0; k < relationMetadataCount[i][j]; ++k) {

				if (relationMetadata[i][j][k].percent >= FK_FREQ_THRESHOLD) {
					// target of links is frequent enough, not an outlier
					int from = relationMetadata[i][j][k].from;
					int to = relationMetadata[i][j][k].to;
					fprintf(fout, "\""BUNFMT"\":\"%s\" -> \""BUNFMT"\" [label=\"%s\"];\n", freqCSset->items[from].csId, propStrEscaped, freqCSset->items[to].csId, propStrShort); // print foreign keys to dot file
				}
			}
			GDKfree(propStrShort);
#else
			for (k = 0; k < relationMetadataCount[i][j]; ++k) {

				if (relationMetadata[i][j][k].percent >= FK_FREQ_THRESHOLD) {
					// target of links is frequent enough, not an outlier
					int from = relationMetadata[i][j][k].from;
					int to = relationMetadata[i][j][k].to;
					fprintf(fout, "\""BUNFMT"\":\"%s\" -> \""BUNFMT"\" [label=\"%s\"];\n", freqCSset->items[from].csId, propStrEscaped, freqCSset->items[to].csId, propStrEscaped); // print foreign keys to dot file
				}
			}
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
	int cNew = label->candidatesNew, cOnto = label->candidatesOntology, cType = label->candidatesType, cFK = label->candidatesFK;

	if (label->candidatesCount < 2) return; // no duplicates

	// loop through all candidates
	for (i = 0; i < label->candidatesCount - 1; ++i) {
		// search (direction: right) whether this value occurs again
		int moveLeft = 0;
		for (j = i + 1; j < label->candidatesCount; ++j) {
			// find out which category (new, onto, type, fk) we are in
			int *cPtr = NULL;
			if (j < label->candidatesNew) cPtr = &cNew;
			else if (j < label->candidatesNew + label->candidatesOntology) cPtr = &cOnto;
			else if (j < label->candidatesNew + label->candidatesOntology + label->candidatesType) cPtr = &cType;
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
		label->candidatesOntology = cOnto;
		label->candidatesType = cType;
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
		} else if (label->candidatesOntology > 0) {
			label->candidatesOntology--;
		} else if (label->candidatesType > 0) {
			label->candidatesType--;
		} else {
			label->candidatesFK--;
		}
	}
}

#if USE_TABLE_NAME
/* For one CS: Choose the best table name out of all collected candidates (ontology, type, fk). */
static
void getTableName(CSlabel* label, int csIdx,  int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, TypeStat* typeStat, int typeStatCount, oid** result, int* resultCount, IncidentFKs* links, oid** ontmetadata, int ontmetadataCount, BAT *ontmetaBat, OntClass *ontclassSet) {
	int		i, j, k;
	oid		*tmpList;
	int		tmpListCount;
	char		nameFound = 0;
	oid		maxDepthOid;
	int		maxFreq;


	(void) ontmetaBat;


	// --- TYPE ---
	// get most frequent type value per type attribute
	tmpList = NULL;
	tmpListCount = 0;
	for (i = 0; i < typeAttributesCount; ++i) {
		if (typeAttributesHistogramCount[csIdx][i] == 0) continue;
		/*   //TODO: Uncomment this path
		for (j = 0; j < typeAttributesHistogramCount[csIdx][i]; j++){
			str typelabel; 
			BUN		ontClassPos; 	//Position of ontology in the ontmetaBat
			oid		typeOid; 	

			typeOid = typeAttributesHistogram[csIdx][i][j].value;
			printf("FreqCS %d : Type[%d][%d][oid] = " BUNFMT, csIdx, i,j, typeOid);
			ontClassPos = BUNfnd(BATmirror(ontmetaBat), &typeOid); 
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
		tmpList = (oid *) realloc(tmpList, sizeof(oid) * (tmpListCount + 1));
		if (!tmpList) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

		// of all values that are >= TYPE_FREQ_THRESHOLD, choose the value with the highest hierarchy level ("deepest" value)
		maxDepthOid = typeAttributesHistogram[csIdx][i][0].value;
		maxFreq = typeAttributesHistogram[csIdx][i][0].freq;
		for (j = 1; j < typeAttributesHistogramCount[csIdx][i]; ++j) {
			int depth, maxDepth;
			int freq;
			if (typeAttributesHistogram[csIdx][i][j].percent < TYPE_FREQ_THRESHOLD) break;
			depth = ontclassSet[BUNfnd(BATmirror(ontmetaBat), &typeAttributesHistogram[csIdx][i][j].value)].hierDepth;
			maxDepth = ontclassSet[BUNfnd(BATmirror(ontmetaBat), &maxDepthOid)].hierDepth;
			freq = typeAttributesHistogram[csIdx][i][j].freq;
			if (depth > maxDepth) {
				// choose value with higher hierarchy level
				maxDepthOid = typeAttributesHistogram[csIdx][i][j].value;
				maxFreq = freq;
			} else if (depth == maxDepth && freq > maxFreq) {
				// if both values are on the same level, choose the value with higher frequency
				maxDepthOid = typeAttributesHistogram[csIdx][i][j].value;
				maxFreq = freq;
			}
		}
		tmpList[tmpListCount] = maxDepthOid;
		tmpListCount += 1;
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
					}
				}
			}
		}
	}


	// --- ONTOLOGY ---
	// add all ontology candidates to list of candidates
	if (resultCount[csIdx] >= 1) {
		label->candidatesOntology = resultCount[csIdx];
		label->candidates = GDKrealloc(label->candidates, sizeof(oid) * (label->candidatesCount + resultCount[csIdx]));
		if (!label->candidates) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		for (i = 0; i < resultCount[csIdx]; ++i) {
			label->candidates[label->candidatesCount + i] = result[csIdx][i];
		}
		label->candidatesCount += resultCount[csIdx];
	}

	// one ontology class --> use it
	if (!nameFound){
	if (resultCount[csIdx] == 1) {
		label->name = result[csIdx][0];
		label->hierarchy = getOntoHierarchy(label->name, &(label->hierarchyCount), ontmetadata, ontmetadataCount);
		nameFound = 1;
		#if INFO_WHERE_NAME_FROM
		label->isOntology = 1; 
		#endif
	}
	}

	if (!nameFound) {
		// multiple ontology classes --> intersect with types
		if (resultCount[csIdx] > 1) {
			tmpList = NULL;
			tmpListCount = 0;
			// search for type values
			for (i = 0; i < typeAttributesCount; ++i) {
				for (j = 0; j < typeAttributesHistogramCount[csIdx][i]; ++j) {
					if (typeAttributesHistogram[csIdx][i][j].percent < TYPE_FREQ_THRESHOLD) break; // sorted

					// intersect type with ontology classes
					for (k = 0; k < resultCount[csIdx]; ++k) {
						if (result[csIdx][k] == typeAttributesHistogram[csIdx][i][j].value) {
							// found, copy ontology class to tmpList
							tmpList = (oid *) realloc(tmpList, sizeof(oid) * (tmpListCount + 1));
							if (!tmpList) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
							tmpList[tmpListCount] = result[csIdx][k];
							tmpListCount += 1;
						}
					}
				}
			}

			// only one left --> use it
			if (tmpListCount == 1) {
				label->name = tmpList[0];
				label->hierarchy = getOntoHierarchy(label->name, &(label->hierarchyCount), ontmetadata, ontmetadataCount);
				free(tmpList);
				nameFound = 1;
				#if INFO_WHERE_NAME_FROM
				label->isOntology = 1; 
				#endif
			}

			if (!nameFound) {
				// multiple left --> use the class that covers most attributes, most popular ontology, ...
				if (tmpListCount > 1) {
					label->name = tmpList[0]; // sorted
					label->hierarchy = getOntoHierarchy(label->name, &(label->hierarchyCount), ontmetadata, ontmetadataCount);
					free(tmpList);
					nameFound = 1;
					
					#if INFO_WHERE_NAME_FROM
					label->isOntology = 1; 
					#endif
				}
			}

			if (!nameFound) {
				// empty intersection -> use the class that covers most attributes, most popular ontology, ..
				label->name = result[csIdx][0]; // sorted
				label->hierarchy = getOntoHierarchy(label->name, &(label->hierarchyCount), ontmetadata, ontmetadataCount);
				free(tmpList);
				nameFound = 1;

				#if INFO_WHERE_NAME_FROM
				label->isOntology = 1; 
				#endif
			}
		}
	}



	// --- FK ---
	// add top3 fk values to list of candidates
	if (links[csIdx].num > 0) {
		label->candidatesFK = MIN(3, links[csIdx].num);
		label->candidates = GDKrealloc(label->candidates, sizeof(oid) * (label->candidatesCount + MIN(3, links[csIdx].num)));
		if (!label->candidates) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		for (i = 0; i < MIN(3, links[csIdx].num); ++i) {
			label->candidates[label->candidatesCount + i] = links[csIdx].fks[0].prop;
		}
		label->candidatesCount += MIN(3, links[csIdx].num);
	}

	if (!nameFound) {
		// incident foreign keys --> use the one with the most occurances (num and freq)
		if (links[csIdx].num > 0) {
			label->name = links[csIdx].fks[0].prop; // sorted
			nameFound = 1;

			#if INFO_WHERE_NAME_FROM
			label->isFK = 1; 
			#endif
		}
	}

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
		labels[i].candidatesOntology = 0;
		labels[i].candidatesType = 0;
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
void getAllLabels(CSlabel* labels, CSset* freqCSset,  int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, TypeStat* typeStat, int typeStatCount, oid** result, int* resultCount, IncidentFKs* links, oid** ontmetadata, int ontmetadataCount, BAT *ontmetaBat, OntClass *ontclassSet) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];

		// get table name
		getTableName(&labels[i], i,  typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeStat, typeStatCount, result, resultCount, links, ontmetadata, ontmetadataCount, ontmetaBat, ontclassSet);

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
void printTree(OntoUsageNode* tree, int level) {
	int i;
	str uriStr;

	takeOid(tree->uri, &uriStr);
	printf("Level %d URI %s Count %d Sum %d Percent %.1f\n", level, uriStr, tree->numOccurances, tree->numOccurancesSum, tree->percentage * 100);

	for (i = 0; i < tree->numChildren; ++i) {
		printTree(tree->lstChildren[i], level+1);
	}
}

static
void createOntoUsageTree(OntoUsageNode** tree, CSset* freqCSset, oid** ontmetadata, int ontmetadataCount, oid** result, int* resultCount, int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount) {
	int 		i, j, k, l;
	oid		*tmpList;
	int		tmpListCount;
	int 		numTuples = 0;

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

		// get ontology
		// copied from getTableName
		if (resultCount[i] == 0) {
			// no hierarchy --> ignore
			continue;
		} else if (resultCount[i] == 1) {
			// one ontology class --> use it
			uri = result[i][0];
		} else {
			// multiple ontology classes --> intersect with types
			tmpList = NULL;
			tmpListCount = 0;
			// search for type values
			for (l = 0; l < typeAttributesCount; ++l) {
				for (j = 0; j < typeAttributesHistogramCount[i][l]; ++j) {
					if (typeAttributesHistogram[i][l][j].percent < TYPE_FREQ_THRESHOLD) break; // sorted
					// intersect type with ontology classes
					for (k = 0; k < resultCount[i]; ++k) {
						if (result[i][k] == typeAttributesHistogram[i][l][j].value) {
							// found, copy ontology class to tmpList
							tmpList = (oid *) realloc(tmpList, sizeof(oid) * (tmpListCount + 1));
							if (!tmpList) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
							tmpList[tmpListCount] = result[i][k];
							tmpListCount += 1;
						}
					}
				}
			}
			if (tmpListCount == 1) {
				// only one left --> use it
				uri = tmpList[0];
			} else if (tmpListCount > 1) {
				// multiple left --> use the class that covers most attributes, most popular ontology, ...
				uri = tmpList[0]; // sorted
			} else {
				// empty intersection -> use the class that covers most attributes, most popular ontology, ..
				uri = result[i][0]; // sorted
			}
			free(tmpList);
		}

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
	printTree(*tree, 0);
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
void freeOntologyLookupResult(oid** ontologyLookupResult, int csCount) {
	int		i;

	for (i = 0; i < csCount; ++i) {
		if (ontologyLookupResult[i])
			free(ontologyLookupResult[i]);
	}
	free(ontologyLookupResult);
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
			"<http://dbpedia.org/ontology/typeOfElectrification>"}; // <...> necessary to get the correct oids

int			typeAttributesCount = 17;
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
	IncidentFKs		*links;
	CSlabel			*labels;

	str		schema = "rdf";
	int		ret;

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

	// Relation (FK)
	relationMetadataCount = initRelationMetadataCount(freqCSset);
	relationMetadata = initRelationMetadata(relationMetadataCount, csrelSet, num, freqCSset);
	links = initLinks(freqCSset->numCSadded);
#if USE_FK_NAMES
	createLinks(freqCSset, relationMetadata, relationMetadataCount, links);
#endif

	// Ontologies
	ontologyLookupResultCount = initOntologyLookupResultCount(freqCSset->numCSadded);
	ontologyLookupResult = initOntologyLookupResult(freqCSset->numCSadded);
#if USE_ONTOLOGY_NAMES
	createOntologyLookupResult(ontologyLookupResult, freqCSset, ontologyLookupResultCount, ontattributes, ontattributesCount, ontmetadata, ontmetadataCount);
	// TODO ont-data have to be freed on shutdown of the database
	// freeOntattributes(ontattributes);
	// freeOntmetadata(ontmetadata);
#else
	(void) ontattributesCount;
	(void) ontattributes;
#endif

	// Assigning Names
	labels = initLabels(freqCSset);
#if USE_TABLE_NAME
	getAllLabels(labels, freqCSset, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeStat, typeStatCount, ontologyLookupResult, ontologyLookupResultCount, links, ontmetadata, ontmetadataCount, ontmetaBat, ontclassSet);
	if (typeStatCount > 0) free(typeStat);
#endif

	// Collect ontology statistics (tree)
	createOntoUsageTree(ontoUsageTree, freqCSset, ontmetadata, ontmetadataCount, ontologyLookupResult, ontologyLookupResultCount, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount);

	free(ontologyLookupResultCount);
	freeOntologyLookupResult(ontologyLookupResult, freqCSset->numCSadded);
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
oid* mergeCandidates(int *candidatesCount, int *candidatesNew, int *candidatesOntology, int *candidatesType, int *candidatesFK, CSlabel cs1, CSlabel cs2, oid commonName) {
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

	// copy "ontology"
	for (i = 0; i < cs1.candidatesOntology; ++i) {
		candidates[counter] = cs1.candidates[cs1.candidatesNew + i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesOntology; ++i) {
		candidates[counter] = cs2.candidates[cs2.candidatesNew + i];
		counter++;
	}
	(*candidatesOntology) = counter - (*candidatesNew);

	// copy "type"
	for (i = 0; i < cs1.candidatesType; ++i) {
		candidates[counter] = cs1.candidates[cs1.candidatesNew + cs1.candidatesOntology + i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesType; ++i) {
		candidates[counter] = cs2.candidates[cs2.candidatesNew + cs2.candidatesOntology + i];
		counter++;
	}
	(*candidatesType) = counter - (*candidatesNew) - (*candidatesOntology);

	// copy "fk"
	for (i = 0; i < cs1.candidatesFK; ++i) {
		candidates[counter] = cs1.candidates[cs1.candidatesNew + cs1.candidatesOntology + cs1.candidatesType + i];
		counter++;
	}
	for (i = 0; i < cs2.candidatesFK; ++i) {
		candidates[counter] = cs2.candidates[cs2.candidatesNew + cs2.candidatesOntology + cs2.candidatesType + i];
		counter++;
	}
	(*candidatesFK) = counter - (*candidatesNew) - (*candidatesOntology) - (*candidatesType);

	return candidates;
}

/* Create labels for merged CS's. Uses rules S1 to S5 (new names!).
 * If no MERGECS is created (subset-superset relation), mergeCSFreqId contains the Id of the superset class.
 * For S1 and S2, parameter 'name' is used to avoid recomputation of CS names
 */
str updateLabel(int ruleNumber, CSset *freqCSset, CSlabel **labels, int newCS, int mergeCSFreqId, int freqCS1, int freqCS2, oid name, oid **ontmetadata, int ontmetadataCount, int *lstFreqId, int numIds){
	int		i;
	int		freqCS1Counter;
	CSlabel		big, small;
	CSlabel		*label;
	CS		cs;	
	#if     USE_MULTIWAY_MERGING
	int		tmpMaxCoverage; 
	int		tmpFreqId;
	#endif
	oid		*mergedCandidates = NULL;
	int		candidatesCount, candidatesNew, candidatesOntology, candidatesType, candidatesFK;

	(void) lstFreqId;
	(void) numIds;

	if (newCS) {
		// realloc labels
		*labels = GDKrealloc(*labels, sizeof(CSlabel) * freqCSset->numCSadded);
		if (!(*labels)) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		(*labels)[mergeCSFreqId].name = BUN_NONE; 
		(*labels)[mergeCSFreqId].candidates = NULL;
		(*labels)[mergeCSFreqId].candidatesCount = 0;
		(*labels)[mergeCSFreqId].candidatesNew = 0;
		(*labels)[mergeCSFreqId].candidatesOntology = 0;
		(*labels)[mergeCSFreqId].candidatesType = 0;
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

		#if     USE_MULTIWAY_MERGING
		(void)ontmetadata;
		(void)ontmetadataCount; 
		(void)freqCS2;

		#else
		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesOntology, &candidatesType, &candidatesFK, (*labels)[freqCS1], (*labels)[freqCS2], label->name);
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesOntology = candidatesOntology;
		label->candidatesType = candidatesType;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);
		if (label->name == BUN_NONE && label->candidates[0] != BUN_NONE) {
			label->name = label->candidates[0];
		}

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

		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesOntology, &candidatesType, &candidatesFK, (*labels)[freqCS1], (*labels)[freqCS2], label->name);
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesOntology = candidatesOntology;
		label->candidatesType = candidatesType;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);
		if (label->name == BUN_NONE && label->candidates[0] != BUN_NONE) {
			label->name = label->candidates[0];
		}

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
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesOntology, &candidatesType, &candidatesFK, (*labels)[freqCS1], (*labels)[freqCS2], label->name); // freqCS1 is superCS, freqCS2 is subCS
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesOntology = candidatesOntology;
		label->candidatesType = candidatesType;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);
		if (label->name == BUN_NONE && label->candidates[0] != BUN_NONE) {
			label->name = label->candidates[0];
		}

		// hierarchy already set
		// properties already set

		break;

		case S4: // FALLTHROUGH
		case S5:
		#if	USE_MULTIWAY_MERGING
		tmpMaxCoverage = 0; 
		tmpFreqId = 0;
		for (i = 0; i < numIds; i++){
			if (freqCSset->items[lstFreqId[i]].coverage > tmpMaxCoverage){
				tmpFreqId = lstFreqId[i];
				tmpMaxCoverage = freqCSset->items[lstFreqId[i]].coverage;
			}
		}
		big = &(*labels)[tmpFreqId];

		#else
		// use label of biggest CS (higher coverage value)
		if (freqCSset->items[freqCS1].coverage > freqCSset->items[freqCS2].coverage) {
			big = (*labels)[freqCS1];
			small = (*labels)[freqCS2];
		} else {
			big = (*labels)[freqCS2];
			small = (*labels)[freqCS1];
		}
		#endif
		label->name = big.name;

		// candidates
		mergedCandidates = mergeCandidates(&candidatesCount, &candidatesNew, &candidatesOntology, &candidatesType, &candidatesFK, big, small, label->name);
		GDKfree(label->candidates);
		label->candidates = mergedCandidates; // TODO check access outside function
		label->candidatesCount = candidatesCount;
		label->candidatesNew = candidatesNew;
		label->candidatesOntology = candidatesOntology;
		label->candidatesType = candidatesType;
		label->candidatesFK = candidatesFK;
		removeDuplicatedCandidates(label);
		if (label->name == BUN_NONE && label->candidates[0] != BUN_NONE) {
			label->name = label->candidates[0];
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

void exportLabels(CSlabel* labels, CSset* freqCSset, CSrel* csRelMergeFreqSet, int freqThreshold, int*  mTblIdxFreqIdxMapping, int* mfreqIdxTblIdxMapping, int numTables) {
	int			**relationMetadataCount;
	Relation		***relationMetadata;


	str             schema = "rdf";
	int             ret;
	
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		fprintf(stderr, "ERROR: Couldn't open tokenizer!\n");
	}
	// FK
	printf("exportLabels: initRelationMetadataCount \n"); 
	relationMetadataCount = initRelationMetadataCount(freqCSset);
	printf("exportLabels: initRelationMetadata2 \n"); 
	relationMetadata = initRelationMetadata2(relationMetadataCount, csRelMergeFreqSet, freqCSset);
	
	// Print and Export
	printf("exportLabels: printUML \n"); 
	printUML2(freqCSset, labels, relationMetadata, relationMetadataCount, freqThreshold);
	convertToSQL(freqCSset, relationMetadata, relationMetadataCount, labels, freqThreshold);
	createSQLMetadata(freqCSset, csRelMergeFreqSet, labels,  mTblIdxFreqIdxMapping, mfreqIdxTblIdxMapping, numTables);
	printTxt(freqCSset, labels, freqThreshold);
	printf("exportLabels: Done \n"); 
	freeRelationMetadata(relationMetadata, freqCSset);
	freeRelationMetadataCount(relationMetadataCount, freqCSset->numCSadded);

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
