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
{{"http:", "www.facebook.com", "2008"}, 3},
{{"http:", "facebook.com", "2008"}, 3},
{{"http:", "developers.facebook.com", "schema"}, 3},
{{"https:", "www.facebook.com", "2008"}, 3},
{{"http:", "purl.org", "dc", "elements", "1.1"}, 5}, // dc DublinCore
{{"http:", "purl.org", "dc", "terms"}, 4}, // DublinCore
{{"http:", "purl.org", "goodrelations", "v1"}, 4}, // GoodRelations
{{"http:", "purl.org", "rss", "1.0", "modules"}, 5},
{{"http:", "purl.org", "stuff"}, 3},
{{"http:", "www.purl.org", "stuff"}, 3},
{{"http:", "ogp.me", "ns"}, 3},
{{"https:", "ogp.me", "ns"}, 3},
{{"http:", "www.w3.org", "1999", "02", "22-rdf-syntax-ns"}, 5}, // rdf
{{"http:", "www.w3.org", "2000", "01", "rdf-schema"}, 5}, // rdfs
{{"http:", "www.w3.org", "2004", "02", "skos", "core"}, 6}, // skos (Simple Knowledge Organization System)
{{"http:", "www.w3.org", "2002", "07", "owl"}, 5},
{{"http:", "www.w3.org", "2006", "vcard", "ns"}, 5}, // vcard
{{"http:", "www.w3.org", "2001", "vcard-rdf", "3.0"}, 5},
{{"http:", "www.w3.org", "2003", "01", "geo", "wgs84_pos"}, 6}, // geo
{{"http:", "www.w3.org", "1999", "xhtml", "vocab"}, 5}, // xhtml
{{"http:", "search.yahoo.com", "searchmonkey"}, 3},
{{"https:", "search.yahoo.com", "searchmonkey"}, 3},
{{"http:", "search.yahoo.co.jp", "searchmonkey"}, 3},
{{"http:", "g.yahoo.com", "searchmonkey"}, 3},
{{"http:", "opengraphprotocol.org", "schema"}, 3},
{{"https:", "opengraphprotocol.org", "schema"}, 3},
{{"http:", "opengraph.org", "schema"}, 3},
{{"https:", "opengraph.org", "schema"}, 3},
{{"http:", "creativecommons.org", "ns"}, 3}, // cc
{{"http:", "rdf.data-vocabulary.org"}, 2}, // by google
{{"http:", "rdfs.org", "sioc", "ns"}, 4}, // sioc (pronounced "shock", Semantically-Interlinked Online Communities Project)
{{"http:", "xmlns.com", "foaf", "0.1"}, 4}, // foaf (Friend of a Friend)
{{"http:", "mixi-platform.com", "ns"}, 3}, // japanese social graph
{{"http:", "commontag.org", "ns"}, 3},
{{"http:", "semsl.org", "ontology"}, 3}, // semantic web for second life
{{"http:", "schema.org"}, 2},
{{"http:", "openelectiondata.org", "0.1"}, 3},
{{"http:", "search.aol.com", "rdf"}, 3},
{{"http:", "www.loc.gov", "loc.terms", "relators"}, 4}, // library of congress
{{"http:", "dbpedia.org", "ontology"}, 3}, // dbo
{{"http:", "dbpedia.org", "resource"}, 3}, // dbpedia
{{"http:", "dbpedia.org", "property"}, 3}, // dbp
{{"http:", "www.aktors.org", "ontology", "portal"}, 4}, // akt (research, publications, ...)
{{"http:", "purl.org", "ontology", "bibo"}, 4}, // bibo (bibliography)
{{"http:", "purl.org", "ontology", "mo"}, 4}, // mo (music)
{{"http:", "www.geonames.org", "ontology"}, 3}, // geonames
{{"http:", "purl.org", "vocab", "frbr", "core"}, 5}, // frbr (Functional Requirements for Bibliographic Records)
{{"http:", "www.w3.org", "2001", "XMLSchema"}, 4}, // xsd
{{"http:", "www.w3.org", "2006", "time"}, 4}, // time
{{"http:", "purl.org", "NET", "c4dm", "event.owl"}, 5}, // event
{{"http:", "www.openarchives.org", "ore", "terms"}, 4}, // ore (Open Archive)
{{"http:", "purl.org", "vocab", "bio", "0.1"}, 5}, // bio (biographical data)
{{"http:", "www.holygoat.co.uk", "owl", "redwood", "0.1", "tags"}, 6}, // tag
{{"http:", "rdfs.org", "ns", "void"}, 4}, // void (Vocabulary of Interlinked Datasets)
{{"http:", "www.w3.org", "2006", "http"}, 4}, // http
{{"http:", "purl.uniprot.org", "core"}, 3}, // uniprot (protein annotation)
{{"http:", "umbel.org", "umbel"}, 3}, // umbel (Upper Mapping and Binding Exchange Layer)
{{"http:", "purl.org", "stuff", "rev"}, 4}, // rev (review)
{{"http:", "purl.org", "linked-data", "cube"}, 4}, // qb (data cube)
{{"http:", "www.w3.org", "ns", "org"}, 4}, // org (organizations)
{{"http:", "purl.org", "vocab", "vann"}, 4}, // vann (vocabulary for annotating vocabulary descriptions)
{{"http:", "data.ordnancesurvey.co.uk", "ontology", "admingeo"}, 4}, // admingeo (administrative geography and civil voting area)
{{"http:", "www.w3.org", "2007", "05", "powder-s"}, 5}, // wdrs (Web Description Resources)
{{"http:", "usefulinc.com", "ns", "doap"}, 4}, // doap (Description of a Project)
{{"http:", "lod.taxonconcept.org", "ontology", "txn.owl"}, 4}, // txn (TaxonConcept, species)
{{"http:", "xmlns.com", "wot", "0.1"}, 4}, // wot (Web Of Trust)
{{"http:", "purl.org", "net", "compass"}, 4}, // compass
{{"http:", "www.w3.org", "2004", "03", "trix", "rdfg-1"}, 6}, // rdfg (RDF graph)
{{"http:", "purl.org", "NET", "c4dm", "timeline.owl"}, 5}, // tl (timeline)
{{"http:", "purl.org", "dc", "dcam"}, 4}, // dcam (DublinCore metadata)
{{"http:", "swrc.ontoware.org", "ontology"}, 3}, // swrc (university, research)
{{"http:", "zeitkunst.org", "bibtex", "0.1", "bibtex.owl"}, 5}, // bib (bibTeX entries)
{{"http:", "purl.org", "ontology", "po"}, 4} // po (tv and radio programmes)
};

#if USE_SHORT_NAMES
/* Extracts the "human-readable" part of an URI (usually the last token). */
static
void getPropNameShort(char** name, char* propStr) {
	char		*token;
	char		*uri;
	int		length = 0;		// number of tokens
	char		**tokenizedUri = NULL;	// list of tokens
	int		i, j;
	int		fit;

	// tokenize uri
	uri = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
	if (!uri) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	strcpy(uri, propStr); // uri will be modified during tokenization
	token = strtok(uri, "/#");
	while (token != NULL) {
		tokenizedUri = realloc(tokenizedUri, sizeof(char*) * ++length);
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
				(*name) = (char *) malloc(sizeof(char) * (totalLength + 1));
				if (!(*name)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				strcpy(*name, "\0");

				for (i = ontologies[j].length; i < length; ++i) {
					strcat(*name, tokenizedUri[i]);
					strcat(*name, "_"); // if label consists of >=2 tokens, use underscores
				}
				// remove trailing underscore
				(*name)[strlen(*name) - 1] = '\0';

				free(tokenizedUri);
				return;
			}
		}
	}

	// no matching ontology found, return content of last token

	if (length == 1) {
		// value
		(*name) = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
		if (!(*name)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(*name, propStr);
	} else {
		(*name) = (char *) malloc(sizeof(char) * (strlen(tokenizedUri[length - 1]) + 1));
		if (!(*name)) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(*name, tokenizedUri[length - 1]);
	}

	free(tokenizedUri);
	free(uri);
	return;
}
#endif

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
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
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
Relation*** initRelationMetadata(int** relationMetadataCount, CSmergeRel* csRelBetweenMergeFreqSet, CSset* freqCSset) {
	int		i, j, k;
	Relation***	relationMetadata;

	int		ret;
	char*		schema = "rdf";

	TKNZRopen (NULL, &schema);

	relationMetadata = (Relation ***) malloc(sizeof(Relation **) * freqCSset->numCSadded);
	if (!relationMetadata) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < freqCSset->numCSadded; ++i) { // CS
		CS cs = (CS) freqCSset->items[i];
		if (cs.parentFreqIdx != -1) continue; // ignore
		relationMetadata[i] = (Relation **) malloc (sizeof(Relation *) * cs.numProp);
		if (!relationMetadata[i]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < cs.numProp; ++j) { // propNo in CS order
			int sum = 0;
			relationMetadataCount[i][j] = 0;
			relationMetadata[i][j] = NULL;
			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) { // propNo in CSrel

				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == cs.lstProp[j]) {
					int toId = csRelBetweenMergeFreqSet[i].lstRefFreqIdx[k];
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

	TKNZRclose(&ret);

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

/* Modifies the parameter! */
/* from:   <URI>/ or <URI>   to:   URI */
static
void removeBrackets(char** s) {
	if (strlen(*s) < 2) return;

	if ((*s)[0] == '<' && (*s)[strlen(*s) - 2] == '>' && (*s)[strlen(*s) - 1] == '/') {
		// case <URI>/
		(*s)[strlen(*s) - 2] = '\0';
		(*s) += 1;
	} else if ((*s)[0] == '<' && (*s)[strlen(*s) - 2] == '/' && (*s)[strlen(*s) - 1] == '>') {
		// case <URI/>
		(*s)[strlen(*s) - 2] = '\0';
		(*s) += 1;
	} else if ((*s)[0] == '<' && (*s)[strlen(*s) - 1] == '>') {
		// case <URI>
		(*s)[strlen(*s) - 1] = '\0';
		(*s) += 1;
	} else if ((*s)[strlen(*s) - 1] == '/') {
		// case URI/
		(*s)[strlen(*s) - 1] = '\0';
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
		if (s[i] == ':' || s[i] == '"' || s[i] == ' ' || s[i] == '-') s[i] = '_';
		s[i] = tolower(s[i]);
	}
}

/* Create SQL CREATE TABLE statements including foreign keys. */
static
void convertToSQL(CSset *freqCSset, Relation*** relationMetadata, int** relationMetadataCount, Labels* labels, int freqThreshold) {
	// tokenizer
	int		ret;
	char*		schema = "rdf";

	// file i/o
	FILE		*fout;
	char		filename[20], tmp[10];

	// looping
	int		i, j, k;

	// tokenizer
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		fprintf(stderr, "could not open the tokenizer\n");
	}

	strcpy(filename, "CS");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".sql");

	fout = fopen(filename, "wt");

	// create statement for every table
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		char *temp;
		if ( freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		temp = (char *) malloc(sizeof(char) * (strlen(labels[i].name) + 1));
		if (!temp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(temp, labels[i].name);
		escapeURIforSQL(temp);
		fprintf(fout, "CREATE TABLE %s_"BUNFMT" (\nsubject VARCHAR(10) PRIMARY KEY,\n", temp, freqCSset->items[i].csId); // TODO underscores?
		free(temp);
		for (j = 0; j < labels[i].numProp; ++j) {
			char *temp2;
			temp2 = (char *) malloc(sizeof(char) * (strlen(labels[i].lstProp[j]) + 1));
			if (!temp2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(temp2, labels[i].lstProp[j]);
			escapeURIforSQL(temp2);

			if (j + 1 < labels[i].numProp) {
				fprintf(fout, "%s_%d BOOLEAN,\n", temp2, j);
			} else {
				// last column
				fprintf(fout, "%s_%d BOOLEAN\n", temp2, j);
			}
			free(temp2);
		}
		fprintf(fout, ");\n\n");
	}

	// add foreign key columns and add foreign keys
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		for (j = 0; j < labels[i].numProp; ++j) {
			char *temp2;
			int refCounter = 0;
			temp2 = (char *) malloc(sizeof(char) * (strlen(labels[i].lstProp[j]) + 1));
			if (!temp2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(temp2, labels[i].lstProp[j]);
			escapeURIforSQL(temp2);

			for (k = 0; k < relationMetadataCount[i][j]; ++k) {
				int from, to;
				char *tempFrom, *tempTo;
				if (relationMetadata[i][j][k].percent < FK_FREQ_THRESHOLD) continue; // foreign key is not frequent enough
				from = relationMetadata[i][j][k].from;
				to = relationMetadata[i][j][k].to;
				tempFrom = (char *) malloc(sizeof(char) * (strlen(labels[from].name) + 1));
				if (!tempFrom) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				tempTo = (char *) malloc(sizeof(char) * (strlen(labels[to].name) + 1));
				if (!tempTo) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				strcpy(tempFrom, labels[from].name);
				escapeURIforSQL(tempFrom);
				strcpy(tempTo, labels[to].name);
				escapeURIforSQL(tempTo);

				fprintf(fout, "ALTER TABLE %s_"BUNFMT" ADD COLUMN %s_%d_%d VARCHAR(10);\n", tempFrom, freqCSset->items[from].csId, temp2, j, refCounter);
				fprintf(fout, "ALTER TABLE %s_"BUNFMT" ADD FOREIGN KEY (%s_%d_%d) REFERENCES %s_"BUNFMT"(subject);\n\n", tempFrom, freqCSset->items[from].csId, temp2, j, refCounter, tempTo, freqCSset->items[to].csId);
				refCounter += 1;
				free(tempFrom);
				free(tempTo);
			}
			free(temp2);
		}
	}

	fclose(fout);
	TKNZRclose(&ret);
}

static
void createSQLMetadata(CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet, Labels* labels) {
	int	**matrix = NULL; // matrix[from][to] frequency
	int	i, j, k;
	FILE	*fout;

	// init
	matrix = (int **) malloc(sizeof(int *) * freqCSset->numCSadded);
	if (!matrix) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		matrix[i] = (int *) malloc(sizeof(int) * freqCSset->numCSadded);
		if (!matrix) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

		for (j = 0; j < freqCSset->numCSadded; ++j) {
			matrix[i][j] = 0;
		}
	}

	// set values
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore

		for (j = 0; j < freqCSset->items[i].numProp; ++j) { // propNo in CS order
			// check foreign key frequency
			int sum = 0;
			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) {
				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == freqCSset->items[i].lstProp[j]) {
					sum += csRelBetweenMergeFreqSet[i].lstCnt[k];
				}
			}

			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) { // propNo in CSrel
				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == freqCSset->items[i].lstProp[j]) {
					int to = csRelBetweenMergeFreqSet[i].lstRefFreqIdx[k];
					if (i == to) continue; // ignore self references
					if ((int) (100.0 * csRelBetweenMergeFreqSet[i].lstCnt[k] / sum + 0.5) < FK_FREQ_THRESHOLD) continue; // foreign key is not frequent enough
					matrix[i][to] += csRelBetweenMergeFreqSet[i].lstCnt[k]; // multiple links from 'i' to 'to'? add the frequencies
				}
			}
		}
	}

	// store matrix as csv
	fout = fopen("adjacencyList.csv", "wt");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < freqCSset->numCSadded; ++j) {
			if (matrix[i][j]) {
				fprintf(fout, "%d,%d,%d\n", i, j, matrix[i][j]);
			}
		}
	}
	fclose(fout);

	// print id -> table name
	fout = fopen("tableIdFreq.csv", "wt");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		char *temp;
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		temp = (char *) malloc(sizeof(char) * (strlen(labels[i].name) + 1));
		if (!temp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		strcpy(temp, labels[i].name);
		escapeURIforSQL(temp);
		fprintf(fout, "%d,\"%s_"BUNFMT"\",%d\n", i, temp, freqCSset->items[i].csId, freqCSset->items[i].support); // TODO underscores?
		free(temp);
	}
	fclose(fout);

	fout = fopen("CSmetadata.sql", "wt");
	fprintf(fout, "CREATE TABLE table_id_freq (id INTEGER, name VARCHAR(100), frequency INTEGER);\n");
	fprintf(fout, "CREATE TABLE adjacency_list (from_id INTEGER, to_id INTEGER, frequency INTEGER);\n");
	fprintf(fout, "COPY INTO table_id_freq from '/export/scratch2/linnea/dbfarm/test/tableIdFreq.csv' USING DELIMITERS ',','\\n','\"';\n");
	fprintf(fout, "COPY INTO adjacency_list from '/export/scratch2/linnea/dbfarm/test/adjacencyList.csv' USING DELIMITERS ',','\\n','\"';");
	fclose(fout);
}

/* Simple representation of the final labels for tables and attributes. */
static
void printTxt(CSset* freqCSset, Labels* labels, int freqThreshold) {
	FILE 		*fout;
	char		filename[20], tmp[10];
	int		i, j;

	strcpy(filename, "labels");
	sprintf(tmp, "%d", freqThreshold);
	strcat(filename, tmp);
	strcat(filename, ".txt");

	fout = fopen(filename, "wt");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		fprintf(fout, "%s (CS "BUNFMT"): ", labels[i].name, freqCSset->items[i].csId);
		for (j = 0; j < labels[i].numProp; ++j) {
			if (j + 1 < labels[i].numProp) fprintf(fout, "%s, ", labels[i].lstProp[j]);
			else fprintf(fout, "%s\n", labels[i].lstProp[j]);
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
/* Loop through all subjects to collect frequency statistics for type attribute values. */
static
void createTypeAttributesHistogram(BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, BAT* mapbat, CSset *freqCSset, int *csIdFreqIdxMap, int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, char** typeAttributes) {
	// looping, extracting
	BUN		p, q;
	oid 		*sbt, *obt, *pbt;
	char 		objType;
	str		propStr, objStr;
	char		*objStrPtr;

	char		*start, *end;
	int		length;

	oid 		objOid;
	BUN		bun;
	BATiter		mapi;
	int		csFreqIdx;
	char		*schema = "rdf";

	// histogram
	int		i, j, k;
	int		fit;

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		fprintf(stderr, "could not open the tokenizer\n");
	}

	if (BATcount(sbat) == 0) {
		fprintf(stderr, "sbat must not be empty");
		/* otherwise, variable sbt is not initialized and thus
		 * cannot be dereferenced after the BATloop below */
	}

	mapi = bat_iterator(mapbat);

	BATloop(sbat, p, q) {
		// Get data
		sbt = (oid *) BUNtloc(si, p);
		pbt = (oid *) BUNtloc(pi, p);

		csFreqIdx = csIdFreqIdxMap[subjCSMap[*sbt]];
		if (csFreqIdx == -1) {
			// subject does not belong to a freqCS
			continue;
		}

		// get property, check if it is a type
		takeOid(*pbt, &propStr);
		for (i = 0; i < typeAttributesCount; ++i) {
			if (strstr(propStr, typeAttributes[i]) != NULL) {
				// prop is a type!

				// lookup maxCS/mergeCS
				csFreqIdx = csIdFreqIdxMap[subjCSMap[*sbt]];
				while (freqCSset->items[csFreqIdx].parentFreqIdx != -1) {
					csFreqIdx = freqCSset->items[csFreqIdx].parentFreqIdx;
				}

				// get object
				obt = (oid *) BUNtloc(oi, p);
				objOid = *obt;
				objType = (char) ((*obt) >> (sizeof(BUN)*8 - 4))  &  7;

				if (objType == URI || objType == BLANKNODE) {
					objOid = objOid - ((oid)objType << (sizeof(BUN)*8 - 4));
					takeOid(objOid, &objStr);
					removeBrackets(&objStr);
					objStrPtr = objStr;
				} else {
					objOid = objOid - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the real objOid from Map or Tokenizer */
					bun = BUNfirst(mapbat);
					objStr = (str) BUNtail(mapi, bun + objOid);

					// get part between enclosing quotation marks
					start = strchr(objStr, '"') + 1;
					end = strrchr(objStr, '"');
					if (start != NULL && end != NULL) {
						length = end - start;
						objStrPtr = (char *) malloc(sizeof(char) * (length + 1));
						if (!objStrPtr) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
						memcpy(objStrPtr, start, length);
						objStrPtr[length] = '\0';
					} else {
						objStrPtr = objStr;
					}
				}

				// add object to histogram
				fit = 0;
				for (j = 0; j < typeAttributesHistogramCount[csFreqIdx][i]; ++j) {
					if (strcmp(typeAttributesHistogram[csFreqIdx][i][j].value, objStrPtr) == 0) {
						// bucket exists
						typeAttributesHistogram[csFreqIdx][i][j].freq += 1;
						fit = 1;
						break;
					}
				}
				if (!fit) {
					// bucket does not exist
					// realloc
					typeAttributesHistogramCount[csFreqIdx][i] += 1;
					typeAttributesHistogram[csFreqIdx][i] = (TypeAttributesFreq *) realloc(typeAttributesHistogram[csFreqIdx][i], sizeof(TypeAttributesFreq) * typeAttributesHistogramCount[csFreqIdx][i]);
					if (!typeAttributesHistogram[csFreqIdx][i]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

					// insert value
					typeAttributesHistogram[csFreqIdx][i][typeAttributesHistogramCount[csFreqIdx][i] - 1].value = (str) malloc(sizeof(char)*(strlen(objStrPtr)+1));
					if (!typeAttributesHistogram[csFreqIdx][i][typeAttributesHistogramCount[csFreqIdx][i] - 1].value) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
					strcpy(typeAttributesHistogram[csFreqIdx][i][typeAttributesHistogramCount[csFreqIdx][i] - 1].value, objStrPtr);
					typeAttributesHistogram[csFreqIdx][i][typeAttributesHistogramCount[csFreqIdx][i] - 1].freq = 1;
				}

				if (!(objType == URI || objType == BLANKNODE)) free(objStrPtr); // malloc, therefore free
				break;
			}
		}
	}

	// sort descending by frequency
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		for (j = 0; j < typeAttributesCount; ++j) {
			qsort(typeAttributesHistogram[i][j], typeAttributesHistogramCount[i][j], sizeof(TypeAttributesFreq), compareTypeAttributesFreqs);
		}
	}

	TKNZRclose(NULL);

	// assign percentage
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < typeAttributesCount; ++j) {
			int sum = 0;
			// get total count of values
			for (k = 0; k < typeAttributesHistogramCount[i][j]; ++k) {
				sum += typeAttributesHistogram[i][j][k].freq;
			}
			// assign percentage values for every value
			for (k = 0; k < typeAttributesHistogramCount[i][j]; ++k) {
				typeAttributesHistogram[i][j][k].percent = (int) (100.0 * typeAttributesHistogram[i][j][k].freq / sum + 0.5);
			}
		}
	}
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
					if (strcmp(typeAttributesHistogram[i][j][k].value, typeStat[l].value) == 0) {
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
					typeStat[*typeStatCount].value = typeAttributesHistogram[i][j][k].value; // pointer, no copy
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
			str		propStr;

			takeOid(cs.lstProp[j], &propStr);
			removeBrackets(&propStr);
			uri = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!uri) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			strcpy(uri, propStr);

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
					propOntologies[i][propOntologiesCount[i]] = propStr;
					(*propOntologiesOids)[i][propOntologiesCount[i]] = cs.lstProp[j];
					propOntologiesCount[i] += 1;
				}
			}
			for (k = 0; k < length; ++k) {
				free(tokenizedUri[k]);
			}
			free(tokenizedUri);
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
str* getOntologyCandidates(str** ontattributes, int ontattributesCount, str** ontmetadata, int ontmetadataCount, int *resultCount, str **list, oid **listOids, int *listCount, int listNum, PropStat *propStat) {
	int		i, j, k, l;
	str		*result = NULL;

	for (i = 0; i < listNum; ++i) {
		int		filledListsCount = 0;
		str		**candidates = NULL;
		int		*candidatesCount;
		ClassStat*	classStat = NULL;
		int		num;
		float		totalTfidfs;

		if (listCount[i] == 0) continue;

		candidates = (str **) malloc(sizeof(str *) * listCount[i]);
		candidatesCount = (int *) malloc(sizeof(int) * listCount[i]);
		if (!candidates || !candidatesCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < listCount[i]; ++j) {
			candidates[j] = NULL;
			candidatesCount[j] = 0;
		}

		for (j = 0; j < ontattributesCount; ++j) {
			str auristr = ontattributes[0][j];
			str aattrstr = ontattributes[1][j];

			for (k = 0; k < listCount[i]; ++k) {
				if (strcmp(aattrstr, list[i][k]) == 0) {
					// attribute found, store auristr in list
					candidates[k] = realloc(candidates[k], sizeof(str) * (candidatesCount[k] + 1));
					if (!candidates[k]) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
					candidates[k][candidatesCount[k]] = auristr;
					candidatesCount[k] += 1;
					if (candidatesCount[k] == 1) filledListsCount += 1; // new list
				}
			}
		}

		if (filledListsCount == 0) {
			continue; // no new results will be generated using this ontology
		}
		totalTfidfs = 0.0;
		num = 0;
		for (j = 0; j < listCount[i]; ++j) { // for each list
			BUN p, bun;
			p = listOids[i][j];
			bun = BUNfnd(BATmirror(propStat->pBat), (ptr) &p);
			for (k = 0; k < candidatesCount[j]; ++k) { // for each candidate
				// search for this class
				int found = 0;
				for (l = 0; l < num; ++l) {
					if (strcmp(candidates[j][k], classStat[l].ontoClass)== 0) {
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
			if (classStat[k].tfidfs < ONTOLOGY_FREQ_THRESHOLD) break; // values not frequent enough (list is sorted by tfidfs)
			for (j = 0; j < ontmetadataCount && (found == 0); ++j) {
				str muristr = ontmetadata[0][j];
				str msuperstr = ontmetadata[1][j];
				if (strcmp(classStat[k].ontoClass, muristr) == 0) {
					if (strcmp(msuperstr, "\x80") == 0) {
						// muristr is a candidate! (because it has no superclass)
						result = realloc(result, sizeof(char *) * ((*resultCount) + 1));
						if (!result) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						result[*resultCount] = (str) malloc(sizeof(char) * (strlen(muristr) + 1));
						if (!result[*resultCount]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
						strcpy(result[*resultCount], muristr);
						*resultCount += 1;
						found = 1;
						break;
					}

					for (l = 0; l < num; ++l) {
						if (classStat[k].tfidfs > classStat[l].tfidfs) break; // superclass has to have a higher/equal tfidf to be concidered
						if (strcmp(msuperstr, classStat[l].ontoClass) == 0) {
							// superclass is in list, therefore do not copy subclass
							found = 1;
							break;
						}
					}
				}
			}
			if (found == 0) {
				// superclass not in list, classStat[k].ontoClass is a candidate!
				result = realloc(result, sizeof(str) * ((*resultCount) + 1));
				if (!result) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				result[*resultCount] = (str) malloc(sizeof(char) * (strlen(classStat[k].ontoClass) + 1));
				if (!result[*resultCount]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
				strcpy(result[*resultCount], classStat[k].ontoClass);
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
str** initOntologyLookupResult(int csCount) {
	str		**result; // result[cs][index] (list of class names per cs)
	int i;

	result = (str **) malloc(sizeof(str *) * csCount);
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
static
void createPropStatistics(PropStat* propStat, int numMaxCSs, CSset* freqCSset) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS)freqCSset->items[i];
		if (cs.parentFreqIdx != -1) continue; // ignore
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
#endif

#if USE_ONTOLOGY_NAMES
static
void freePropStat(PropStat *propStat) {
	BBPreclaim(propStat->pBat);
	free(propStat->freqs);
	free(propStat->tfidfs);
}
#endif

#if USE_ONTOLOGY_NAMES
/* For all CS: Calculate the ontology classes that are similar (tfidf) to the list of attributes. */
static
void createOntologyLookupResult(str** result, CSset* freqCSset, int* resultCount, str** ontattributes, int ontattributesCount, str** ontmetadata, int ontmetadataCount) {
	int		i, j;
	PropStat	*propStat;
	int		numCS = 0;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx == -1) numCS += 1;
	}
	propStat = initPropStat();
	createPropStatistics(propStat, numCS, freqCSset);

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS		cs;
		str		**propOntologies = NULL;
		oid		**propOntologiesOids = NULL;
		int		*propOntologiesCount = NULL;

		cs = (CS) freqCSset->items[i];
		if (cs.parentFreqIdx != -1) continue; // ignore

		// order properties by ontologies
		propOntologiesCount = (int *) malloc(sizeof(int) * ontologyCount);
		if (!propOntologiesCount) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < ontologyCount; ++j) {
			propOntologiesCount[j] = 0;
		}
		propOntologies = findOntologies(cs, propOntologiesCount, &propOntologiesOids);

		// get class names
		resultCount[i] = 0;
		result[i] = getOntologyCandidates(ontattributes, ontattributesCount, ontmetadata, ontmetadataCount, &(resultCount[i]), propOntologies, propOntologiesOids, propOntologiesCount, ontologyCount, propStat);

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
static
void printUML(CSset *freqCSset, int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, str** result, int* resultCount, IncidentFKs* links, Labels* labels, Relation*** relationMetadata, int** relationMetadataCount, int freqThreshold) {
	str 		propStr;
	int		ret;
	char*   	schema = "rdf";

#if SHOW_CANDIDATES
	char*           resultStr = NULL;
	unsigned int    resultStrSize = 100;
	int		found;
#endif

	int 		i, j, k;
	FILE 		*fout;
	char 		filename[20], tmp[10];

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		fprintf(stderr, "could not open the tokenizer\n");
	}

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
		if (cs.parentFreqIdx != -1) continue; // ignore

#if SHOW_CANDIDATES
		/* DATA SOURCES */
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
				str propStr;
				char *temp = NULL;
#if USE_SHORT_NAMES
				char *resultShort = NULL;
#endif

				takeOid(links[i].fks[j].prop, &propStr);
				removeBrackets(&propStr);
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
			takeOid(cs.lstProp[j], &propStr);

			// copy propStr to propStrEscaped because .dot-PORTs cannot contain colons and quotes
			removeBrackets(&propStr);
			propStrEscaped = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!propStrEscaped) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(propStrEscaped, propStr, (strlen(propStr) + 1));
			escapeURI(propStrEscaped);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
#endif

			if (cs.parentFreqIdx == -1) {
				// if it is a type, include top-3 values
#if USE_SHORT_NAMES
				fprintf(fout, "<TR><TD PORT=\"%s\">%s</TD></TR>\n", propStrEscaped, propStrShort);
#else
				fprintf(fout, "<TR><TD PORT=\"%s\">%s</TD></TR>\n", propStrEscaped, propStr);
#endif
			}
			free(propStrEscaped);

		}
		fprintf(fout, "</TABLE>>\n");
		fprintf(fout, "];\n\n");
	}

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];
		if (cs.parentFreqIdx != -1) continue; // ignore
		for (j = 0; j < cs.numProp; ++j) {
			char    *propStrEscaped = NULL;
#if USE_SHORT_NAMES
			char    *propStrShort = NULL;
#endif
			takeOid(cs.lstProp[j], &propStr);

			// copy propStr to propStrEscaped because .dot-PORTs cannot contain colons and quotes
			removeBrackets(&propStr);
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

	TKNZRclose(&ret);
}

#if USE_TABLE_NAME
/* For one CS: Choose the best table name out of all collected candidates (ontology, type, fk). */
static
void getTableName(char** name, int csIdx,  int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, TypeStat* typeStat, int typeStatCount, str** result, int* resultCount, IncidentFKs* links) {
	int		i, j, k;
	str		*tmpList;
	int		tmpListCount;

	// --- ONTOLOGY ---
	// one ontology class --> use it
	if (resultCount[csIdx] == 1) {
#if USE_SHORT_NAMES
		getPropNameShort(name, result[csIdx][0]);
#else
		(*name) = (char *) malloc(sizeof(char) * (strlen(result[csIdx][0]) + 1));
		strcpy(*name, result[csIdx][0]);
#endif
		return;
	}

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
					if (strcmp(result[csIdx][k], typeAttributesHistogram[csIdx][i][j].value) == 0) {
						// found, copy ontology class to tmpList
						tmpList = (str *) realloc(tmpList, sizeof(str) * (tmpListCount + 1));
						if (!tmpList) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
						tmpList[tmpListCount] = result[csIdx][k]; // pointer, no copy
						tmpListCount += 1;
					}
				}
			}
		}
		// only one left --> use it
		if (tmpListCount == 1) {
#if USE_SHORT_NAMES
			getPropNameShort(name, tmpList[0]);
#else
			(*name) = (char *) malloc(sizeof(char) * (strlen(tmpList[0]) + 1));
			strcpy(*name, tmpList[0]);
#endif
			free(tmpList);
			return;
		}
		// multiple left --> use the class that covers most attributes, most popular ontology, ...
		if (tmpListCount > 1) {
#if USE_SHORT_NAMES
			getPropNameShort(name, tmpList[0]); // sorted
#else
			(*name) = (char *) malloc(sizeof(char) * (strlen(tmpList[0]) + 1));
			strcpy(*name, tmpList[0]); // sorted
#endif
			free(tmpList);
			return;
		}
		// empty intersection -> use the class that covers most attributes, most popular ontology, ..
#if USE_SHORT_NAMES
		getPropNameShort(name, result[csIdx][0]); // sorted
#else
		(*name) = (char *) malloc(sizeof(char) * (strlen(result[csIdx][0]) + 1));
		strcpy(*name, result[csIdx][0]); // sorted
#endif
		free(tmpList);
		return;
	}

	// --- TYPE ---
	// get most frequent type value per type attribute
	tmpList = NULL;
	tmpListCount = 0;
	for (i = 0; i < typeAttributesCount; ++i) {
		if (typeAttributesHistogramCount[csIdx][i] == 0) continue;
		if (typeAttributesHistogram[csIdx][i][0].percent < TYPE_FREQ_THRESHOLD) continue; // sorted
		tmpList = (str *) realloc(tmpList, sizeof(str) * (tmpListCount + 1));
		if (!tmpList) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		tmpList[tmpListCount] = typeAttributesHistogram[csIdx][i][0].value; // pointer, no copy
		tmpListCount += 1;
	}
	// one type attribute --> use most frequent one
	if (tmpListCount == 1) {
		// only one type attribute, use most frequent value (sorted)
#if USE_SHORT_NAMES
		getPropNameShort(name, tmpList[0]);
#else
		(*name) = (char *) malloc(sizeof(char) * (strlen(tmpList[0]) + 1));
		strcpy(*name, tmpList[0]);
#endif
		return;
	}
	// multiple type attributes --> use the one with fewest occurances in other CS's
	if (tmpListCount > 1) {
		for (i = 0; i < typeStatCount; ++i) {
			for (j = 0; j < tmpListCount; ++j) {
				if (strcmp(typeStat[i].value, tmpList[j]) == 0) {
#if USE_SHORT_NAMES
					getPropNameShort(name, tmpList[j]);
#else
					(*name) = (char *) malloc(sizeof(char) * (strlen(tmpList[j]) + 1));
					strcpy(*name, tmpList[j]);
#endif
					return;
				}
			}
		}
	}

	// --- FK ---
	// incident foreign keys --> use the one with the most occurances (num and freq)
	if (links[csIdx].num > 0) {
		str propStr;
		takeOid(links[csIdx].fks[0].prop, &propStr); // sorted
		removeBrackets(&propStr);
#if USE_SHORT_NAMES
		getPropNameShort(name, propStr);
#else
		(*name) = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
		strcpy(*name, propStr);
#endif
		return;
	}

	// --- NOTHING ---
	(*name) = (char *) malloc(sizeof(char) * 6);
	strcpy(*name, "DUMMY");
	return;
}
#endif

static
Labels* initLabels(CSset *freqCSset) {
	Labels		*labels;
	int		i;

	labels = (Labels *) malloc(sizeof(Labels) * freqCSset->numCSadded);
	if (!labels) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		labels[i].name = NULL;
		labels[i].numProp = 0;
		labels[i].lstProp = NULL;
	}
	return labels;
}

#if USE_TABLE_NAME
/* Creates the final result of the labeling: table name and attribute names. */
static
void getAllLabels(Labels* labels, CSset* freqCSset,  int typeAttributesCount, TypeAttributesFreq*** typeAttributesHistogram, int** typeAttributesHistogramCount, TypeStat* typeStat, int typeStatCount, str** result, int* resultCount, IncidentFKs* links) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		CS cs = (CS) freqCSset->items[i];
		char *temp = NULL;
		if (cs.parentFreqIdx != -1) continue; // ignore

		// get table name
		getTableName(&temp, i,  typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeStat, typeStatCount, result, resultCount, links);
		labels[i].name = (char *) realloc(labels[i].name, sizeof(char) * (strlen(temp) + 1));
		if (!labels[i].name) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		memcpy(labels[i].name, temp, sizeof(char) * (strlen(temp) + 1));

		// get attribute names
		labels[i].numProp = cs.numProp;
		labels[i].lstProp = (str *) malloc(sizeof(str) * cs.numProp);
		if (!labels[i].lstProp) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < cs.numProp; ++j) {
			str propStr;
#if USE_SHORT_NAMES
			char *propStrShort = NULL;
#endif
			takeOid(cs.lstProp[j], &propStr);
			removeBrackets(&propStr);
#if USE_SHORT_NAMES
			getPropNameShort(&propStrShort, propStr);
			labels[i].lstProp[j] = (char *) malloc(sizeof(char) * (strlen(propStrShort) + 1));
			if (!labels[i].lstProp[j]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(labels[i].lstProp[j], propStrShort, sizeof(char) * (strlen(propStrShort) + 1));
#else
			labels[i].lstProp[j] = (char *) malloc(sizeof(char) * (strlen(propStr) + 1));
			if (!labels[i].lstProp[j]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
			memcpy(labels[i].lstProp[j], propStr, sizeof(char) * (strlen(propStr) + 1));
#endif
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
		if (cs.parentFreqIdx != -1) continue; // ignore
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
		if (cs.parentFreqIdx != -1) continue; // ignore
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
void freeOntologyLookupResult(str** ontologyLookupResult, int csCount) {
	int		i;

	for (i = 0; i < csCount; ++i) {
		if (ontologyLookupResult[i])
			free(ontologyLookupResult[i]);
	}
	free(ontologyLookupResult);
}

/* Creates labels for all CS (without a parent). */
Labels* createLabels(CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet, BAT *sbat, BATiter si, BATiter pi, BATiter oi, oid *subjCSMap, BAT* mbat, int *csIdFreqIdxMap, int freqThreshold, str** ontattributes, int ontattributesCount, str** ontmetadata, int ontmetadataCount) {
#if USE_TYPE_NAMES
	char*		typeAttributes[] = {
				"http://ogp.me/ns#type",
				"https://ogp.me/ns#type",
				"http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
				"http://purl.org/dc/elements/1.1/type",
				"http://mixi-platform.com/ns#type",
				"http://ogp.me/ns/fb#type",
				"http://opengraph.org/schema/type",
				"http://opengraphprotocol.org/schema/type",
				"http://purl.org/dc/terms/type",
				"http://purl.org/goodrelations/v1#typeOfGood",
				"http://search.yahoo.com/searchmonkey/media/type",
				"https://opengraphprotocol.org/schema/type",
				"https://search.yahoo.com/searchmonkey/media/type",
				"http://www.w3.org/1999/xhtmltype",
				"http://dbpedia.org/ontology/longtype",
				"http://dbpedia.org/ontology/type",
				"http://dbpedia.org/ontology/typeOfElectrification"};
#endif
	int			typeAttributesCount = 17;
	int			**typeAttributesHistogramCount;
	TypeAttributesFreq	***typeAttributesHistogram;
	TypeStat		*typeStat = NULL;
	int			typeStatCount = 0;
	int			**relationMetadataCount;
	Relation		***relationMetadata;
	str			**ontologyLookupResult;
	int			*ontologyLookupResultCount;
	IncidentFKs		*links;
	Labels			*labels;

	// Type
	typeAttributesHistogramCount = initTypeAttributesHistogramCount(typeAttributesCount, freqCSset->numCSadded);
	typeAttributesHistogram = initTypeAttributesHistogram(typeAttributesCount, freqCSset->numCSadded);
#if USE_TYPE_NAMES
	createTypeAttributesHistogram(sbat, si, pi, oi, subjCSMap, mbat, freqCSset, csIdFreqIdxMap, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeAttributes);
	typeStat = getTypeStats(&typeStatCount, freqCSset->numCSadded, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount);
#else
	(void) sbat;
	(void) si;
	(void) pi;
	(void) oi;
	(void) subjCSMap;
	(void) mbat;
	(void) csIdFreqIdxMap;
#endif

	// Relation (FK)
	relationMetadataCount = initRelationMetadataCount(freqCSset);
	relationMetadata = initRelationMetadata(relationMetadataCount, csRelBetweenMergeFreqSet, freqCSset);
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
	(void) ontmetadataCount;
	(void) ontattributes;
	(void) ontmetadata;
#endif

	// Assigning Names
	labels = initLabels(freqCSset);
#if USE_TABLE_NAME
	getAllLabels(labels, freqCSset, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, typeStat, typeStatCount, ontologyLookupResult, ontologyLookupResultCount, links);
	if (typeStatCount > 0) free(typeStat);
#endif

	// Print and Export
	printUML(freqCSset, typeAttributesCount, typeAttributesHistogram, typeAttributesHistogramCount, ontologyLookupResult, ontologyLookupResultCount, links, labels, relationMetadata, relationMetadataCount, freqThreshold);
	free(ontologyLookupResultCount);
	freeOntologyLookupResult(ontologyLookupResult, freqCSset->numCSadded);
	freeTypeAttributesHistogram(typeAttributesHistogram, freqCSset->numCSadded, typeAttributesCount);
	freeTypeAttributesHistogramCount(typeAttributesHistogramCount, freqCSset->numCSadded);
	freeLinks(links, freqCSset->numCSadded);
	convertToSQL(freqCSset, relationMetadata, relationMetadataCount, labels, freqThreshold);
	freeRelationMetadata(relationMetadata, freqCSset);
	freeRelationMetadataCount(relationMetadataCount, freqCSset->numCSadded);
	createSQLMetadata(freqCSset, csRelBetweenMergeFreqSet, labels);
	printTxt(freqCSset, labels, freqThreshold);

	return labels;
}

void freeLabels(Labels* labels, CSset* freqCSset) {
	int		i, j;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		for (j = 0; j < labels[i].numProp; ++j) {
			free(labels[i].lstProp[j]);
		}
		free(labels[i].name);
		if (labels[i].lstProp)
			free(labels[i].lstProp);
	}
	free(labels);
}
