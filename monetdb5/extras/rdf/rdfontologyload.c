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
#include "mal_exception.h"
#include "url.h"
#include "tokenizer.h"
#include <gdk.h>
#include <rdf.h>
#include <rdfparser.h>
#include <rdfontologyload.h>

/**
 * Ontology vocabulary is stored in two tables
 * Table1: ontMetaTbl [uri, label, subclassof]
 * Table2: ontAttTbl [uri, attribute]
 */
typedef enum{
	OntURI,
	OntLabel,
	OntSubclassof,
	OntAttURI, 
	OntAttAtt
} OntTableCol; 

#define N_ONTOLOGY_BAT (OntAttAtt+1)

typedef struct OntBATdef {
	OntTableCol batType;     /* BAT type             */
	str name;                /* name of the BAT/column      */
	int headType;            /* type of left column  */
        int tailType;            /* type of right column */
} OntBATdef;

static OntBATdef ontBatdef[N_ONTOLOGY_BAT]= {
	{OntURI, "uri", TYPE_void, TYPE_oid},
	{OntLabel, "label", TYPE_void, TYPE_str},	
	{OntSubclassof, "subclassof", TYPE_void, TYPE_oid},
	{OntAttURI, "uriatt", TYPE_void, TYPE_oid},	
	{OntAttAtt, "attribute", TYPE_void, TYPE_oid}	
};



static BAT*
create_OntologyBAT(int ht, int tt, int size)
{
	BAT *b = BATnew(ht, tt, size);
	if (b == NULL) {
		return b;
	}
	BATseqbase(b, 0);

	/* disable all properties */
	b->tsorted = FALSE;
	b->tdense = FALSE;
	b->tkey = FALSE;
	b->hdense = TRUE;

	return b;
}

static parserData*
createOntoglogyParser (str location, BAT** graph){

	int i;

	parserData *pdata = (parserData *) GDKmalloc(sizeof(parserData));
	if (pdata == NULL) return NULL;

	pdata->tcount = 0;
	pdata->exception = 0;
	pdata->fatal = 0;
	pdata->error = 0;
	pdata->lasterror = 0;
	pdata->warning = 0;
	pdata->location = location;
	pdata->graph = graph;

	for (i = 0; i < N_ONTOLOGY_BAT; i++) {
		pdata->graph[i] = NULL;
		pdata->graph[i] = create_OntologyBAT (
				ontBatdef[i].headType,
				ontBatdef[i].tailType,
				batsz);                       /* TODO: estimate size */
		if (pdata->graph[i] == NULL) {
			return NULL;
		}
	}

	return pdata; 

}	

static void
clean(parserData *pdata){
	int iret; 
	if (pdata != NULL) {
		for (iret = 0; iret < N_ONTOLOGY_BAT; iret++) {
			if (pdata->graph[iret] != NULL)
				BBPreclaim(pdata->graph[iret]);
		}

		GDKfree(pdata);
	}
}

static void
tripleHandler(void* user_data, const raptor_statement* triple)
{
	parserData *pdata = ((parserData *) user_data);

	printf("%s   %s   %s\n",(char*) raptor_term_to_string(triple->subject),(char *) raptor_term_to_string(triple->predicate), (char *)raptor_term_to_string(triple->object));
	
	pdata->tcount++;
	return; 
}


/*
static 
OntClass* creatOC(oid ocId, oid subClassofId, int numP, oid* buff)
{
	OntClass *oc = (OntClass*)malloc(sizeof(OntClass)); 
	oc->lstProp =  (oid*) malloc(sizeof(oid) * numP);
	
	if (oc->lstProp == NULL){
		printf("Malloc failed. at %d", numP);
		exit(-1); 
	}

	copyOidSet(oc->lstProp, buff, numP); 
	oc->ocId = ocId;
	oc->subclassof = subClassofId; 
	oc->numProp = numP; 
	oc->numAllocation = numP; 
	return oc; 
}

str
RDFOntologyRead(int *ret, bat *ontcBatid, bat *ontaBatid, OntClassset* ontclassset){
	BAT* ontcBat; 	// BAT for ontology classes 
	BAT* ontaBat; 	// BAT for ontology 
	if ((ontcBat = BATdescriptor(*ontcBatid)) == NULL) {
		throw(MAL, "rdf.RDFOntologyParser", RUNTIME_OBJECT_MISSING);
	}

	if ((ontaBat = BATdescriptor(*ontaBatid)) == NULL) {
		BBPreleaseref(ontcBat->batCacheid);
		throw(MAL, "rdf.RDFOntologyParser", RUNTIME_OBJECT_MISSING);
	}

	*ret = 1; 
	return MAL_SUCCEED; 
}

*/
str
RDFOntologyParser(int *xret, str *location, str *schema){

	raptor_parser *ontparser;
	parserData *pdata;
	raptor_uri *uri;
	bit isURI;
	str ret;
	int iret;
	raptor_world *world; 

	BAT *graph[N_ONTOLOGY_BAT];

	printf("Loading ontology from %s to %s schema \n", *location, *schema);

	/* init tokenizer */
	if (TKNZRopen (NULL, schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfShred",
				"could not open the tokenizer\n");
	}

	/* Init pdata  */
	pdata = createOntoglogyParser(*location,graph);
	if (pdata == NULL) {
		TKNZRclose(&iret);
		clean(pdata);		
		throw(RDF, "rdf.rdfShred",
				"could not allocate enough memory for pdata\n");
	}

	/* Init raptor2 */
	world = raptor_new_world();
	pdata->rparser = ontparser = raptor_new_parser(world,"guess");
	//pdata->ontparser = ontparser = raptor_new_parser(world,"turtle");

	if (ontparser == NULL) {
		TKNZRclose(&iret);
		raptor_free_world(world);
		
		clean(pdata);

		throw(RDF, "rdf.rdfShred", "could not create raptor parser object\n");
	}
	
	/* set callback handler for triples */
	raptor_parser_set_statement_handler   ( ontparser,  pdata,  (raptor_statement_handler) tripleHandler);
	/* set message handlers */
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) fatalHandler);
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) warningHandler);
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) errorHandler);

	/* Parse URI or local file. */
	ret = URLisaURL(&isURI, location);
	if (ret != MAL_SUCCEED) {
		TKNZRclose(&iret);
		clean(pdata);	

		return ret;
	} else if (isURI) {
		uri = raptor_new_uri(world, (unsigned char *) pdata->location);
		iret = raptor_parser_parse_uri(ontparser, uri, NULL);
	} else {
		
                uri = raptor_new_uri(world,
                                raptor_uri_filename_to_uri_string(pdata->location));
                iret = raptor_parser_parse_file(ontparser, uri, NULL);
	}
	
	/* Free memory of raptor */
	raptor_free_parser(ontparser);
	raptor_free_uri(uri);
	raptor_free_world(world);

	TKNZRclose(&iret);

	if (iret) {
		
		clean(pdata);
		throw(RDF, "rdf.rdfShred", "parsing failed\n");
	}
	
	printf("Finish Ontology parsing: \n Total number of error %d , fatal %d , warning %d \n", pdata->error, pdata->fatal, pdata->warning);

	clean(pdata);
	
	*xret = 1; 

	return MAL_SUCCEED;
}

// defined in rdfschema.c
extern oid **ontattributes;
extern int ontattributesCount;
extern oid **ontmetadata;
extern int ontmetadataCount;

str
RDFloadsqlontologies(int *ret, bat *auriid, bat *aattrid, bat *muriid, bat *msuperid){
	BUN			p, q;
	BAT			*auri = NULL, *aattr = NULL, *muri = NULL, *msuper = NULL;
	BATiter			aurii, aattri, murii, msuperi;
	BUN			bun, bun2, bun3, bun4;
	BUN			auriCount, muriCount;
	int			i;
	str			schema = "rdf";

	TKNZRopen (NULL, &schema);

	if (ontattributesCount != 0 || ontmetadataCount != 0) {
		// ontology data is already loaded
		*ret = 1;
		return MAL_SUCCEED;
	}

	if ((auri = BATdescriptor(*auriid)) == NULL) {
		throw(MAL, "rdf.RDFloadsqlontologies", RUNTIME_OBJECT_MISSING);
	}

	if ((aattr = BATdescriptor(*aattrid)) == NULL) {
		BBPreleaseref(auri->batCacheid);
		throw(MAL, "rdf.RDFloadsqlontologies", RUNTIME_OBJECT_MISSING);
	}
	if ((muri = BATdescriptor(*muriid)) == NULL) {
		BBPreleaseref(auri->batCacheid);
		BBPreleaseref(aattr->batCacheid);
		throw(MAL, "rdf.RDFloadsqlontologies", RUNTIME_OBJECT_MISSING);
	}
	if ((msuper = BATdescriptor(*msuperid)) == NULL) {
		BBPreleaseref(auri->batCacheid);
		BBPreleaseref(aattr->batCacheid);
		BBPreleaseref(muri->batCacheid);
		throw(MAL, "rdf.RDFloadsqlontologies", RUNTIME_OBJECT_MISSING);
	}

	aurii = bat_iterator(auri);
	aattri = bat_iterator(aattr);
	murii = bat_iterator(muri);
	msuperi = bat_iterator(msuper);

	// load ontattributes
	i = 0;
	bun = BUNfirst(auri);
	bun2 = BUNfirst(aattr);

	auriCount = BATcount(auri);

	ontattributes = (oid**) malloc(sizeof(oid *) * 2);
	if (!ontattributes) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	ontattributes[0] = malloc(sizeof(str) * auriCount); // uri
	ontattributes[1] = malloc(sizeof(str) * auriCount); // attr
	if (!ontattributes[0] || !ontattributes[1]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	BATloop(auri, p, q){
		str auristr = (str) BUNtail(aurii, bun + i);
		str aattrstr = (str) BUNtail(aattri, bun2 + i);

		oid aurioid, aattroid;

		// add <...> around strings
		str auristr2 = (str) GDKmalloc(strlen(auristr) + 3);
		str aattrstr2 = (str) GDKmalloc(strlen(aattrstr) + 3);
		if (!auristr2 || !aattrstr2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		auristr2[0] = '<';
		strcpy(&(auristr2[1]), auristr);
		auristr2[strlen(auristr) + 1] = '>';
		auristr2[strlen(auristr) + 2] = '\0';
		aattrstr2[0] = '<';
		strcpy(&(aattrstr2[1]), aattrstr);
		aattrstr2[strlen(aattrstr) + 1] = '>';
		aattrstr2[strlen(aattrstr) + 2] = '\0';

		TKNZRappend(&aurioid, &auristr2);
		TKNZRappend(&aattroid, &aattrstr2);

		ontattributes[0][ontattributesCount] = aurioid;
		ontattributes[1][ontattributesCount] = aattroid;
		ontattributesCount += 1;

		++i;

		GDKfree(auristr2);
		GDKfree(aattrstr2);
	}

	// load ontmetadata
	i = 0;
	bun3 = BUNfirst(muri);
	bun4 = BUNfirst(msuper);

	muriCount = BATcount(muri);

	ontmetadata = (oid**) malloc(sizeof(oid *) * 2);
	if (!ontmetadata) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
	ontmetadata[0] = malloc(sizeof(str) * muriCount); // uri
	ontmetadata[1] = malloc(sizeof(str) * muriCount); // superclass
	if (!ontmetadata[0] || !ontmetadata[1]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	BATloop(muri, p, q){
		str muristr = (str) BUNtail(murii, bun3 + i);
		str msuperstr = (str) BUNtail(msuperi, bun4 + i);

		oid murioid, msuperoid;

		// add <...> around strings
		str muristr2 = (str) GDKmalloc(strlen(muristr) + 3);
		str msuperstr2 = (str) GDKmalloc(strlen(msuperstr) + 3);
		if (!muristr2 || !msuperstr2) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		muristr2[0] = '<';
		strcpy(&(muristr2[1]), muristr);
		muristr2[strlen(muristr) + 1] = '>';
		muristr2[strlen(muristr) + 2] = '\0';
		msuperstr2[0] = '<';
		strcpy(&(msuperstr2[1]), msuperstr);
		msuperstr2[strlen(msuperstr) + 1] = '>';
		msuperstr2[strlen(msuperstr) + 2] = '\0';

		TKNZRappend(&murioid, &muristr2);
		TKNZRappend(&msuperoid, &msuperstr2);

		ontmetadata[0][ontmetadataCount] = murioid;
		if (strcmp(msuperstr, "\x80") == 0) {
			ontmetadata[1][ontmetadataCount] = BUN_NONE;
		} else {
			ontmetadata[1][ontmetadataCount] = msuperoid;
		}
		ontmetadataCount += 1;

		++i;

		GDKfree(muristr2);
		GDKfree(msuperstr2);
	}

	BBPreclaim(auri);
	BBPreclaim(aattr);
	BBPreclaim(muri);
	BBPreclaim(msuper);

	TKNZRclose(ret);

	*ret = 1;

	return MAL_SUCCEED;
}


