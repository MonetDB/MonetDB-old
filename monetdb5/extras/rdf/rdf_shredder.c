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

/*
 * @a L.Sidirourgos, Minh-Duc Pham
 *
 * @+ Shredder for RDF Documents
 */
#include "monetdb_config.h"
#include "mal_exception.h"
#include "url.h"
#include "tokenizer.h"
#include <gdk.h>
#include <rdf.h>
#include <raptor2.h>

typedef struct graphBATdef {
	graphBATType batType;    /* BAT type             */
	str name;                /* name of the BAT      */
	int headType;            /* type of left column  */
	int tailType;            /* type of right column */
} graphBATdef;

static BUN batsz = 10000000;

/* this list should be kept alligned with the graphBATType enum */
#if STORE == TRIPLE_STORE
 static graphBATdef graphdef[N_GRAPH_BAT] = {
	{S_sort,   "_s_sort",   TYPE_void, TYPE_oid},
	{P_sort,   "_p_sort",   TYPE_void, TYPE_oid},
	{O_sort,   "_o_sort",   TYPE_void, TYPE_oid},

	{P_PO,     "_p_po",     TYPE_void, TYPE_oid},
	{O_PO,     "_o_po",     TYPE_void, TYPE_oid},
	{P_OP,     "_p_op",     TYPE_void, TYPE_oid},
	{O_OP,     "_o_op",     TYPE_void, TYPE_oid},

	{S_SO,     "_s_so",     TYPE_void, TYPE_oid},
	{O_SO,     "_o_so",     TYPE_void, TYPE_oid},
	{S_OS,     "_s_os",     TYPE_void, TYPE_oid},
	{O_OS,     "_o_os",     TYPE_void, TYPE_oid},

	{S_SP,     "_s_sp",     TYPE_void, TYPE_oid},
	{P_SP,     "_p_sp",     TYPE_void, TYPE_oid},
	{S_PS,     "_s_ps",     TYPE_void, TYPE_oid},
	{P_PS,     "_p_ps",     TYPE_void, TYPE_oid},

	{MAP_LEX, "_map_lex",   TYPE_void, TYPE_str}
 };

#elif STORE == MLA_STORE
static graphBATdef graphdef[N_GRAPH_BAT] = {
	{S_sort,   "_s_sort",   TYPE_void, TYPE_oid},
	{P_sort,   "_p_sort",   TYPE_void, TYPE_oid},
	{O_sort,   "_o_sort",   TYPE_void, TYPE_oid},
	{MAP_LEX, "_map_lex",   TYPE_void, TYPE_str}
};
#endif /* STORE */

typedef struct parserData {
	                              /**PROPERTIES             */
	str location;                 /* rdf data file location */
	oid tcount;                   /* triple count           */
	raptor_parser *rparser;       /* the parser object      */
	                              /**ERROR HANDLING         */
	int exception;                /* raise an exception     */
	int warning;                  /* number of warning msgs */
	int error;                    /* number of error   msgs */
	int fatal;                    /* number of fatal   msgs */
	const char *exceptionMsg;     /* exception msgs         */
	const char *warningMsg;       /* warning msgs           */
	const char *errorMsg;         /* error   msgs           */
	const char *fatalMsg;         /* fatal   msgs           */
	int line;                     /* locator for errors     */
	int column;                   /* locator for errors     */
	                              /**GRAPH DATA             */
	BAT **graph;                  /* BATs for the result
	                                 shredded RDF graph     */
} parserData;

/*
 * @-
 * The (fatal) errors and warnings produced by the raptor parser are handled
 * by the next three message handler functions.
 */

static void
fatalHandler (void *user_data, raptor_log_message* message)
{
	parserData *pdata = ((parserData *) user_data);
	pdata->fatalMsg = GDKstrdup(message->text);
	mnstr_printf(GDKout, "rdflib: fatal:%s\n", pdata->fatalMsg);
	pdata->fatal++;

	/* check for a valid locator object and only then use it */
	if (message->locator != NULL) {
		pdata->line = message->locator->line;
		pdata->column = message->locator->column;
		mnstr_printf(GDKout, "rdflib: fatal: at line %d column %d\n", pdata->line, pdata->column);
	} 
	
}


static void
errorHandler (void *user_data, raptor_log_message* message)
{
	parserData *pdata = ((parserData *) user_data);
	pdata->errorMsg = GDKstrdup(message->text);
	mnstr_printf(GDKout, "rdflib: error:%s\n", pdata->errorMsg);
	pdata->error++;

	/* check for a valid locator object and only then use it */
	if (message->locator != NULL) {
		pdata->line = message->locator->line;
		pdata->column = message->locator->column;
		mnstr_printf(GDKout, "rdflib: error: at line %d column %d\n", pdata->line, pdata->column);
	} 
	
}


static void
warningHandler (void *user_data, raptor_log_message* message)
{
	parserData *pdata = ((parserData *) user_data);
	pdata->warningMsg = GDKstrdup(message->text);
	mnstr_printf(GDKout, "rdflib: warning:%s\n", pdata->warningMsg);
	pdata->warning++;

	/* check for a valid locator object and only then use it */
	if (message->locator != NULL) {
		pdata->line = message->locator->line;
		pdata->column = message->locator->column;
		mnstr_printf(GDKout, "rdflib: warning: at line %d column %d\n", pdata->line, pdata->column);
	} 
	
}

static void 
raptor_exception(parserData *pdata, const char* msg){
	pdata->exception++;
	pdata->exceptionMsg =  GDKstrdup(msg);
	raptor_parser_parse_abort (pdata->rparser);
}
/*
static void 
rdf_BUNappend_unq(parserData* pdata, BAT *b, void* value, BUN* bun){
	
	*bun = BUNfnd(BATmirror(b),(ptr) (str)value);
	if (*bun == BUN_NONE) {
		if (BATcount(b) > 4 * b->T->hash->mask) {
			HASHdestroy(b);
			BAThash(BATmirror(b), 2*BATcount(b));
		}
		*bun = (BUN) b->batCount;
		b = BUNappend(b, (ptr) (str)value, TRUE);
		if (b == NULL) {
			pdata->exception++;
			pdata->exceptionMsg =  "could not append to a BAT";
			raptor_parser_parse_abort (pdata->rparser);
		}
	}
}
*/

static void 
rdf_tknzr_insert(void* value, BUN* bun)
{
	str t = (str) value;
	TKNZRappend(bun,&t);
}

/*
static void
rdf_insert(parserData* pdata, BAT *b, void* value, BUN* bun){

	#ifdef _TKNZR_H
	rdf_tknzr_insert(value, bun);
	#else
	rdf_BUNappend_unq(pdata, b, value, bun);
	#endif
}
*/

static void
rdf_BUNappend(parserData* pdata, BAT *b, BUN* bun){
	b = BUNappend(b, bun, TRUE);
	if (b == NULL) {
		pdata->exception++;
		pdata->exceptionMsg =  "could not append to a BAT with rdf_BUNappend";
		raptor_parser_parse_abort (pdata->rparser);
	}

}

/* For inserting an literal object value of RDF triple */

static void 
rdf_BUNappend_unq_ForObj(parserData* pdata, BAT *b, void* objStr, ObjectType objType, BUN* bun){

	*bun = BUNfnd(BATmirror(b),(ptr) (str)objStr);
	if (*bun == BUN_NONE) {
		if (b->T->hash && BATcount(b) > 4 * b->T->hash->mask) {
			HASHdestroy(b);
			BAThash(BATmirror(b), 2*BATcount(b));
		}
		
		*bun = (BUN) (RDF_MIN_LITERAL + (b)->batCount);
	
		/* Add the type here by changing 2 bits at position 62, 63 of oid */
		if ( objType == DATETIME){ 
			printf("Datetime appears here \n Before: " BUNFMT "\n", *bun);
			*bun |= (BUN)1 << (sizeof(BUN)*8 - 3);
			printf("After: " BUNFMT "\n", *bun);
		}
		else if ( objType == NUMERIC){
			printf("Numeric value appears here \n Before: " BUNFMT "\n", *bun);
			*bun |= (BUN)2 << (sizeof(BUN)*8 - 3);
			printf("After: " BUNFMT "\n", *bun);
		}
		else { /*  objType == STRING */
			printf("String value appears here \n Before: " BUNFMT "\n", *bun);
			*bun |= (BUN)3 << (sizeof(BUN)*8 - 3);
			printf("After: " BUNFMT "\n", *bun);
		}

		//b = BUNappend(b, (ptr) (str)objStr, TRUE);
		b = BUNins(b, (ptr) bun, (ptr) (str)objStr, TRUE); 

		if (b == NULL) {
		
			pdata->exception++;
			pdata->exceptionMsg =  "could not append in Object bat";
			raptor_parser_parse_abort (pdata->rparser);
		}
	} else {
		*bun = (b)->hseqbase + *bun;
	}

}


/*
* Get the specific type of the object value in an RDF triple
* The URI object can be recoginized by raptor parser. 
* If the object value is not an URI ==> it is a literal, and 
* specifically, a numeric, a dateTime or a string. 
* This function will find the specific type of Object value
*/

static ObjectType 
getObjectType(unsigned char* objStr){
	ObjectType obType; 
	if (strstr((const char*) objStr, "XMLSchema#date") != NULL){
		obType = DATETIME;
		printf("%s: DateTime \n", objStr); 
	}
	else if (strstr((const char*) objStr, "XMLSchema#float") != NULL
		|| strstr((const char*) objStr, "XMLSchema#integer") != NULL
		)
	{
		obType = NUMERIC;
		printf("%s: Numeric \n", objStr); 
	}
	else {
		obType = STRING;
		printf("%s: String \n", objStr); 
	}

	return obType; 
}


/*
 * @-
 * The raptor parser needs to register a callback function that handles one triple
 * at a time. Function rdf_parser_triple_handler() does exactly this.
 */

static void 
tripleHandler(void* user_data, const raptor_statement* triple)
{
	parserData *pdata = ((parserData *) user_data);
	BUN bun = BUN_NONE;
	BAT **graph = pdata->graph;

	if (triple->subject->type == RAPTOR_TERM_TYPE_URI
			|| triple->subject->type == RAPTOR_TERM_TYPE_BLANK) {
		unsigned char* subjectStr; 
		subjectStr = raptor_term_to_string(triple->subject);
		//rdf_insert(pdata, graph[MAP_LEX], (str) subjectStr, &bun);
		rdf_tknzr_insert((str) subjectStr, &bun);
		rdf_BUNappend(pdata, graph[S_sort], &bun); 
			
		bun = BUN_NONE;
		free(subjectStr);
	} else {
		raptor_exception(pdata, "could not determine type of subject");
	}

	if (triple->predicate->type == RAPTOR_TERM_TYPE_URI) {
		unsigned char* predicateStr;
		predicateStr = raptor_term_to_string(triple->predicate);
		//rdf_insert(pdata, graph[MAP_LEX], (str) predicateStr, &bun);
		rdf_tknzr_insert((str) predicateStr, &bun);
		rdf_BUNappend(pdata, graph[P_sort], &bun); 

		bun = BUN_NONE;
		free(predicateStr);
	} else {
		raptor_exception(pdata, "could not determine type of property");
	}

	if (triple->object->type == RAPTOR_TERM_TYPE_URI
			|| triple->object->type == RAPTOR_TERM_TYPE_BLANK) {
		unsigned char* objStr;
		objStr = raptor_term_to_string(triple->object);
		//rdf_insert(pdata, graph[MAP_LEX], (str) objStr, &bun);
		rdf_tknzr_insert((str) objStr, &bun);
		rdf_BUNappend(pdata, graph[O_sort], &bun); 

		bun = BUN_NONE;
		free(objStr);
	} else if (triple->object->type == RAPTOR_TERM_TYPE_LITERAL) {
		unsigned char* objStr;
		ObjectType objType;
		objStr = raptor_term_to_string(triple->object);
		objType = getObjectType(objStr);

		rdf_BUNappend_unq_ForObj(pdata, graph[MAP_LEX], (str)objStr, objType, &bun);	
		rdf_BUNappend(pdata, graph[O_sort], &bun); 

		bun = BUN_NONE;
		free(objStr);
	} else {
		raptor_exception(pdata, "could not determine type of object");

	}

	pdata->tcount++;

	return;
}

/*
 * @-
 * Function RDFParser() is the entry point to parse an RDF document.
 */


/* creates a BAT for the triple table */
static BAT*
create_BAT(int ht, int tt, int size)
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
parserData_create (str location, BAT** graph)
{
	int i;

	parserData *pdata = (parserData *) GDKmalloc(sizeof(parserData));
	if (pdata == NULL) return NULL;

	pdata->tcount = 0;
	pdata->exception = 0;
	pdata->fatal = 0;
	pdata->error = 0;
	pdata->warning = 0;
	pdata->location = location;
	pdata->graph = graph;

	for (i = 0; i <= N_GRAPH_BAT; i++) {
		pdata->graph[i] = NULL;
	}

	/* New empty BATs for shredding. We only reserve memory for
	 * S_sort, P_sort, O_sort and MAP_LEX in this stage, since these
	 * are the ones to be populated now, while the rest will be
	 * created in a post-shredding processing step
	 */
	for (i = 0; i <= O_sort; i++) {
		pdata->graph[i] = create_BAT (
				graphdef[i].headType,
				graphdef[i].tailType,
				batsz);                       /* DOTO: estimate size */
		if (pdata->graph[i] == NULL) {
			return NULL;
		}
	}

	/* create the MAP_LEX BAT */
	pdata->graph[MAP_LEX] = create_BAT (
			graphdef[MAP_LEX].headType,
			graphdef[MAP_LEX].tailType,
			batsz);                           /* DOTO: estimate size */
	if (pdata->graph[MAP_LEX] == NULL) {
		return NULL;
	}
	/* MAP_LEX must have the key property */
	BATseqbase(pdata->graph[MAP_LEX], RDF_MIN_LITERAL);
	pdata->graph[MAP_LEX]->tkey = BOUND2BTRUE;
	pdata->graph[MAP_LEX]->T->nokey[0] = 0;
	pdata->graph[MAP_LEX]->T->nokey[1] = 0;

	/* Reset the dense property of graph[MAP_LEX] */
	pdata->graph[MAP_LEX]->hdense = FALSE;

	return pdata;
}



/*
 * @-
 * After the RDF document has been shredded into 3 bats and a lexical value
 * dictionary, a post-shred processing step follows that orders the lexical
 * dictionary, re-maps oids to match the ordered dictionary and finaly creates
 * all 6 permutations of the (subject, predicate, object) order.
 *
 * However, it is still to be examined if it worth the time to refine the order
 * of the last column. In most cases, during query time, the last column will need
 * to be re-order for a subsequent sort-merge join. We introduce sort3 and sort2
 * so we can investigate both possibilities. In addition, the first column need to
 * be stored only once for each couple of orders with the same first column. For
 * example, it holds that S_SPO == S_SOP.
 */

int CTrefine(BAT **ret, BAT *b, BAT *a); /* from modules/kernel/group.mx */



static str
post_processing (parserData *pdata)
{
	BAT *map_oid = NULL, *S = NULL, *P = NULL, *O = NULL;
	BAT **graph = pdata->graph;
	BUN cnt;
#if STORE == TRIPLE_STORE
	BAT *ctref= NULL;
#endif
#ifdef _TKNZR_H

	//BATiter bi, mi;
	//BUN p, d, r;
	//oid *bt;

	/* order MAP_LEX */

	/* Do not order the MAP_LEX BAT */
	#ifdef ORDER_MAPLEX
	BATorder(BATmirror(graph[MAP_LEX]));
	map_oid = BATmark(graph[MAP_LEX], RDF_MIN_LITERAL);   /* BATmark will create a copy */
	BATorder(map_oid);
	BATsetaccess(map_oid, BAT_READ);        /* force BAtmark not to copy bat */
	map_oid = BATmirror(BATmark(BATmirror(map_oid), RDF_MIN_LITERAL));

	BATsetaccess(graph[MAP_LEX], BAT_READ); /* force BATmark not to copy bat */
	graph[MAP_LEX] = BATmirror(BATmark(BATmirror(graph[MAP_LEX]), RDF_MIN_LITERAL));

	/* convert old oids of O_sort to new ones */
	bi = bat_iterator(graph[O_sort]);
	mi = bat_iterator(map_oid);
	BATloop(graph[O_sort], p, d) {
		bt = (oid *) BUNtloc(bi, p);
		if (*bt >= (RDF_MIN_LITERAL)) {
			BUNfndVOID(r, mi, bt);
			void_inplace(graph[O_sort], p, BUNtloc(mi, r), 1);
		}
	}
	BBPreclaim(map_oid);

	#endif

	S = graph[S_sort];
	P = graph[P_sort];
	O = graph[O_sort];
	cnt = BATcount(S);
#else
	/* order MAP_LEX */
	BATorder(BATmirror(graph[MAP_LEX]));
	map_oid = BATmark(graph[MAP_LEX], 0);   /* BATmark will create a copy */
	BATorder(map_oid);
	BATsetaccess(map_oid, BAT_READ);        /* force BAtmark not to copy bat */
	map_oid = BATmirror(BATmark(BATmirror(map_oid), 0));
	BATsetaccess(graph[MAP_LEX], BAT_READ); /* force BATmark not to copy bat */
	graph[MAP_LEX] = BATmirror(BATmark(BATmirror(graph[MAP_LEX]), 0));

	/* convert old oids of S_sort, P_sort, O_sort to new ones */
	cnt = BATcount(graph[S_sort]);
	S = BATleftfetchjoin(graph[S_sort], map_oid, cnt);
	if (S == NULL) goto bailout;
	BBPreclaim(graph[S_sort]);
	P = BATleftfetchjoin(graph[P_sort], map_oid, cnt);
	if (P == NULL) goto bailout;
	BBPreclaim(graph[P_sort]);
	O = BATleftfetchjoin(graph[O_sort], map_oid, cnt);
	if (O == NULL) goto bailout;
	BBPreclaim(graph[O_sort]);
	BBPreclaim(map_oid);
#endif

#if STORE == MLA_STORE
	graph[S_sort] = S;
	graph[P_sort] = P;
	graph[O_sort] = O;

	return MAL_SUCCEED;

#elif STORE == TRIPLE_STORE
	/* order SPO/SOP */
	graph[S_sort] = BATmirror(BATsort(BATmirror(S))); /* sort on S */
	
	if ( !(CTrefine(&map_oid, graph[S_sort], P)         /* refine O given graph[S_sort]= sorted  */
		&& CTrefine(&ctref, map_oid, O)))/* refine PO given O          */
		goto bailout;

	BBPreclaim(map_oid);                   /* free map_oid                       */
	map_oid = BATmirror(BATmark(ctref,0)); /* map_oid[void,oid] gives the order  */
	BBPreclaim(ctref);                     /* free o                             */

	/* leftfetchjoin to re-order all BATs */
	graph[P_PO] = BATleftfetchjoin(map_oid, P, cnt);
	if (graph[P_PO] == NULL) goto bailout;
	BBPcold(graph[P_PO]->batCacheid);
	graph[O_PO] = BATleftfetchjoin(map_oid, O, cnt);
	if (graph[O_PO] == NULL) goto bailout;
	BBPcold(graph[O_PO]->batCacheid);
	/* free map_oid */
	BBPreclaim(map_oid);


	if ( !(CTrefine(&map_oid, graph[S_sort], O)         /* refine P given graph[S_sort]= sorted  */
		&& CTrefine(&ctref, map_oid, P)))/* refine OP given P          */
		goto bailout;

	BBPreclaim(map_oid);                   /* free map_oid                       */
	map_oid = BATmirror(BATmark(ctref,0)); /* map_oid[void,oid] gives the order  */
	BBPreclaim(ctref);                     /* free o                             */

	/* leftfetchjoin to re-order all BATs */
	graph[O_OP] = BATleftfetchjoin(map_oid, O, cnt);
	if (graph[O_OP] == NULL) goto bailout;
	BBPcold(graph[O_OP]->batCacheid);
	graph[P_OP] = BATleftfetchjoin(map_oid, P, cnt);
	if (graph[P_OP] == NULL) goto bailout;
	BBPcold(graph[P_OP]->batCacheid);
	/* free map_oid */
	BBPreclaim(map_oid);


	BATsetaccess(graph[S_sort], BAT_READ); /* force BATmark not to copy bat */
	graph[S_sort] = BATmirror(BATmark(BATmirror(graph[S_sort]), 0));

	
	/* order PSO/POS */
	graph[P_sort] = BATmirror(BATsort(BATmirror(P))); /* sort on P */
	

	if ( !(CTrefine(&map_oid, graph[P_sort], S)         /* refine O given graph[P_sort]= sorted  */
		&& CTrefine(&ctref, map_oid, O)))/* refine SO given O          */
		goto bailout;

	BBPreclaim(map_oid);                   /* free map_oid                       */
	map_oid = BATmirror(BATmark(ctref,0)); /* map_oid[void,oid] gives the order  */
	BBPreclaim(ctref);                     /* free o                             */

	/* leftfetchjoin to re-order all BATs */
	graph[S_SO] = BATleftfetchjoin(map_oid, S, cnt);
	if (graph[S_SO] == NULL) goto bailout;
	BBPcold(graph[S_SO]->batCacheid);
	graph[O_SO] = BATleftfetchjoin(map_oid, O, cnt);
	if (graph[O_SO] == NULL) goto bailout;
	BBPcold(graph[O_SO]->batCacheid);
	/* free map_oid */
	BBPreclaim(map_oid);


	if ( !(CTrefine(&map_oid, graph[P_sort], O)         /* refine S given graph[P_sort]= sorted  */
		&& CTrefine(&ctref, map_oid, S)))/* refine OS given S          */
		goto bailout;

	BBPreclaim(map_oid);                   /* free map_oid                       */
	map_oid = BATmirror(BATmark(ctref,0)); /* map_oid[void,oid] gives the order  */
	BBPreclaim(ctref);                     /* free o                             */

	/* leftfetchjoin to re-order all BATs */
	graph[O_OS] = BATleftfetchjoin(map_oid, O, cnt);
	if (graph[O_OS] == NULL) goto bailout;
	BBPcold(graph[O_OS]->batCacheid);
	graph[S_OS] = BATleftfetchjoin(map_oid, S, cnt);
	if (graph[S_OS] == NULL) goto bailout;
	BBPcold(graph[S_OS]->batCacheid);
	/* free map_oid */
	BBPreclaim(map_oid);



	BATsetaccess(graph[P_sort], BAT_READ); /* force BATmark not to copy bat */
	graph[P_sort] = BATmirror(BATmark(BATmirror(graph[P_sort]), 0));


	/* order OPS/OSP */
	graph[O_sort] = BATmirror(BATsort(BATmirror(O))); /* sort on O */
	
	if ( !(CTrefine(&map_oid, graph[O_sort], P)         /* refine S given graph[O_sort]= sorted  */
		&& CTrefine(&ctref, map_oid, S)))/* refine PS given S          */
		goto bailout;

	BBPreclaim(map_oid);                   /* free map_oid                       */
	map_oid = BATmirror(BATmark(ctref,0)); /* map_oid[void,oid] gives the order  */
	BBPreclaim(ctref);                     /* free o                             */

	/* leftfetchjoin to re-order all BATs */
	graph[P_PS] = BATleftfetchjoin(map_oid, P, cnt);
	if (graph[P_PS] == NULL) goto bailout;
	BBPcold(graph[P_PS]->batCacheid);
	graph[S_PS] = BATleftfetchjoin(map_oid, S, cnt);
	if (graph[S_PS] == NULL) goto bailout;
	BBPcold(graph[S_PS]->batCacheid);
	/* free map_oid */
	BBPreclaim(map_oid);


	if ( !(CTrefine(&map_oid, graph[O_sort], S)         /* refine P given graph[O_sort]= sorted  */
		&& CTrefine(&ctref, map_oid, P)))/* refine SP given P          */
		goto bailout;

	BBPreclaim(map_oid);                   /* free map_oid                       */
	map_oid = BATmirror(BATmark(ctref,0)); /* map_oid[void,oid] gives the order  */
	BBPreclaim(ctref);                     /* free o                             */

	/* leftfetchjoin to re-order all BATs */
	graph[S_SP] = BATleftfetchjoin(map_oid, S, cnt);
	if (graph[S_SP] == NULL) goto bailout;
	BBPcold(graph[S_SP]->batCacheid);
	graph[P_SP] = BATleftfetchjoin(map_oid, P, cnt);
	if (graph[P_SP] == NULL) goto bailout;
	BBPcold(graph[P_SP]->batCacheid);
	/* free map_oid */
	BBPreclaim(map_oid);


	BATsetaccess(graph[O_sort], BAT_READ); /* force BATmark not to copy bat */
	graph[O_sort] = BATmirror(BATmark(BATmirror(graph[O_sort]), 0));

	/* free memory */
	BBPunfix(S->batCacheid);
	BBPunfix(P->batCacheid);
	BBPunfix(O->batCacheid);

	return MAL_SUCCEED;

bailout:
	if (map_oid != NULL) BBPreclaim(map_oid);
	if (ctref   != NULL) BBPreclaim(ctref);
	if (S       != NULL) BBPreclaim(S);
	if (P       != NULL) BBPreclaim(P);
	if (O       != NULL) BBPreclaim(O);
	return NULL;
#endif
}


static void
clean(parserData *pdata){
	int iret; 
	if (pdata != NULL) {
		for (iret = 0; iret < N_GRAPH_BAT; iret++) {
			if (pdata->graph[iret] != NULL)
				BBPreclaim(pdata->graph[iret]);
		}

		GDKfree(pdata);
	}
}

/* Main RDF parser function that drives raptor */
str
RDFParser (BAT **graph, str *location, str *graphname, str *schema)
{
	raptor_parser *rparser;
	parserData *pdata;
	raptor_uri *uri;
	bit isURI;
	str ret;
	int iret;
	raptor_world *world; 

	(void) graphname;

	/* init tokenizer */
#ifdef _TKNZR_H
	if (TKNZRopen (NULL, schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfShred",
				"could not open the tokenizer\n");
	}
#else
	(void) schema;
#endif

	/* Init pdata  */
	pdata = parserData_create(*location,graph);
	if (pdata == NULL) {
#ifdef _TKNZR_H
		TKNZRclose(&iret);
#endif
		clean(pdata);		
		throw(RDF, "rdf.rdfShred",
				"could not allocate enough memory for pdata\n");
	}

	/* Init raptor2 */
	world = raptor_new_world();
	//pdata->rparser = rparser = raptor_new_parser(world,"guess");
	pdata->rparser = rparser = raptor_new_parser(world,"turtle");

	if (rparser == NULL) {
#ifdef _TKNZR_H
		TKNZRclose(&iret);
#endif
		raptor_free_world(world);
		
		clean(pdata);

		throw(RDF, "rdf.rdfShred", "could not create raptor parser object\n");
	}
	
	/* set callback handler for triples */
	raptor_parser_set_statement_handler   ( rparser,  pdata,  (raptor_statement_handler) tripleHandler);
	/* set message handlers */
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) fatalHandler);
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) errorHandler);
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) warningHandler);


	//raptor_parser_set_option(rparser, 0);	//MDPHAM: CHECK FOR THIS SETTING
	//raptor_parser_set_option(rparser, RAPTOR_OPTION_SCANNING);

	/* Parse URI or local file. */
	ret = URLisaURL(&isURI, location);
	if (ret != MAL_SUCCEED) {
#ifdef _TKNZR_H
		TKNZRclose(&iret);
#endif
		clean(pdata);	

		return ret;
	} else if (isURI) {
		uri = raptor_new_uri(world, (unsigned char *) pdata->location);
		iret = raptor_parser_parse_uri(rparser, uri, NULL);
	} else {
		
                uri = raptor_new_uri(world,
                                raptor_uri_filename_to_uri_string(pdata->location));
                iret = raptor_parser_parse_file(rparser, uri, NULL);
	}
	
	/* Free memory of raptor */
	raptor_free_parser(rparser);
	raptor_free_uri(uri);
	raptor_free_world(world);

#ifdef _TKNZR_H
	TKNZRclose(&iret);
#endif

	assert (pdata->tcount == BATcount(graph[S_sort]) &&
			pdata->tcount == BATcount(graph[P_sort]) &&
			pdata->tcount == BATcount(graph[O_sort]));

	/* error check */
	if (iret) {
		
		clean(pdata);
		throw(RDF, "rdf.rdfShred", "parsing failed\n");
	}
	if (pdata->exception) {
		throw(RDF, "rdf.rdfShred", "%s\n", pdata->exceptionMsg);
	} else if (pdata->fatal) {
		throw(RDF, "rdf.rdfShred", "last fatal error was:\n%s\n",
				pdata->fatalMsg);
	} else if (pdata->error) {
		throw(RDF, "rdf.rdfShred", "last error was:\n%s\n",
				pdata->errorMsg);
	} else if (pdata->warning) {
		throw(RDF, "rdf.rdfShred", "last warning was:\n%s\n",
				pdata->warningMsg);
	}

	/* post processing step */
	ret = post_processing(pdata);
	if (ret != MAL_SUCCEED) {
		clean(pdata);
		throw(RDF, "rdf.rdfShred", "could not post-proccess data");
	}
	GDKfree(pdata);
	return MAL_SUCCEED;
}

