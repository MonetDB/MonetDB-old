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
#include <rdftypes.h>
#include <rdfparser.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <ctype.h>
#include <rdfparams.h>

typedef struct graphBATdef {
	graphBATType batType;    /* BAT type             */
	str name;                /* name of the BAT      */
	int headType;            /* type of left column  */
	int tailType;            /* type of right column */
} graphBATdef;

//static BUN batsz = 10000000;

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


/*
static void 
rdf_BUNappend_unq(parserData* pdata, BAT *b, void* value, BUN* bun){
	
	*bun = BUNfnd(b,(ptr) (str)value);
	if (*bun == BUN_NONE) {
		if (BATcount(b) > 4 * b->T->hash->mask) {
			HASHdestroy(b);
			BAThash(b, 2*BATcount(b));
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
	if (BUNappend(b, bun, TRUE) == GDK_FAIL){
		pdata->exception++;
		pdata->exceptionMsg =  "could not append to a BAT with rdf_BUNappend";
		raptor_parser_parse_abort (pdata->rparser);
	}

}


static void
rdf_BUNappend_BlankNode_Obj(parserData* pdata, BAT *b, BUN* bun){
	*bun |= (BUN)BLANKNODE << (sizeof(BUN)*8 - 4);		//Blank node	
	if (BUNappend(b, bun, TRUE) == GDK_FAIL){
		pdata->exception++;
		pdata->exceptionMsg =  "could not append to a BAT with rdf_BUNappend_BlankNode_Obj";
		raptor_parser_parse_abort (pdata->rparser);
	}

}

/* For inserting an literal object value of RDF triple */

static void 
rdf_BUNappend_unq_ForObj(parserData* pdata, BAT *b, void* objStr, ObjectType objType, BUN* bun){

	*bun = BUNfnd(b,(ptr) (str)objStr);
	if (*bun == BUN_NONE) {
		if (b->T->hash && BATcount(b) > 4 * b->T->hash->mask) {
			HASHdestroy(b);
			BAThash(b, 2*BATcount(b));
		}
		
		*bun = (BUN) (RDF_MIN_LITERAL + (b)->batCount);
	
		/* Add the type here by changing 2 bits at position 62, 63 of oid */
		*bun |= (BUN)objType << (sizeof(BUN)*8 - 4);
		
		//b = BUNappend(b, (ptr) (str)objStr, TRUE);
		if (BUNins(b, (ptr) bun, (ptr) (str)objStr, TRUE) == GDK_FAIL){

			pdata->exception++;
			pdata->exceptionMsg =  "could not append in Object bat";
			raptor_parser_parse_abort (pdata->rparser);
		}
	} else {
		*bun += RDF_MIN_LITERAL;
		*bun |= (BUN)objType << (sizeof(BUN)*8 - 4);
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
getObjectType_and_Value(unsigned char* objStr, ValPtr vrPtrRealValue){
	ObjectType obType = STRING; 
	unsigned char* endpart;
	char* valuepart; 
	const char* pos = NULL; 
	int	len = 0; 
	int	subLen = 0; 

        double  realDbl;
        int     realInt;
	lng 	nTime; 
	lng	realLng; 

	len = strlen((str)objStr);
	if (len > 100 || len <= 30){
		//String is too long for any kind of data or there is no XMLschema
		obType = STRING; 
	}
	else{ //(len > 20)
		endpart = objStr + (len - 29);   /* XMLSchema#dateTime> */
		//printf("Original: %s  --> substring: %s \n", (str)objStr, (str)endpart); 

		if ( (pos = strstr((str)endpart , "XMLSchema#date>")) != NULL || (pos = strstr((str)endpart, "XMLSchema#dateTime>")) != NULL ){
			obType = DATETIME;
			
			subLen = (int) (pos - (str)objStr - 28);
			valuepart = substring((char*)objStr, 2 , subLen); 
			
			if (convertDateTimeToLong(valuepart, &nTime) == 1){

				//
				//
				//store numeric datetime value in long value
				realLng = nTime; 
				VALset(vrPtrRealValue,TYPE_lng, &realLng);
				//printf("Real numeric value of datetime %s is: %lld [vs %ld] \n", valuepart, vrPtrRealValue->val.lval,nTime);
			}
			else 
				obType = STRING;	

			GDKfree(valuepart);

		}
		else if ((pos = strstr((str) endpart, "XMLSchema#int>")) != NULL 
			  || (pos = strstr((str)endpart, "XMLSchema#integer>")) != NULL
			  || (pos = strstr((str)endpart, "XMLSchema#nonNegativeInteger>")) != NULL) {
			//TODO: Consider nonNegativeInteger
			obType = INTEGER;
			subLen = (int) (pos - (str)objStr - 28);
			valuepart = substring((char*)objStr, 2 , subLen); 
			/* printf("%s: Integer \n. Length of value %d ==> value %s \n", objStr, (int) (pos - (str)objStr - 28), valuepart); */
			if (isInt(valuepart, subLen) == 1){	/* Check whether the real value is an integer */
				realInt = (BUN) atoi(valuepart); 
				VALset(vrPtrRealValue,TYPE_int, &realInt);
				//printf("Real int value is: %d \n", vrPtrRealValue->val.ival);
			}
			else 
				obType = STRING;	

			GDKfree(valuepart); 

		}
		else if ((pos = strstr((str) endpart, "XMLSchema#float>")) != NULL 
				|| (pos = strstr((str) endpart, "XMLSchema#double>")) != NULL  
				|| (pos = strstr((str) endpart, "XMLSchema#decimal>")) != NULL){
			obType = DOUBLE;
			subLen = (int) (pos - (str)objStr - 28);
			valuepart = substring((char*)objStr, 2 , subLen);
			if (isDouble(valuepart, subLen) == 1){
				realDbl = atof(valuepart);
				VALset(vrPtrRealValue,TYPE_dbl, &realDbl);
				//printf("Real double value is: %.10f \n", vrPtrRealValue->val.dval);
			}
			else
				obType = STRING;

			GDKfree(valuepart);
		}
		else {
			obType = STRING;
			/* printf("%s: String \n", objStr); */
		}
	}

	return obType; 
}

#if BUILD_ONTOLOGIES_HISTO
static
void printHistogram(parserData *pdata){
	str	value; 
	int	total = 0; 
	int	i;
	int	numOnt = (int)BATcount(pdata->ontBat); 
	BATiter	onti; 
	
	(void) value;
	//BATprint(pdata->ontBat);
	onti = bat_iterator(pdata->ontBat); 
	for (i = 0; i < numOnt; i++){
		value = (str)BUNtail(onti, i); 
		printf("%s: %d \n", value, pdata->numOntInstances[i]); 
		total +=  pdata->numOntInstances[i];
	}
	total += pdata->numNonOnt;
	printf("Number of non-ontology data: %d / %d (%f) \n", pdata->numNonOnt, total, (float)pdata->numNonOnt/total);
}
#endif


/*
 * @-
 * The raptor parser needs to register a callback function that handles one triple
 * at a time. Function rdf_parser_triple_handler() does exactly this.
 */

static void 
tripleHandler(void* user_data, const raptor_statement* triple)
{
#if CHECK_NUM_DBPONTOLOGY
	const char* pos = NULL;
#endif
#if BUILD_ONTOLOGIES_HISTO
	char* 	ontpart;
	char*	ptrEndOnt; 
	int	ontlen = 0; 
	BUN	bunOnt;
#endif
	parserData *pdata = ((parserData *) user_data);
	BUN bun = BUN_NONE;
	ValRecord vrRealValue; 

	BAT **graph = pdata->graph;

	//printf("%s   %s   %s\n",raptor_term_to_string(triple->subject),raptor_term_to_string(triple->predicate),raptor_term_to_string(triple->object));
	if (pdata->error > pdata->lasterror){
		unsigned char* objStr;
		int objLen; 
		//printf("[Incorrect or wrong syntax triple] %s \n ", pdata->errorMsg);
		pdata->lasterror = pdata->error; 
		objStr = raptor_term_to_string(triple->object);
		objLen =  strlen((const char*)objStr);
		//printf("Object: %s %d \n", objStr, objLen);
		if (objLen == 2) 
			printf("EMPTY OBJECT STRING \n");
		free(objStr); 
	}
	else{
		if (triple->subject->type == RAPTOR_TERM_TYPE_URI
				|| triple->subject->type == RAPTOR_TERM_TYPE_BLANK) {
			unsigned char* subjectStr; 
			subjectStr = raptor_term_to_string(triple->subject);
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
			#if CHECK_NUM_DBPONTOLOGY
			pos = strstr((str)predicateStr , "http://dbpedia.org/ontology");
			if ( pos != NULL){
				pdata->numOntologyTriples++;
			}

			//Store all ontology prefixes in a BAT, 
			//Extract the prefix from each predicateStr by locating the last # (if available), or /)  --> strrchr
			//Then, look for this prefix in the BAT, must prepare hash for the BAT	
			#endif
			#if BUILD_ONTOLOGIES_HISTO
			ptrEndOnt = NULL;
			ptrEndOnt = strrchr((str)predicateStr, '#');
			if (ptrEndOnt == NULL) ptrEndOnt = strrchr((str)predicateStr, '/');
			if (ptrEndOnt == NULL) {
				raptor_exception(pdata, "Could not get the ontology");
			}
			
			if (ptrEndOnt != NULL){
				ontlen = (int) (ptrEndOnt - (str)predicateStr);

				ontpart = substring((char*)predicateStr, 1, ontlen); 
				//printf("Ontpart of %s is %s \n", (str)predicateStr, (str)ontpart);
				(void) ontpart; 

				//Check whether ontpart appear in the ontBat
				bunOnt = BUNfnd(pdata->ontBat,(ptr) (str)ontpart);	
				if (bunOnt == BUN_NONE){
					pdata->numNonOnt++;
				}
				else{
					pdata->numOntInstances[(int)bunOnt]++;
				}

				GDKfree(ontpart); 
			}
			else{
				pdata->numNonOnt++;
			}
			#endif

			rdf_tknzr_insert((str) predicateStr, &bun);
			rdf_BUNappend(pdata, graph[P_sort], &bun); 

			bun = BUN_NONE;
			free(predicateStr);
		} else {
			raptor_exception(pdata, "could not determine type of property");
		}

		if (triple->object->type == RAPTOR_TERM_TYPE_URI) {
			unsigned char* objStr;
			objStr = raptor_term_to_string(triple->object);
			rdf_tknzr_insert((str) objStr, &bun);
			rdf_BUNappend(pdata, graph[O_sort], &bun); 
#if 	CHECK_NUM_VALUES_PER_TYPE
			pdata->numValuesPertype[URI]++;
#endif 			
			bun = BUN_NONE;
			free(objStr);
		} else if (triple->object->type == RAPTOR_TERM_TYPE_BLANK) {
			unsigned char* objStr;
			objStr = raptor_term_to_string(triple->object);
			rdf_tknzr_insert((str) objStr, &bun);
			rdf_BUNappend_BlankNode_Obj(pdata, graph[O_sort], &bun); 
			//rdf_BUNappend(pdata, graph[O_sort], &bun); 

#if 	CHECK_NUM_VALUES_PER_TYPE
			pdata->numValuesPertype[BLANKNODE]++;
#endif 			
			bun = BUN_NONE;
			free(objStr);
		
		} else if (triple->object->type == RAPTOR_TERM_TYPE_LITERAL) {
			unsigned char* objStr;
			ObjectType objType = STRING;
			objStr = raptor_term_to_string(triple->object);
			objType = getObjectType_and_Value(objStr, &vrRealValue);

			if (objType == STRING){
				rdf_BUNappend_unq_ForObj(pdata, graph[MAP_LEX], (str)objStr, objType, &bun);	

			}
			else{	//For handling dateTime, Integer, Float values
				encodeValueInOid(&vrRealValue, objType, &bun);
			}

#if 	CHECK_NUM_VALUES_PER_TYPE
			pdata->numValuesPertype[objType]++;
#endif 			
			rdf_BUNappend(pdata, graph[O_sort], &bun); 

			VALclear(&vrRealValue);
			
			/*
			if (objType == INTEGER){
				decodeValueFromOid(bun, objType, &vrRealValue);
				printf("Decoded integer value is: %d \n", vrRealValue.val.ival);
			}
			if (objType == DOUBLE){
				decodeValueFromOid(bun, objType, &vrRealValue);
				printf("Decoded double value is: %.10f \n", vrRealValue.val.dval);
			}
			if (objType == DATETIME){
				decodeValueFromOid(bun, objType, &vrRealValue);
				printf("Decoded numeric datetime value is: %lld \n", vrRealValue.val.lval);

				{
				timestamp ts; 
				convert_encodedLng_toTimestamp(vrRealValue.val.lval, &ts); 
				printf("Msecs is %d (seconds) \n",ts.msecs);
				printf("Days is %d (days) \n", ts.days);
				}
			}
			*/
			//printf("Object string is %s --> object type is %d (oid = " BUNFMT " \n",objStr,objType, bun);

			bun = BUN_NONE;
			free(objStr);
		} else {
			raptor_exception(pdata, "could not determine type of object");

		}

		pdata->tcount++;
		if (pdata->tcount % 500000 == 0) printf(".");
	}

	return;
}

/*
 * @-
 * Function RDFParser() is the entry point to parse an RDF document.
 */


/* creates a BAT for the triple table */
static BAT*
create_BAT(int tt, int size)
{
	BAT *b = BATnew(TYPE_void, tt, size, TRANSIENT);
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
parserData_create (str location, BAT** graph, bat *ontbatid)
{
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
#if CHECK_NUM_DBPONTOLOGY
	pdata->numOntologyTriples = 0;
#endif

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
				graphdef[i].tailType,
				batsz);                       /* DOTO: estimate size */
		if (pdata->graph[i] == NULL) {
			return NULL;
		}
	}

	/* create the MAP_LEX BAT */
	pdata->graph[MAP_LEX] = create_BAT (
			graphdef[MAP_LEX].tailType,
			batsz);                           /* DOTO: estimate size */
	if (pdata->graph[MAP_LEX] == NULL) {
		return NULL;
	}
	/* MAP_LEX must have the key property */
	BATseqbase(pdata->graph[MAP_LEX], RDF_MIN_LITERAL);
	//printf("SEQBASE is "BUNFMT "\n",(pdata->graph[MAP_LEX])->hseqbase);
	pdata->graph[MAP_LEX]->tkey = BOUND2BTRUE;
	pdata->graph[MAP_LEX]->T->nokey[0] = 0;
	pdata->graph[MAP_LEX]->T->nokey[1] = 0;

	/* Reset the dense property of graph[MAP_LEX] */
	pdata->graph[MAP_LEX]->hdense = FALSE;

	(void) ontbatid; 
	#if BUILD_ONTOLOGIES_HISTO
	if (ontbatid == NULL){
		printf("There is no ontology list loaded \n"); 
		return NULL; 
	}
	else{
		if ((pdata->ontBat = BATdescriptor(*ontbatid)) == NULL) {
			return NULL; 
		}
		(void)BAThash(pdata->ontBat,0);
		if (!(pdata->ontBat->T->hash)){
			return NULL; 
		}

		printf("Ontology list contains "BUNFMT" ontologies \n ", BATcount(pdata->ontBat)); 

		pdata->numOntInstances = GDKmalloc(sizeof(int) * BATcount(pdata->ontBat));
		for (i = 0; i < (int)BATcount(pdata->ontBat); i++){
			pdata->numOntInstances[i] = 0; 
		}

		pdata->numNonOnt = 0;
	}
	#endif
#if     CHECK_NUM_VALUES_PER_TYPE
	for (i = 0; i < MULTIVALUES; i++){
		pdata->numValuesPertype[i] = 0;
	}
#endif	
	return pdata;
}

static 
void freeParserData(parserData *pdata){
	
	#if BUILD_ONTOLOGIES_HISTO
	BBPunfix(pdata->ontBat->batCacheid);
	GDKfree(pdata->numOntInstances);
	#endif
	GDKfree(pdata);

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
	#if IS_COMPACT_TRIPLESTORE == 0
	clock_t beginT, endT; 	
	#endif
	BAT *map_oid = NULL, *S = NULL, *P = NULL, *O = NULL;
	BAT **graph = pdata->graph;
#if STORE == TRIPLE_STORE
	BAT *o1,*o2,*o3;
	BAT *g1,*g2,*g3;
#endif
#ifdef _TKNZR_H

	//BATiter bi, mi;
	//BUN p, d, r;
	//oid *bt;

	/* order MAP_LEX */

	/* Do not order the MAP_LEX BAT */
	/* This piece of code is for ordering the dictionary of literal
	 * and re-assign the oid for objects in graph[O_sort] */
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
#else
	BUN cnt; 
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
	if (BATsubsort(&graph[S_sort], &o1, &g1, S, NULL, NULL, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[P_PO], &o2, &g2, P, o1, g1, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[O_PO], &o3, &g3, O, o2, g2, 0, 0) == GDK_FAIL)
		goto bailout;
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);
	
	#if IS_COMPACT_TRIPLESTORE == 0
	beginT = clock(); 

	if (BATsubsort(&graph[O_OP], &o2, &g2, O, o1, g1, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[P_OP], &o3, &g3, P, o2, g2, 0, 0) == GDK_FAIL)
		goto bailout;
	BBPunfix(o1->batCacheid);
	BBPunfix(g1->batCacheid);
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);

	/* order PSO/POS */
	if (BATsubsort(&graph[P_sort], &o1, &g1, P, NULL, NULL, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[S_SO], &o2, &g2, S, o1, g1, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[O_SO], &o3, &g3, O, o2, g2, 0, 0) == GDK_FAIL)
		goto bailout;
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);
	if (BATsubsort(&graph[O_OS], &o2, &g2, O, o1, g1, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[S_OS], &o3, &g3, S, o2, g2, 0, 0) == GDK_FAIL)
		goto bailout;
	BBPunfix(o1->batCacheid);
	BBPunfix(g1->batCacheid);
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);

	/* order OPS/OSP */
	if (BATsubsort(&graph[O_sort], &o1, &g1, O, NULL, NULL, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[P_PS], &o2, &g2, P, o1, g1, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[S_PS], &o3, &g3, S, o2, g2, 0, 0) == GDK_FAIL)
		goto bailout;
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);
	if (BATsubsort(&graph[S_SP], &o2, &g2, S, o1, g1, 0, 0) == GDK_FAIL)
		goto bailout;
	if (BATsubsort(&graph[P_SP], &o3, &g3, P, o2, g2, 0, 0) == GDK_FAIL)
		goto bailout;
	BBPunfix(o1->batCacheid);
	BBPunfix(g1->batCacheid);
	BBPunfix(o2->batCacheid);
	BBPunfix(g2->batCacheid);
	BBPunfix(o3->batCacheid);
	BBPunfix(g3->batCacheid);

	endT = clock(); 
	printf (" Sorting remaining tables in post_processing took %f seconds.\n", ((float)(endT - beginT))/CLOCKS_PER_SEC);
	
	#endif /* IS_COMPACT_TRIPLESTORE == 0 */

	/* free memory */
	BBPunfix(S->batCacheid);
	
	#if IS_COMPACT_TRIPLESTORE == 0
	BBPunfix(P->batCacheid);
	BBPunfix(O->batCacheid);
	#endif /* IS_COMPACT_TRIPLESTORE == 0 */


	return MAL_SUCCEED;

bailout:
	if (map_oid != NULL) BBPreclaim(map_oid);
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
		freeParserData(pdata);
	}
}


/* Main RDF parser function that drives raptor */
str
RDFParser (BAT **graph, str *location, str *graphname, str *schema, bat *ontbatid)
{
	raptor_parser *rparser;
	parserData *pdata;
	raptor_uri *uri;
	bit isURI;
	str ret;
	int iret;
	raptor_world *world; 

	struct stat s;
	DIR *dp;
       	struct dirent *ep;
	//char *pch; 
	char tmpfilename[200];
	clock_t tmpbeginT, tmpendT; 

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
	pdata = parserData_create(*location,graph, ontbatid);
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
	pdata->rparser = rparser = raptor_new_parser(world,"guess");
	//pdata->rparser = rparser = raptor_new_parser(world,"turtle");

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
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) warningHandler);
	raptor_world_set_log_handler          (world,  pdata, (raptor_log_handler) errorHandler);

	//raptor_parser_set_option(rparser, 0);	//MDPHAM: CHECK FOR THIS SETTING
	//raptor_parser_set_option(rparser, RAPTOR_OPTION_SCANNING,"RAPTOR_OPTION_SCANNING",1);

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
	} else {	// A dir or a file
		if( stat(pdata->location,&s) == 0 ){
			if( s.st_mode & S_IFDIR ){
				//it's a directory
				printf("Load directory %s \n",pdata->location);
				// Go through each file
				dp = opendir (pdata->location);
				if (dp != NULL){
					while ((ep = readdir (dp)) != NULL){
						printf("Checking file %s \n",ep->d_name);	
						if (strstr (ep->d_name,".nt")!= NULL || strstr (ep->d_name,".ttl")!= NULL ){
							tmpbeginT = clock(); 
							sprintf(tmpfilename,"%s%s",pdata->location,ep->d_name);
							printf("Loading file %s ..",tmpfilename);
							uri = raptor_new_uri(world,raptor_uri_filename_to_uri_string(tmpfilename));
							iret = raptor_parser_parse_file(rparser, uri, NULL);
							raptor_free_uri(uri);
							#if CHECK_NUM_DBPONTOLOGY
							printf(".. Done (No errors: %d | Loaded: " BUNFMT " | Ontology-based: %d) \n",pdata->error, pdata->tcount, pdata->numOntologyTriples);
							#else
							printf(".. Done (No errors: %d | Loaded: " BUNFMT ") \n",pdata->error, pdata->tcount);
							#endif
							tmpendT = clock(); 
							printf ("  Loading took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);
						}
					}
					closedir (dp);
				}
			}
			else if( s.st_mode & S_IFREG )	{
				//it's a file
				tmpbeginT = clock();
				printf("Load file %s \n", pdata->location);
				uri = raptor_new_uri(world,
						raptor_uri_filename_to_uri_string(pdata->location));
				iret = raptor_parser_parse_file(rparser, uri, NULL);

				raptor_free_uri(uri);
				tmpendT = clock();
				printf ("  Loading took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);
			}
			else{
				printf("This is a not a file/directory or uri \n");
				TKNZRclose(&iret);
				clean(pdata);
			}
		}


	}
	
	/* Free memory of raptor */
	raptor_free_parser(rparser);
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
	
#if	NOT_IGNORE_ERROR_TRIPLE	
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

#else
	#if CHECK_NUM_DBPONTOLOGY
	printf("Total number of triples loaded: " BUNFMT " (Number of available ontology-based triples: %d) \n", pdata->tcount, pdata->numOntologyTriples);
	#else
	printf("Total number of triples loaded: " BUNFMT "\n", pdata->tcount);
	#endif
	printf("Total number of error %d , fatal %d , warning %d \n", pdata->error, pdata->fatal, pdata->warning);
#endif
	#if	BUILD_ONTOLOGIES_HISTO
	printHistogram(pdata);
	#endif
#if 	CHECK_NUM_VALUES_PER_TYPE
	printf("Number of URI %d \n", pdata->numValuesPertype[URI]);
	printf("Number of DATETIME %d \n", pdata->numValuesPertype[DATETIME]);
	printf("Number of INTEGER %d \n", pdata->numValuesPertype[INTEGER]);
	printf("Number of DOUBLE %d \n", pdata->numValuesPertype[DOUBLE]);
	printf("Number of BLANKNODE %d \n", pdata->numValuesPertype[BLANKNODE]);
#endif 			
	/* post processing step */
	tmpbeginT = clock();
	ret = post_processing(pdata);
	if (ret != MAL_SUCCEED) {
		clean(pdata);
		throw(RDF, "rdf.rdfShred", "could not post-proccess data");
	}
	tmpendT = clock();
	printf ("Post processing took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

	//Create default paramters file 
	createDefaultParamsFile();

	freeParserData(pdata);
	return MAL_SUCCEED;
}


