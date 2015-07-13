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

/* This contains db kernel operators for RDF/MonetRDF */

/*
 * The input query should be in the format 
 * RP1[low1,high1]|RP2[low2,high2]|?OP1|?RP3|.....
 * In which, each RPi or OPi is an URI.
 * The low, high are first parsed as string values.
 * Then, depending the datatype of the corresponding
 * columns we will find the type-specific values for low, high
 */

#include "monetdb_config.h"
#include <gdk.h>
#include "tokenizer.h"
#include "rdf.h"
#include "rdfscan.h"

#define MAX_PARAMS_NO	20

static 
str queryParser(RdfScanParams *rsParam, str query, str schema){

	int paramNo; 
	str parts[MAX_PARAMS_NO]; 
	int i = 0, j; 
	int numRP = 0, numOP = 0; 
	int opIdx, rpIdx;
	int ret; 
	
	(void) schema; 

	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema", "could not open the tokenizer\n");
	}
	
	paramNo = TKNZRtokenize(query, parts, '|');
	
	printf("Number of params from query %s is: %d \n", query, paramNo);

	for (i = 0; i < paramNo; i++){
		if (parts[i][0] == '?'){
			//parts[i]++;
			printf("Param %s is an optional param\n", parts[i]); 
			numOP++; 
		}
		else{
			printf("Param %s is a required param\n", parts[i]);
			numRP++; 
			//Check low, high
		}
	}
	printf("Number of RPs is %d \n", numRP);

	rsParam->numRP = numRP; 
	rsParam->lstRPstr = (char **)GDKmalloc(sizeof(char*) * rsParam->numRP); 
	rsParam->lstLow = (char **)GDKmalloc(sizeof(char*) * rsParam->numRP); 
	rsParam->lstHi = (char **)GDKmalloc(sizeof(char*) * rsParam->numRP); 

	rsParam->numOP = numOP;
	rsParam->lstOPstr = (char **)GDKmalloc(sizeof(char*) * rsParam->numOP); 
	
	opIdx = 0;
	rpIdx = 0; 
	for (i = 0; i < paramNo; i++){
		if (parts[i][0] == '?'){	//optional param
			parts[i]++;			
			rsParam->lstOPstr[opIdx] = GDKstrdup(parts[i]);
			opIdx++;
		}
		else{
			int tmplen = strlen(parts[i]);
			char* tmpLow = NULL; 
			char* tmpHi = NULL;

			rsParam->lstLow[rpIdx] = NULL; 
			rsParam->lstHi[rpIdx] = NULL; 

			//Find low, high value
			for (j = 0; j < tmplen; j++){
				if (parts[i][j] == '['){
					tmpLow = parts[i] + j+1; 
					parts[i][j] = '\0'; 
				}	
				if (parts[i][j] == ',' && tmpLow != NULL){	//End of low value
					tmpHi = parts[i] + j+1;
					parts[i][j] = '\0';
				}
				if (parts[i][j] == ']'){	//End of Hi value
					parts[i][j] = '\0';
					if (tmpHi == NULL){	//End of low value
						rsParam->lstLow[rpIdx] = GDKstrdup(tmpLow);
					}
					else{			//end of Hi value
						rsParam->lstLow[rpIdx] = GDKstrdup(tmpLow);
						rsParam->lstHi[rpIdx] = GDKstrdup(tmpHi);
					}
				}
			}
			
			rsParam->lstRPstr[rpIdx] = GDKstrdup(parts[i]);

			rpIdx++;
		}
	}


	TKNZRclose(&ret);

	return MAL_SUCCEED; 
}

static
void printParams(RdfScanParams *rsParam){
	int i; 

	for (i = 0; i < rsParam->numRP; i++){
		printf("RP[%d] = %s\n", i, rsParam->lstRPstr[i]); 
		if (rsParam->lstLow[i] != NULL) printf("   Low %s\n",rsParam->lstLow[i]);
		if (rsParam->lstHi[i] != NULL) printf("   Hi %s\n",rsParam->lstHi[i]);
	}
	for (i = 0; i < rsParam->numOP; i++){
		printf("OP[%d] = %s\n", i, rsParam->lstOPstr[i]);
	}	
	
}

static
void freeParams(RdfScanParams *rsParam){
	int i; 
	for (i = 0; i < rsParam->numRP; i++){
		GDKfree(rsParam->lstRPstr[i]);
		if (rsParam->lstLow[i] != NULL) GDKfree(rsParam->lstLow[i]);
		if (rsParam->lstHi[i] != NULL) GDKfree(rsParam->lstHi[i]);
	}
	GDKfree(rsParam->lstRPstr);
	GDKfree(rsParam->lstLow);
	GDKfree(rsParam->lstHi); 

	for (i = 0; i < rsParam->numOP; i++){
		GDKfree(rsParam->lstOPstr[i]);
	}
	GDKfree(rsParam->lstOPstr);

	GDKfree(rsParam); 

}

str RDFscan_old(str s, str schema){
	
	str query; 
	RdfScanParams *rsParam = NULL; 

	if ((query = GDKstrdup(s)) == NULL) {
		throw(MAL, "tokenizer.append", OPERATION_FAILED MAL_MALLOC_FAIL);
	}

	rsParam = (RdfScanParams*)GDKmalloc(sizeof(RdfScanParams)); 

	queryParser(rsParam, query, schema); 

	printParams(rsParam); 

	freeParams(rsParam);

	GDKfree(query); 

	return MAL_SUCCEED; 
}

str
RDFscan(oid *props, BAT **resBATs){
	(void) props; 
	(void) resBATs;
	return MAL_SUCCEED; 
}
