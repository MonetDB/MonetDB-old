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

/*
 * @f rdf
 * @a Minh-Duc Pham
 *
 * @* The RDF module For MonetDB5 (aka. MonetDB/RDF)
 *
 */
#ifndef _RDF_PARSER_H_
#define _RDF_PARSER_H_

#include <raptor2.h>

#define CHECK_NUM_DBPONTOLOGY   1       /* Check how many rdf triples use dbpontology */

typedef struct parserData {
	                              /**PROPERTIES             */
	str location;                 /* rdf data file location */
	oid tcount;                   /* triple count           */
	raptor_parser *rparser;       /* the parser object      */
	                              /**ERROR HANDLING         */
	int exception;                /* raise an exception     */
	int warning;                  /* number of warning msgs */
	int error;                    /* number of error   msgs */
	int lasterror;		      /* # errors before next triple */
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
#if CHECK_NUM_DBPONTOLOGY
	int numOntologyTriples; 
#endif
	
} parserData;

void fatalHandler (void *user_data, raptor_log_message* message);
void errorHandler (void *user_data, raptor_log_message* message);
void warningHandler (void *user_data, raptor_log_message* message);
void raptor_exception(parserData *pdata, const char* msg);

#endif /* _RDF_PARSER_H_ */
