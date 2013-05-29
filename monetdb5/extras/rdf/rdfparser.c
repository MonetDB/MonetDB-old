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

/* Store the common functions for rdf parser */

/*
 * @-
 * The (fatal) errors and warnings produced by the raptor parser are handled
 * by the next three message handler functions.
 */

#include "monetdb_config.h"
#include "rdf.h"
#include "algebra.h"
#include <gdk.h>
#include <rdfparser.h>

void
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


void
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


void
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

void
raptor_exception(parserData *pdata, const char* msg){
	pdata->exception++;
	pdata->exceptionMsg =  GDKstrdup(msg);
	raptor_parser_parse_abort (pdata->rparser);
}

