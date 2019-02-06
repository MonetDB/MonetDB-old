/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

/*
 * (author) M.L. Kersten
 * These routines assume that the signatures for all MAL files are defined as text in mal_embdded.h
 * They are parsed upon system restart without access to their source files.
 * This way the definitions are part of the library upon compilation.
 */
#include "monetdb_config.h"
#include "mal_type.h"
#include "mal_namespace.h"
#include "mal_exception.h"
#include "mal_private.h"
#include "mal_embedded.h"

static int embeddedinitialized = 0;

/* The source for the MAL signatures */
static struct{
	str filename, source;
} malSignatures[] = 
{
{ "welcome", "io.print(\"Load MAL signatures\");"},
{ 0, 0}
}
;
str
malEmbeddedBoot(Client c)
{
	int i;

	(void) c;
	if( embeddedinitialized )
		return MAL_SUCCEED;
	for(i = 0; malSignatures[i].filename; i++){
		fprintf(stderr, "Load the file %s\n", malSignatures[i].source);
	}
	embeddedinitialized = 1;
	return MAL_SUCCEED;
}

str
malEmbeddedStop(Client c)
{
	(void) c;
	return MAL_SUCCEED;
}

str
malEmbeddedRestart(Client c)
{
	(void) c;
	return MAL_SUCCEED;
}
