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
 * It assumes that all necessary libraries are already loaded.
 * A failure to bind the address in the context of an embedded version is not considered an error.
 */
#include "monetdb_config.h"
#include "mal_type.h"
#include "mal_namespace.h"
#include "mal_exception.h"
#include "mal_private.h"
#include "mal_builder.h"
#include "mal_embedded.h"

static int embeddedinitialized = 0;

/* The source for the MAL signatures*/
static struct{
	str modnme, source;
} malSignatures[] = 
{
// Include the MAL definitions files in the proper order.

#include "../modules/mal/mdb.include"
#include "../modules/kernel/alarm.include"
#include "../modules/kernel/mmath.include"
#include "../modules/atoms/streams.include"

#include "../modules/kernel/bat5.include"
#include "../modules/mal/batExtensions.include"
#include "../modules/kernel/algebra.include"
#include "../modules/mal/orderidx.include"
#include "../modules/kernel/status.include"
#include "../modules/mal/groupby.include"
#include "../modules/kernel/group.include"
#include "../modules/kernel/aggr.include"
#include "../modules/mal/mkey.include"

#include "../modules/atoms/blob.include"
#include "../modules/atoms/color.include"
#include "../modules/atoms/str.include"
#include "../modules/atoms/url.include"
#include "../modules/atoms/uuid.include"
#include "../modules/atoms/json.include"
#include "../modules/mal/json_util.include"
#include "../modules/atoms/mtime.include"
#include "../modules/atoms/inet.include"
#include "../modules/atoms/identifier.include"
#include "../modules/atoms/xml.include"
#include "../modules/atoms/batxml.include"

#include "../modules/kernel/batmmath.include"
#include "../modules/mal/batmtime.include"
#include "../modules/kernel/batstr.include"
#include "../modules/kernel/batcolor.include"

#include "../modules/mal/sabaoth.include"
#include "../modules/mal/pcre.include"
#include "../modules/mal/clients.include"
#include "../modules/mal/bbp.include"
#include "../modules/mal/mal_io.include"
#include "../modules/mal/manifold.include"
#include "../modules/mal/factories.include"
#include "../modules/mal/remote.include"

#include "../modules/mal/mat.include"
#include "../modules/mal/inspect.include"
#include "../modules/mal/manual.include"
#include "../modules/mal/language.include"

#include "../modules/mal/profiler.include"
#include "../modules/mal/querylog.include"
#include "../modules/mal/sysmon.include"
#include "../modules/mal/sample.include"

#include "../optimizer/optimizer.include"

#include "../modules/mal/iterator.include"
#include "../modules/mal/txtsim.include"
#include "../modules/mal/tokenizer.include"

#include "../modules/mal/mal_mapi.include"
#include "../modules/mal/oltp.include"
#include "../modules/mal/wlc.include"

// Any extensions (MAL scripts) that should be automatically loaded upon
// startup can be placed in the autoload directory.  One typically finds
// the SQL module in here, but it can also be used to load custom scripts.

//include calc; --- moved to autoload/01_calc
//include batcalc; -- moved to autoload/01_batcalc
// include autoload;
// test case
//{ "mdbextension", "module dummy; pattern dummy.embed(mod:str, fcn:str): void address MDBstartFactory; "
//"pattern dummy.genius():void address MDBnotknown;"},
{ 0, 0}
}
;


str
malEmbeddedBoot(Client c)
{
	int i;
	str msg = MAL_SUCCEED;

	
	if( embeddedinitialized )
		return MAL_SUCCEED;
	for(i = 0; malSignatures[i].modnme; i++){
		msg = callString(c, malSignatures[i].source, FALSE);
		if (msg) {
			fprintf(stderr,"!ERROR: malEmbeddedBoot: %s\n", msg);
			GDKfree(msg);
		}
	}
	embeddedinitialized = 1;
	return msg;
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
