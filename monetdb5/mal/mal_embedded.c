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
#include "mal_embedded.h"

static bool embeddedinitialized = false;
static int nDefaultModules = 0;

/* The source for the MAL signatures*/
malSignatures malModules[MAXMODULES] =
{
// Include the MAL definitions files in the proper order.

#ifndef NDEBUG
#include "mdb.mal.h"
#endif
#include "alarm.mal.h"
#include "mmath.mal.h"
#include "streams.mal.h"

#include "bat5.mal.h"
#include "batExtensions.mal.h"
#include "algebra.mal.h"
#include "orderidx.mal.h"
#include "status.mal.h"
#include "groupby.mal.h"
#include "group.mal.h"
#include "aggr.mal.h"
#include "mkey.mal.h"

#include "blob.mal.h"
#include "str.mal.h"
#include "mtime.mal.h"
#ifndef HAVE_EMBEDDED
#include "color.mal.h"
#include "url.mal.h"
#include "uuid.mal.h"
#include "json.mal.h"
#include "json_util.mal.h"
#include "inet.mal.h"
#include "identifier.mal.h"
#include "xml.mal.h"
#endif

#include "batmmath.mal.h"
#include "batmtime.mal.h"
#include "batstr.mal.h"
#ifndef HAVE_EMBEDDED
#include "batcolor.mal.h"
#include "batxml.mal.h"
#endif

#include "pcre.mal.h"
#ifndef HAVE_EMBEDDED
#include "clients.mal.h"
#endif
#include "bbp.mal.h"
#include "mal_io.mal.h"
#include "manifold.mal.h"
#ifndef HAVE_EMBEDDED
#include "factories.mal.h"
#include "remote.mal.h"
#endif

#include "mat.mal.h"
#include "inspect.mal.h"
#include "manual.mal.h"
#include "language.mal.h"

#ifndef HAVE_EMBEDDED
#include "profiler.mal.h"
#include "querylog.mal.h"
#include "sysmon.mal.h"
#endif
#include "sample.mal.h"

#include "optimizer.mal.h"

#include "iterator.mal.h"
#ifndef HAVE_EMBEDDED
#include "txtsim.mal.h"
#include "tokenizer.mal.h"
#include "mal_mapi.mal.h"
#endif
#include "oltp.mal.h"
#include "microbenchmark.mal.h"
#ifndef HAVE_EMBEDDED
#include "wlc.mal.h"
#endif

#ifdef HAVE_HGE
#include "00_aggr_hge.mal.h"
#include "00_batcalc_hge.mal.h"
#include "00_calc_hge.mal.h"
#include "00_batExtensions_hge.mal.h"
#include "00_iterator_hge.mal.h"
#include "00_language_hge.mal.h"
#include "00_mkey_hge.mal.h"
#include "00_mal_mapi_hge.mal.h"
#include "00_json_hge.mal.h"
#endif

#include "language.mal.h"
#include "01_batcalc.mal.h"
#include "01_calc.mal.h"

#ifndef HAVE_EMBEDDED
#include "run_adder.mal.h"
#include "run_isolate.mal.h"
#include "run_memo.mal.h"
#endif
{ 0, 0}
};

str
malEmbeddedBoot(Client c)
{
	int i = 0;
	str msg = MAL_SUCCEED;

	if( embeddedinitialized )
		return MAL_SUCCEED;
	for(; malModules[i].modnme; i++) {
		if ((msg = callString(c, malModules[i].source, FALSE)) != MAL_SUCCEED)
			return msg;
	}
	nDefaultModules = i;
	embeddedinitialized = true;
	return msg;
}

str
malExtraModulesBoot(Client c, malSignatures extraMalModules[])
{
	int i, j, k;
	str msg = MAL_SUCCEED;

	for (i = 0; malModules[i].modnme; i++);
	if (i == MAXMODULES-1) //the last entry must be set to NULL
		throw(MAL, "malInclude", "too many MAL modules loaded");

	for (j = 0, k = i; k < MAXMODULES-1 && extraMalModules[j].modnme && extraMalModules[j].source; k++, j++);
	if (k == MAXMODULES-1)
		throw(MAL, "malInclude", "the number of MAL modules to load, exceed the available MAL modules slots");

	memcpy(&malModules[i], &extraMalModules[0], j * sizeof(malSignatures));
	memset(&malModules[k], 0, sizeof(malSignatures));

	for(i = 0; extraMalModules[i].modnme && extraMalModules[i].source; i++) {
		if ((msg = callString(c, extraMalModules[i].source, FALSE)) != MAL_SUCCEED)
			return msg;
	}
	return msg;
}

str
malEmbeddedStop(void) //remove extra modules and set to non-initialized again
{
	memset(&malModules[nDefaultModules], 0, (MAXMODULES-1 - nDefaultModules) * sizeof(malSignatures));
	embeddedinitialized = false;
	return MAL_SUCCEED;
}

str
malEmbeddedRestart(Client c)
{
	(void) c;
	return MAL_SUCCEED;
}
