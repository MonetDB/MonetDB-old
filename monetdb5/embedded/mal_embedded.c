/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

/*
 * (author) M.L. Kersten
 * These routines assume that the signatures for all MAL files are defined as text in mal_embedded.h
 * They are parsed upon system restart without access to their source files.
 * This way the definitions are part of the library upon compilation.
 * It assumes that all necessary libraries are already loaded.
 * A failure to bind the address in the context of an embedded version is not considered an error.
 */

#include "monetdb_config.h"

#include "mal_embedded.h"
#include "mal_builder.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "mal_session.h"
#include "mal_scenario.h"
#include "mal_parser.h"
#include "mal_interpreter.h"
#include "mal_namespace.h"  /* for initNamespace() */
#include "mal_client.h"
#include "mal_dataflow.h"
#include "mal_private.h"
#include "mal_runtime.h"
#include "mal_atom.h"
#include "mal_resource.h"
#include "mal_atom.h"
#ifndef HAVE_EMBEDDED
#include "msabaoth.h"
#include "mal_authorize.h"
#include "mal_profiler.h"
#endif

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
#include "tablet.mal.h"
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

extern void WLCreset(void); // Don't include wlc.h or opt_support.h, it creates a circular dependency
extern void opt_pipes_reset(void);

str
malEmbeddedBoot(const char* library_path)
{
	Client c;
	int i = 0;
	str msg = MAL_SUCCEED;

	if( embeddedinitialized )
		return MAL_SUCCEED;

	if (!MCinit())
		throw(MAL, "malEmbeddedBoot", "MAL debugger failed to start");
#ifndef NDEBUG
	if (!mdbInit()) {
		mal_client_reset();
		throw(MAL, "malEmbeddedBoot", "MAL debugger failed to start");
	}
#endif
	if (!initLinker(library_path)) {
		mal_client_reset();
		throw(MAL, "malEmbeddedBoot", "MAL linker failed to start");
	}
	monet_memory = MT_npages() * MT_pagesize();
	initNamespace();
	initParser();
#ifndef HAVE_EMBEDDED
	initHeartbeat();
	initResource();
	initProfiler();
#endif

	c = MCinitClient((oid) 0, bstream_create(GDKstdin, 0), GDKstdout);
	if(c == NULL)
		throw(MAL, "malEmbeddedBoot", "Failed to initialize client");
	c->curmodule = c->usermodule = userModule();
	if(c->usermodule == NULL) {
		MCcloseClient(c);
		throw(MAL, "malEmbeddedBoot", "Failed to initialize client MAL module");
	}
	if ( (msg = defaultScenario(c)) ) {
		MCcloseClient(c);
		return msg;
	}
	if((msg = MSinitClientPrg(c, "user", "main")) != MAL_SUCCEED) {
		MCcloseClient(c);
		return msg;
	}
	for(; malModules[i].modnme; i++) {
		if ((msg = callString(c, malModules[i].source, FALSE)) != MAL_SUCCEED) {
			MCcloseClient(c);
			return msg;
		}
	}
	nDefaultModules = i;
#ifndef HAVE_EMBEDDED
	if ((msg = malInclude(c, "mal_init", 0)) != MAL_SUCCEED) {
		MCcloseClient(c);
		return msg;
	}
	pushEndInstruction(c->curprg->def);
	chkProgram(c->usermodule, c->curprg->def);
	if ( (msg= c->curprg->def->errors) != MAL_SUCCEED ) {
		MCcloseClient(c);
		return msg;
	}
	msg = MALengine(c);
#endif
	embeddedinitialized = true;
	MCcloseClient(c);
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

/*
 * Upon exit we should attempt to remove all allocated memory explicitly.
 * This seemingly superflous action is necessary to simplify analyis of
 * memory leakage problems later ons and to allow an embedded server to
 * restart the server properly.
 *
 * It is the responsibility of the enclosing application to finish/cease all
 * activity first.
 * This function should be called after you have issued sql_reset();
 */

void
malEmbeddedReset(void) //remove extra modules and set to non-initialized again
{
	memset(&malModules[nDefaultModules], 0, (MAXMODULES-1 - nDefaultModules) * sizeof(malSignatures));
	embeddedinitialized = false;
	str err = 0;

	GDKprepareExit();
	MCstopClients(0);
#ifndef HAVE_EMBEDDED
	WLCreset();
	setHeartbeat(-1);
	stopProfiler();
	AUTHreset();
	if (!GDKinmemory()) {
		if ((err = msab_wildRetreat()) != NULL) {
			MT_fprintf(stderr, "!%s", err);
			free(err);
		}
		if ((err = msab_registerStop()) != NULL) {
			MT_fprintf(stderr, "!%s", err);
			free(err);
		}
	}
	mal_factory_reset();
	mal_runtime_reset();
#endif
	mal_dataflow_reset();
	mal_client_reset();
	mal_linker_reset();
	mal_resource_reset();
	mal_module_reset();
	mal_atom_reset();
	opt_pipes_reset();
#ifndef NDEBUG
	mdbExit();
#endif

	memset((char*)monet_cwd, 0, sizeof(monet_cwd));
	monet_memory = 0;
	memset((char*)monet_characteristics,0, sizeof(monet_characteristics));
	mal_namespace_reset();
	/* No need to clean up the namespace, it will simply be extended
	 * upon restart mal_namespace_reset(); */
#ifndef HAVE_EMBEDDED
	GDKreset(0);	// terminate all other threads
#endif
}

/* stopping clients should be done with care, as they may be in the mids of
 * transactions. One safe place is between MAL instructions, which would
 * abort the transaction by raising an exception. All sessions are
 * terminate this way.
 * We should also ensure that no new client enters the scene while shutting down.
 * For this we mark the client records as BLOCKCLIENT.
 *
 * Beware, mal_exit is also called during a SIGTERM from the monetdb tool
 */

void
malEmbeddedStop(int status)
{
	malEmbeddedReset();
	exit(status); /* properly end GDK */
}
