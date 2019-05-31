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

#include "mal_inline.h"
#include "mal_inline_names.h"

extern void WLCreset(void); // Don't include wlc.h or opt_support.h, it creates a circular dependency
extern void opt_pipes_reset(void);

str
malEmbeddedBoot(void)
{
	Client c;
	str msg = MAL_SUCCEED;

	if( embeddedinitialized )
		return MAL_SUCCEED;

	if (!MCinit())
		throw(MAL, "malEmbeddedBoot", "MAL debugger failed to start");
#if !defined(NDEBUG) && !defined(HAVE_EMBEDDED)
	if (!mdbInit()) {
		mal_client_reset();
		throw(MAL, "malEmbeddedBoot", "MAL debugger failed to start");
	}
#endif
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
	if ((msg = MSinitClientPrg(c, "user", "main")) != MAL_SUCCEED) {
		MCcloseClient(c);
		return msg;
	}
	if ((msg = malInlineBoot(c, "mal_boot_scripts", mal_inline, 0)) != MAL_SUCCEED) {
		MCcloseClient(c);
		return msg;
	}
	for(nDefaultModules = 0; malModules[nDefaultModules]; nDefaultModules++);
#ifndef HAVE_EMBEDDED
	if ((msg = malInclude(c, "mal_init", 0)) != MAL_SUCCEED) {
		MCcloseClient(c);
		return msg;
	}
#endif
	pushEndInstruction(c->curprg->def);
	chkProgram(c->usermodule, c->curprg->def);
	if ( (msg= c->curprg->def->errors) != MAL_SUCCEED ) {
		MCcloseClient(c);
		return msg;
	}
	msg = MALengine(c);
	embeddedinitialized = true;
	MCcloseClient(c);
	return msg;
}

str
malExtraModulesBoot(Client c, str extraMalModules[], char* mal_scripts)
{
	int i, j, k;

	for (i = 0; malModules[i]; i++);
	if (i == MAXMODULES-1) //the last entry must be set to NULL
		throw(MAL, "malInclude", "too many MAL modules loaded");

	for (j = 0, k = i; k < MAXMODULES-1 && extraMalModules[j]; k++, j++);
	if (k == MAXMODULES-1)
		throw(MAL, "malInclude", "the number of MAL modules to load, exceed the available MAL modules slots");

	memcpy(&malModules[i], &extraMalModules[0], j * sizeof(str));
	memset(&malModules[k], 0, sizeof(str));

	return malInlineBoot(c, "mal_extra_scripts", mal_scripts, 0);
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
	if (!embeddedinitialized)
		return;

	memset(&malModules[nDefaultModules], 0, (MAXMODULES-1 - nDefaultModules) * sizeof(str));
	GDKprepareExit();
	MCstopClients(0);
#ifndef HAVE_EMBEDDED
	WLCreset();
	setHeartbeat(-1);
	stopProfiler();
	AUTHreset();
	if (!GDKinmemory()) {
		str msg = MAL_SUCCEED;
		if ((msg = msab_wildRetreat()) != NULL) {
			MT_fprintf(stderr, "!%s", msg);
			free(msg);
		}
		if ((msg = msab_registerStop()) != NULL) {
			MT_fprintf(stderr, "!%s", msg);
			free(msg);
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
#if !defined(NDEBUG) && !defined(HAVE_EMBEDDED)
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
	embeddedinitialized = false;
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
