/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#include "monetdb_config.h"

#include "sql_embedded.h"
#include "mal_embedded.h"
#include "gdk.h"
#include "mal_client.h"
#include "mal_import.h"

static volatile bool SQLembeddedinitialized = false;

malSignatures sqlMalModules[] =
{
#include "sql_decimal.mal.h"
#include "sql_rank.mal.h"
#include "sql_aggr_bte.mal.h"
#include "sql_aggr_sht.mal.h"
#include "sql_aggr_int.mal.h"
#include "sql_aggr_lng.mal.h"
#include "sql_aggr_flt.mal.h"
#include "sql_aggr_dbl.mal.h"
#include "sql_inspect.mal.h"
#include "sql_generator.mal.h"

#ifdef HAVE_HGE
#include "sql_decimal_hge.mal.h"
#include "sql_rank_hge.mal.h"
#include "sql_aggr_hge.mal.h"
#include "sql_hge.mal.h"
#include "sql_generator_hge.mal.h"
#endif

#include "sqlcatalog.mal.h"
#include "sql_transaction.mal.h"
#include "wlr.mal.h"
#include "sql.mal.h"
	{NULL, NULL}
};

int
sqlEmbeddedBoot(void)
{
	str msg;
	Client c;

	if( SQLembeddedinitialized )
		return 0;

	if (mal_init("libsql")) {
		MT_fprintf(stderr, "MAL init failed\n");
		return -1;
	}

	c = MCinitClient((oid) 0, bstream_create(GDKstdin, 0), GDKstdout);
	if (!MCvalid(c)) {
		MT_fprintf(stderr, "MAL init failed\n");
		mal_exit(1);
		return -1;
	}
	c->curmodule = c->usermodule = userModule();
	if ((msg = MSinitClientPrg(c, "user", "main")) != MAL_SUCCEED) {
		MT_fprintf(stderr, "%s\n", msg);
		freeException(msg);
		mal_exit(1);
		return -1;
	}
	if ((msg = malEmbeddedBoot(c)) != MAL_SUCCEED) {
		MT_fprintf(stderr, "%s\n", msg);
		freeException(msg);
		mal_exit(1);
		return -1;
	}
	if ((msg = malExtraModulesBoot(c, sqlMalModules)) != MAL_SUCCEED) {
		MT_fprintf(stderr, "%s\n", msg);
		freeException(msg);
		mal_exit(1);
		return -1;
	}

	MCcloseClient(c);
	SQLembeddedinitialized = true;
	return 0;
}

int
sqlEmbeddedShutdown(void)
{
	if( !SQLembeddedinitialized )
		return 0;
	(void) malEmbeddedStop();
	mserver_reset();
	SQLembeddedinitialized = false;
	return 0;
}
