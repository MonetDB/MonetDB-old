/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#include "monetdb_config.h"

#include "sql_embedded.h"
#include "gdk.h"
#include "mal.h"
#include "mal_client.h"
#include "language.h"

static struct mal_sql_scripts {
	char *modnme;
	char *script;
} mal_sql_scripts[] = {
#include "sql_decimal.mal.h"
#include "sql_rank.mal.h"
#include "sql_aggr_bte.mal.h"
#include "sql_aggr_sht.mal.h"
#include "sql_aggr_int.mal.h"
#include "sql_aggr_lng.mal.h"
#include "sql_aggr_flt.mal.h"
#include "sql_aggr_dbl.mal.h"
#include "sql_inspect.mal.h"
#include "sqlcatalog.mal.h"
#include "sql_transaction.mal.h"
#include "wlr.mal.h"
#include "40_sql.mal.h"
#include "sql.mal.h"

#ifdef HAVE_HGE
#include "sql_decimal_hge.mal.h"
#include "sql_rank_hge.mal.h"
#include "sql_aggr_hge.mal.h"
#include "41_sql_hge.mal.h"
#include "sql_hge.mal.h"
#endif
	{NULL, NULL}
};

int
sqlEmbeddedBoot(void)
{
	str msg;
	Client c;

	if (mal_init())
		return -1;

	if ((msg = MSinitClientPrg(mal_clients, "user", "main")) != MAL_SUCCEED) {
		fprintf(stderr, "%s\n", msg);
		freeException(msg);
		mal_exit(1);
		return -1;
	}

	c = &mal_clients[0];
	for (int i = 0; mal_sql_scripts[i].modnme; i++) {
		if ((msg = callString(c, mal_sql_scripts[i].script, FALSE)) != MAL_SUCCEED) {
			fprintf(stderr,"#sqlEmbeddedBoot: Failed to start SQL MAL scripts: %s", msg);
			freeException(msg);
			mal_exit(1);
			return -1;
		}
	}

	return 0;
}
