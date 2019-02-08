/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#include "monetdb_config.h"

#include "sql_scripts.h"
#include "gdk.h"
#include "mal_client.h"
#include "sql_catalog.h"

static struct sql_scripts {
	char *name;
	char *script;
} scripts[] = {
#include "09_like.sql.h"
#include "10_math.sql.h"
#include "11_times.sql.h"
#include "12_url.sql.h"
#include "13_date.sql.h"
#include "14_inet.sql.h"
#include "15_querylog.sql.h"
#include "16_tracelog.sql.h"
#include "17_temporal.sql.h"
#include "18_index.sql.h"
#include "20_vacuum.sql.h"
#include "21_dependency_views.sql.h"
#include "22_clients.sql.h"
#include "23_skyserver.sql.h"
#include "25_debug.sql.h"
#include "26_sysmon.sql.h"
#include "27_rejects.sql.h"
#include "39_analytics.sql.h"
#include "40_json.sql.h"
#include "41_md5sum.sql.h"
#include "45_uuid.sql.h"
#include "46_profiler.sql.h"
#include "51_sys_schema_extension.sql.h"
#include "60_wlcr.sql.h"
#include "75_storagemodel.sql.h"
#include "80_statistics.sql.h"
#include "99_system.sql.h"
#ifdef HAVE_HGE
#include "39_analytics_hge.sql.h"
#include "40_json_hge.sql.h"
#endif
	{NULL, NULL}
};

extern str SQLstatementIntern(Client c, str *expr, str nme, bit execute, bit output, res_table **result);

str
install_sql_scripts(Client c)
{
	str err;

	for (int i = 0 ; scripts[i].name ; i++) {
		fprintf(stdout, "# loading sql script: %s.sql\n", scripts[i].name);
		if ((err = SQLstatementIntern(c, &scripts[i].script, scripts[i].name, 1, 0, NULL)) != MAL_SUCCEED)
			return err;
	}
	return MAL_SUCCEED;
}
