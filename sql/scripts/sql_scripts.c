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

struct sql_scripts {
	char *name;
	char *script;
};

static struct sql_scripts scripts1[] = {
#include "09_like.sql.h"
#include "10_math.sql.h"
#include "11_times.sql.h"
#ifndef HAVE_EMBEDDED
#include "12_url.sql.h"
#endif
#include "13_date.sql.h"
#ifndef HAVE_EMBEDDED
#include "14_inet.sql.h"
#include "15_querylog.sql.h"
#include "16_tracelog.sql.h"
#endif
#include "17_temporal.sql.h"
#include "18_index.sql.h"
#include "20_vacuum.sql.h"
#include "21_dependency_views.sql.h"
#ifndef HAVE_EMBEDDED
#include "22_clients.sql.h"
#include "23_skyserver.sql.h"
#include "25_debug.sql.h"
#include "26_sysmon.sql.h"
#include "27_rejects.sql.h"
#endif
#include "39_analytics.sql.h"
#ifdef HAVE_HGE
#include "39_analytics_hge.sql.h"
#endif
#ifndef HAVE_EMBEDDED
#include "40_json.sql.h"
#ifdef HAVE_HGE
#include "40_json_hge.sql.h"
#endif
#endif
#ifndef HAVE_EMBEDDED
#include "41_md5sum.sql.h"
#include "45_uuid.sql.h"
#include "46_profiler.sql.h"
#endif
#include "51_sys_schema_extension.sql.h"
#ifndef HAVE_EMBEDDED
#include "60_wlcr.sql.h"
#endif
#include "70_storagemodel.sql.h"
#include "71_statistics.sql.h"
#include "90_generator.sql.h"
#ifdef HAVE_HGE
#include "90_generator_hge.sql.h"
#endif
	{NULL, NULL}
};

static struct sql_scripts scripts2[] = {
#include "99_system.sql.h"
	{NULL, NULL}
};

extern str SQLstatementIntern(Client c, str *expr, str nme, bit execute, bit output, res_table **result);

static str
install_sql_scripts_array(Client c, struct sql_scripts scripts[])
{
	str err;
	for (int i = 0 ; scripts[i].name ; i++) {
		MT_fprintf(stdout, "# loading sql script: %s.sql\n", scripts[i].name);
		if ((err = SQLstatementIntern(c, &scripts[i].script, scripts[i].name, 1, 0, NULL)) != MAL_SUCCEED)
			return err;
	}
	return MAL_SUCCEED;
}

str
install_sql_scripts1(Client c)
{
	str err;
	if ((err = install_sql_scripts_array(c, scripts1)) != MAL_SUCCEED)
		return err;
	return MAL_SUCCEED;
}

str
install_sql_scripts2(Client c)
{
	str err;
	if ((err = install_sql_scripts_array(c, scripts2)) != MAL_SUCCEED)
		return err;
	return MAL_SUCCEED;
}
