/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _REL_OPTIMIZER_H_
#define _REL_OPTIMIZER_H_

#include "sql_relation.h"
#include "sql_mvc.h"

sql_extern sql_rel * rel_optimizer(mvc *sql, sql_rel *rel, int value_based_opt);

sql_extern int exp_joins_rels(sql_exp *e, list *rels);

sql_extern void *name_find_column( sql_rel *rel, const char *rname, const char *name, int pnr, sql_rel **bt );
sql_extern int exps_unique(mvc *sql, sql_rel *rel, list *exps);

sql_extern sql_rel * rel_dce(mvc *sql, sql_rel *rel);

#endif /*_REL_OPTIMIZER_H_*/
