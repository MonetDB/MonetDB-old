/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _REL_UPDATES_H_
#define _REL_UPDATES_H_

#include "sql_list.h"
#include "sql_symbol.h"
#include "sql_mvc.h"
#include "sql_relation.h"
#include "sql_query.h"

sql_extern sql_exp **table_update_array(mvc *sql, sql_table *t);
sql_extern sql_rel *rel_update(mvc *sql, sql_rel *t, sql_rel *uprel, sql_exp **updates, list *exps);

sql_extern sql_rel *rel_insert(sql_query *query, sql_rel *t, sql_rel *inserts);
sql_extern sql_rel *rel_delete(sql_allocator *sa, sql_rel *t, sql_rel *deletes);
sql_extern sql_rel *rel_truncate(sql_allocator *sa, sql_rel *t, int drop_action, int check_identity);

sql_extern sql_exp * rel_parse_val(mvc *m, char *query, char emode, sql_rel *from);

sql_extern sql_rel *rel_updates(sql_query *query, symbol *sym);

sql_extern sql_table *insert_allowed(mvc *sql, sql_table *t, char *tname, char *op, char *opname);
sql_extern sql_table *update_allowed(mvc *sql, sql_table *t, char *tname, char *op, char *opname, int is_delete);

#endif /*_REL_UPDATES_H_*/
