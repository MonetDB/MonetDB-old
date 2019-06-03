/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _SQL2MAL_H
#define _SQL2MAL_H

#include "sql.h"
#include "mal_backend.h"
#include "sql_atom.h"
#include "sql_statement.h"
#include "sql_env.h"
#include "sql_mvc.h"
#include "mal_function.h"

sql_extern Symbol backend_dumpproc(backend *be, Client c, cq *q, sql_rel *r);
sql_extern int backend_callinline(backend *be, Client c);
sql_extern int backend_dumpstmt(backend *be, MalBlkPtr mb, sql_rel *r, int top, int addend, const char *query);
sql_extern void backend_call(backend *be, Client c, cq *q);
sql_extern int monet5_resolve_function(ptr M, sql_func *f);
sql_extern int backend_create_func(backend *be, sql_func *f, list *restypes, list *ops);
sql_extern int backend_create_subfunc(backend *be, sql_subfunc *f, list *ops);
sql_extern int backend_create_subaggr(backend *be, sql_subaggr *f);

sql_extern int monet5_create_relational_function(mvc *m, const char *mod, const char *name, sql_rel *rel, stmt *call, list *rel_ops, int inline_func);

sql_extern void rel_print(mvc *sql, sql_rel *rel, int depth);
sql_extern void _rel_print(mvc *sql, sql_rel *rel);

sql_extern int constantAtom(backend *be, MalBlkPtr mb, atom *a);
sql_extern InstrPtr table_func_create_result(MalBlkPtr mb, InstrPtr q, sql_func *f, list *restypes);
sql_extern InstrPtr relational_func_create_result(mvc *sql, MalBlkPtr mb, InstrPtr q, sql_rel *f);

#endif /* _SQL2MAL_H */
