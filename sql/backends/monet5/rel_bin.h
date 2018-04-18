/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#ifndef _REL_BIN_H_
#define _REL_BIN_H_

#include "rel_semantic.h"
#include "sql_statement.h"
#include "mal_backend.h"

extern stmt * output_rel_bin(backend *be, sql_rel *rel);
extern stmt * sql_parse(backend *be, sql_allocator *sa, char *query, char mode);
extern stmt * exp_bin(backend *be, sql_exp *e, stmt *left, stmt *right, stmt *grp, stmt *ext, stmt *cnt, stmt *sel);
extern stmt * subrel_bin(backend *be, sql_rel *rel, list *refs);
extern stmt * bin_find_column(backend *be, stmt *sub, const char *rname, const char *name);
extern list *rel_dependencies(sql_allocator *sa, sql_rel *r);

#endif /*_REL_BIN_H_*/
