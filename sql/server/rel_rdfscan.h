/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#ifndef _REL_RDFSCAN_H_
#define _REL_RDFSCAN_H_

#include "rel_semantic.h"
#include "sql_semantic.h"

extern sql_rel *rel_rdfscan_create(sql_allocator *sa, sql_rel *l, list *exps, rdf_rel_prop *r_r_rpop);

extern void free_rdf_rel_prop(rdf_rel_prop *r_r_prop);

extern rdf_rel_prop *init_rdf_rel_prop(int ncol, int n_ijgroup, int *nnodes_per_ijgroup); 

extern sql_rel *rel_rdfscan_func(mvc *sql, sql_table *t, int numprop, int numRP, oid *lstprops, oid *los, oid *his, list *exps, char **lstAlias); 


#endif /*_REL_RDFSCAN_H_*/
