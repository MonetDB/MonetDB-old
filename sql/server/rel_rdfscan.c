/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "rel_rdfscan.h"
#include "sql_semantic.h"	/* TODO this dependency should be removed, move
				   the dependent code into sql_mvc */
#include "sql_privileges.h"
#include "sql_env.h"
#include "rel_exp.h"
#include "rel_xml.h"
#include "rel_dump.h"
#include "rel_prop.h"
#include "rel_psm.h"
#include "rel_schema.h"
#include "rel_sequence.h"
#include "rel_select.h"
#ifdef HAVE_HGE
#include "mal.h"		/* for have_hge */
#endif


rdf_rel_prop *init_rdf_rel_prop(int ncol, int n_ijgroup, int *nnodes_per_ijgroup){
	int i; 
	rdf_rel_prop *r_r_prop = (rdf_rel_prop *) malloc(sizeof(rdf_rel_prop));
	r_r_prop->ncol = ncol; 
	r_r_prop->nopt = (int *) malloc(sizeof(int) * n_ijgroup); 
	memcpy(r_r_prop->nopt, nnodes_per_ijgroup, sizeof(int) * n_ijgroup);
	r_r_prop->lstcol = (char **)malloc(sizeof(char *) * ncol); 
	for (i = 0; i < ncol; i++){
		r_r_prop->lstcol[i] = NULL; 
	}
	r_r_prop->mv_prop = (char *)malloc(sizeof(char) * ncol); 
	r_r_prop->containMV = 0; 
	return r_r_prop; 
}

void free_rdf_rel_prop(rdf_rel_prop *r_r_prop){
	int i; 
	for (i = 0; i < r_r_prop->ncol; i++){
		if (r_r_prop->lstcol[i]) GDKfree(r_r_prop->lstcol[i]); 
	}
	if (r_r_prop->lstcol) free(r_r_prop->lstcol);
	if (r_r_prop->nopt) free(r_r_prop->nopt);
	if (r_r_prop->mv_prop) free(r_r_prop->mv_prop); 

	free(r_r_prop);
}

/*
 * Create rdfscan sql_rel.
 * This is similar to rel_select_copy
 * */

sql_rel *
rel_rdfscan_create(sql_allocator *sa, sql_rel *l, list *exps, rdf_rel_prop *r_r_prop)
{
	sql_rel *rel = rel_create(sa);
	
	rel->l = l;
	rel->r = NULL;
	rel->rrp = r_r_prop; 
	rel->op = op_rdfscan;
	rel->exps = exps?list_dup(exps, (fdup)NULL):NULL;
	rel->card = CARD_ATOM; /* no relation */
	if (l) {
		rel->card = l->card;
		rel->nrcols = l->nrcols;
	}
	return rel;
}


static list *
table_column_types(sql_allocator *sa, sql_table *t)
{
	node *n;
	list *types = sa_list(sa);

	if (t->columns.set) for (n = t->columns.set->h; n; n = n->next) {
		sql_column *c = n->data;
		if (c->base.name[0] != '%')
			append(types, &c->type);
	}
	return types;
}

sql_rel *
rel_rdfscan_func(mvc *sql, sql_table *t, int numprop, int nRP, oid *lstprop, oid *los, oid *his, list *sel_exps)
{
	sql_rel *resbase;
	sql_rel *res;
	list *exps, *args;
	node *n;
	sql_exp *import;
	int i; 
	sql_schema *sys = mvc_bind_schema(sql, "sys");
	sql_subfunc *f = sql_find_func(sql->sa, sys, "rdfscan", -1, F_UNION, NULL); 
	
	
	if (!f) /* we do expect copyfrom to be there */
		return NULL;

	f->res = table_column_types(sql->sa, t);
	args = new_exp_list(sql->sa); 
	append(args, exp_atom_int(sql->sa, numprop));
	append(args, exp_atom_int(sql->sa, nRP)); 
	
	for (i = 0; i < numprop; i++){
		append(args, exp_atom_oid(sql->sa, lstprop[i]));  	
	}
	for (i = 0; i < numprop; i++){
		append(args, exp_atom_oid(sql->sa, los[i]));
	}
	
	for (i = 0; i < numprop; i++){
		append(args, exp_atom_oid(sql->sa, his[i]));
	}

	import = exp_op(sql->sa, args, f); 
	
	exps = new_exp_list(sql->sa);
	i = 0; 
	for (n = t->columns.set->h; n; n = n->next) {
		sql_column *c = n->data;
		if (c->base.name[0] != '%'){
			append(exps, exp_column(sql->sa, t->base.name, c->base.name, &c->type, CARD_MULTI, c->null, 0));
			//str s_or_o = NULL; 
			//int idx = i % 2; 
			//s_or_o = strlen((str)objStr);
			//append(exps, exp_alias(sql->sa, lstAlias[idx], s_or_o, t->base.name, c->base.name, &c->type, CARD_MULTI, c->null, 0));
		}
		//i++; 
	}
	resbase = rel_table_func(sql->sa, NULL, import, exps, 1);

	if (0) res = rel_select_copy(sql->sa, resbase, sel_exps);
	(void) res; 
	//return res;
	return resbase; 
}
