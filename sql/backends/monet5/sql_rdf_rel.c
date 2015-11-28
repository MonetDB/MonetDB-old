/*
 * The contents of this file are subject to the MonetDB Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.monetdb.org/Legal/MonetDBLicense
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is the MonetDB Database System.
 *
 * The Initial Developer of the Original Code is CWI.
 * Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
 * Copyright August 2008-2014 MonetDB B.V.
 * All Rights Reserved.
 */

#include "monetdb_config.h"
#include <sql_rdf_rel.h>
#include <rdf.h>
#include <rel_dump.h>
#include <rel_select.h>
#include <rdflabels.h>
#include <rel_exp.h>
#include <sql_rdf.h>
#include <rdfschema.h>


sql_rel *
rdf_rel_join(sql_allocator *sa, sql_rel *l, sql_rel *r, list *exps, operator_type join)
{
	sql_rel *rel = rel_create(sa);

	rel->l = l;
	rel->r = r;
	rel->op = join;
	rel->exps = exps;
	rel->card = CARD_MULTI;
	rel->nrcols = l->nrcols + r->nrcols;
	return rel;
}

/*
* Simply combine the set of columns and the exps from
* two rels. This is applied when we combine required and optional columnns
* and put a conditional project on top of the combination for 
* handling OPTIONAL
* Note that: rel1_orig may be op_join if the required props contain 
* multi-valued props. 
*/
sql_rel *
rdf_rel_simple_combine_with_optional_cols(sql_allocator *sa, sql_rel *rel1_orig, sql_rel *rel2_orig){
	list *new_base_exps = new_exp_list(sa);
	list *new_exps = new_exp_list(sa); 
	node *e; 
	sql_rel *rel1 = NULL, *rel2 = NULL; 
	
	list *old_base1 = NULL, *old_base2 = NULL, *old_exp1 = NULL, *old_exp2 = NULL;  
	sql_rel *rel_base1 = NULL, *rel_base2 = NULL; 
	sql_rel *rel_combine = NULL; 
	sql_rel *rel_combine_select = NULL; 

	rel1 = rel1_orig; 

	while (rel1->op != op_select){
		assert (rel1->op == op_join); 
		rel1 = (sql_rel *) rel1->l; 
	}

	rel2 = rel2_orig; 

	assert(rel1->op == op_select); 
	assert(rel2->op == op_select);
	
	rel_base1 = (sql_rel *) rel1->l;
	rel_base2 = (sql_rel *) rel2->l; 
	
	assert(rel_base1->op == op_basetable); 
	assert(rel_base2->op == op_basetable); 

	old_base1 = rel_base1->exps; 
	old_base2 = rel_base2->exps; 
	
	old_exp1 = rel1->exps;
	old_exp2 = rel2->exps; 

	for (e = old_base1->h; e; e=e->next){
		sql_exp *tmpexp = (sql_exp *) e->data; 
		sql_exp *newexp = NULL;
		assert(tmpexp->type == e_column || tmpexp->type == e_atom); 

		newexp = exp_copy(sa, tmpexp); 
		append(new_base_exps, newexp); 
	}


	for (e = old_base2->h; e; e=e->next){
		sql_exp *tmpexp = (sql_exp *) e->data; 
		sql_exp *newexp = NULL; 
		assert(tmpexp->type == e_column || tmpexp->type == e_atom); 

		newexp = exp_copy(sa, tmpexp); 
		append(new_base_exps, newexp); 
	}

	
	for (e = old_exp1->h; e; e=e->next){
		sql_exp *tmpexp = (sql_exp *) e->data; 
		sql_exp *newexp = NULL; 

		newexp = exp_copy(sa, tmpexp); 
		append(new_exps, newexp); 
	}


	for (e = old_exp2->h; e; e=e->next){
		sql_exp *tmpexp = (sql_exp *) e->data; 
		sql_exp *newexp = NULL;

		newexp = exp_copy(sa, tmpexp); 
		append(new_exps, newexp); 
	}

	rel_combine = rel_copy(sa, rel1_orig); 
	if (rel_combine->op != op_select){
		assert(rel_combine->op == op_join); 

		rel_combine_select = rel_combine->l;

		while (rel_combine_select->op != op_select){
			assert (rel_combine_select->op == op_join); 
			rel_combine_select = (sql_rel *) rel_combine_select->l; 
		}

		assert(((sql_rel *)rel_combine_select->l)->op == op_basetable); 
		((sql_rel *)rel_combine_select->l)->exps = new_base_exps; 
		rel_combine_select->exps = new_exps; 
	}
	else {	
		assert(((sql_rel *)rel_combine->l)->op == op_basetable); 
		((sql_rel *)rel_combine->l)->exps = new_base_exps; 
		rel_combine->exps = new_exps; 
	}
	
	return rel_combine; 
}
