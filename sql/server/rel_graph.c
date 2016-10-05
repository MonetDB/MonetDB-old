/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "rel_graph.h"

#include <assert.h>

#include "rel_dump.h"
#include "rel_exp.h"
#include "rel_rel.h"
#include "rel_select.h"


sql_rel* rel_graph_reaches(mvc *sql, sql_rel *rel, symbol *sq, int context){
	dnode* lstoperands = NULL; // temp to navigate over the operands
	symbol* sym_qfrom = NULL; // the `from' column in the ast
	symbol* sym_qto = NULL; // the `to' column in the ast
	sql_exp* qfrom = NULL; // reference to the `from' column
	sql_exp* qto = NULL; // reference to the `to' column
	symbol* sym_edges_tbl = NULL; // the table edges in the ast
	sql_rel* tbl_edges = NULL; // the edges table exp~
	symbol* sym_edges_from = NULL; // reference to the `edges from' column in the ast
	symbol* sym_edges_to = NULL; // ref to the `edges to' column in the ast
	sql_exp* efrom = NULL; // ref to the edges column `from'
	sql_exp* eto= NULL; // ref to the edges column `to'
	sql_subtype* exptype = NULL; // the expression type for all columns
	sql_exp* graph_join = NULL; // the produced predicate for the join
	sql_rel* result = NULL; // final output operator
	exp_kind exp_kind_value = {type_value, card_column, TRUE};
	int use_views_old = 0; // temporary to remember the old value of sql->use_views

	assert(sq->token == SQL_GRAPH_REACHES && "Invalid label in the AST, expected SQL_GRAPH_REACHES");

	// disable stmt caching for this query as WIP
	sql->caching = false;

	// let's see what we have got so far
	printf("[Semantic analysis] Input relation: %s", rel_to_str(sql, rel));

	lstoperands = sq->data.lval->h;
	sym_qfrom = lstoperands->data.sym; // first operand symbol( dlist( table, column ) )
	lstoperands = lstoperands->next; // move next
	sym_qto = lstoperands->data.sym; // second operand symbol( dlist( table, column ) )

	qfrom = rel_value_exp(sql, &rel, sym_qfrom, context, exp_kind_value);
	if(!qfrom) return NULL; // cannot refer to qfrom
	qto = rel_value_exp(sql, &rel, sym_qto, context, exp_kind_value);
	if(!qto) return NULL; // cannot refer to qto

	// edges table
	lstoperands = lstoperands->next;
	sym_edges_tbl = lstoperands->data.sym;
	use_views_old = sql->use_views; // TODO: ask Ni3ls the rationale of this trick
	sql->use_views = TRUE; // table_ref can find references declared with `WITH'
	tbl_edges = table_ref(sql, NULL, sym_edges_tbl, /* lateral = */ 0);
	sql->use_views = use_views_old; // restore the previous value
	if(!tbl_edges) return NULL; // error

	// find the columns in tbl_edges
	lstoperands = lstoperands->next;
	sym_edges_from = lstoperands->data.sym;
	efrom = rel_value_exp(sql, &tbl_edges, sym_edges_from, context, exp_kind_value);
	if(!efrom) return NULL; // error
	lstoperands = lstoperands->next;
	sym_edges_to = lstoperands->data.sym;
	eto = rel_value_exp(sql, &tbl_edges, sym_edges_to, context, exp_kind_value);
	if(!eto) return NULL; // error

	// check the types match
	exptype = exp_subtype(efrom);
	if(subtype_cmp(exptype, exp_subtype(eto)) < 0){
		return sql_error(sql, 03, "Types for the edge columns `%s' and `%s' do not match", efrom->name, eto->name);
	}
	qfrom = rel_check_type(sql, exptype, qfrom, type_equal);
	if(!qfrom) return NULL; // cannot convert qfrom into the same type of efrom
	qto = rel_check_type(sql, exptype, qto, type_equal);
	if(!qto) return NULL; // cannot convert qto into the same type of eto

	// build the new operator graph join operator
	graph_join = exp_graph_join(
			/* memory pool = */ sql->sa,
			/* list query from = */ exp2list(sql->sa, qfrom),
			/* list query to = */ exp2list(sql->sa, qto),
			/* ref table exp~ edges = */ tbl_edges,
			/* list edges from = */ exp2list(sql->sa, efrom),
			/* list edges to = */ exp2list(sql->sa, eto)
	);
	if(!graph_join) return NULL;

	result = rel_push_join(sql, rel, qfrom, qto, NULL, graph_join);

	// let us see if what we are creating makes sense
	printf("[Semantic analysis] Output relation: %s\n", rel_to_str(sql, result));

	return result;
}
