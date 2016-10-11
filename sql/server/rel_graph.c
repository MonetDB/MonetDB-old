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


/*****************************************************************************
 *                                                                           *
 * CHEAPEST SUM, semantic phase                                              *
 *                                                                           *
 *****************************************************************************/

static bool error_reported(mvc* sql){ return (sql->session->status < 0); }


static sql_exp* bindg_ret(mvc *sql, sql_exp* bind1, sql_exp* bind2){
	if (error_reported(sql)){ // an error already occurred
		return NULL;
	} else if(bind1 && bind2){
		return sql_error(sql, ERR_AMBIGUOUS, "Ambiguous expression for CHEAPEST SUM: %s, %s", exp_name(bind1), exp_name(bind2));
	} else if(bind1){
		return bind1;
	} else {
		return bind2; // either if it has a value or it is null */
	}
}

static sql_exp* bindg_exp(mvc *sql, sql_exp *exp, symbol *sym){
	graph_join *g;
	sql_exp* e;
	exp_kind exp_kind_value = {type_value, card_column, TRUE};

	assert(exp && "Expected an expression");

	if(exp->type != e_cmp || get_cmp(exp) != cmp_filter_graph){
		// this is not a graph join, move along
		return NULL;
	}

	g = exp->f;

	// try to bind the expression
	e = rel_value_exp(sql, &(g->edges), sym, sql_sel, exp_kind_value);
	if(!e){ return NULL; }

	// an expression has already been bound
	if(g->cost){
		return sql_error(sql, 02, "TODO: At the moment you cannot bind multiple CHEAPEST SUM expression against the same join");
	}

	// found it!
	g->cost = exp_label(sql->sa, e, ++sql->label);
	return g->cost;
}


static sql_exp* bindg_exps(mvc *sql, list *exps, symbol *sym){
	sql_exp *result = NULL;

	// edge case
	if(!exps || error_reported(sql)) return NULL;

	for(node* n = exps->h; n; n = n->next){
		sql_exp *bound = bindg_exp(sql, n->data, sym);
		result = bindg_ret(sql, result, bound);
		if(error_reported(sql)) return NULL; // ERROR! => stop processing
	}

	return result;
}

static sql_exp* bindg_rel(mvc *sql, sql_rel* relation, symbol *sym){
	// edge case
	if(!relation || error_reported(sql)) return NULL;

	switch(relation->op){
	case op_full:
	case op_left:
	case op_right:
	case op_semi:
		assert("I haven't thought about these cases yet");
		break;
	case op_join: {
		sql_exp *exp1 = NULL, *exp2 = NULL, *exp3 = NULL, *ret = NULL;

		exp1 = bindg_rel(sql, relation->l, sym);
		exp2 = bindg_rel(sql, relation->r, sym);
		ret = bindg_ret(sql, exp1, exp2);
		exp3 = bindg_exps(sql, relation->exps, sym);
		return bindg_ret(sql, ret, exp3);
	} break;
	case op_select: {
		sql_exp* exp1 = bindg_exps(sql, relation->exps, sym);
		sql_exp* exp2 = bindg_rel(sql, relation->l, sym);
		return bindg_ret(sql, exp1, exp2);
	} break;
	case op_groupby:
		// move up the tree
		return bindg_rel(sql, relation->l, sym);
	default:
		return NULL;
	}

	return NULL; // silent the warning
}


sql_exp* rel_graph_cheapest_sum(mvc *sql, sql_rel **rel, symbol *sym, int context){
	sql_exp* exp_bound = NULL;
	sql_exp* result = NULL;

	// Check the context is the SELECT clause
	if(context != sql_sel){
		sql_error(sql, 02, "CHEAPEST SUM is only allowed inside the SELECT clause");
		return NULL;
	}

	// Check whether an argument has been specified
	if(!sym->data.sym){
		// TODO this should be already handled by the parser (i.e. it's not part of the language)
		sql_error(sql, 02, "Empty argument for CHEAPEST SUM");
		return NULL;
	}

	// Find the relation where the sub the expression binds to
	assert(is_project((*rel)->op) && "Unexpected relation type");
	exp_bound = bindg_rel(sql, (*rel)->l, sym->data.sym);
	if(!exp_bound){ return NULL; }

	// Create the new column
	result = exp_column(sql->sa, NULL, exp_bound->name, exp_subtype(exp_bound), (*rel)->card, /* has_nil = */ FALSE, /* is_intern = */ FALSE);
	return result;
}
