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
#include "sql_mem.h" // sql_ref_init
#include "sql_relation.h" // rel_graph

/*****************************************************************************
 *                                                                           *
 * Create the graph operator                                                 *
 *                                                                           *
 *****************************************************************************/

sql_graph* rel_graph_create(sql_allocator *sa) {
	sql_graph *r = SA_NEW(sa, sql_graph);
	if(!r) return NULL;
	memset(r, 0, sizeof(sql_graph));
	sql_ref_init(&(r->relation.ref));
	return r;
}

sql_graph* rel_graph_move(mvc* sql, sql_rel* rel, sql_rel* l, sql_rel* r, sql_exp* e){
	sql_graph* graph_old = NULL;
	sql_graph* graph_new = NULL;
	sql_rel* graph_rel = NULL;

	assert(rel && is_graph(rel->op));
	graph_old = (sql_graph*) rel;
	graph_new = rel_graph_create(sql->sa);
	if(!graph_new) return NULL;
	memcpy((char*) graph_new + sizeof(sql_ref), (char*) graph_old + sizeof(sql_ref), sizeof(sql_graph) - sizeof(sql_ref));
	graph_rel = &(graph_new->relation);
	graph_rel->l = l;
	graph_rel->r = r;
	graph_rel->exps = new_exp_list(sql->sa);
	list_append(graph_rel->exps, e);

	return graph_new;
}

sql_rel* rel_graph_move2rel(mvc* sql, sql_rel* rel, sql_rel* l, sql_rel* r, sql_exp* e){
	return (sql_rel*) rel_graph_move(sql, rel, l, r, e);
}

/*****************************************************************************
 *                                                                           *
 * Parse the REACHES clause                                                  *
 *                                                                           *
 *****************************************************************************/
sql_rel* rel_graph_reaches(mvc *sql, sql_rel *rel, symbol *sq, int context){
	// TODO handle edge components defined with multiple attributes
	// this needs changes in the parser to accept list of columns & scalars

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
    sql_exp* eto = NULL; // ref to the edges column `to'
    sql_subtype* exptype = NULL; // the expression type for all columns
    exp_kind exp_kind_value = {type_value, card_column, TRUE};
    sql_graph* graph_ptr = NULL; // the created operator
    sql_exp* exp_ptr = NULL; // the created expression ( x reaches y )
    sql_rel* result = NULL; // final output operator
    int use_views_old = 0; // temporary to remember the old value of sql->use_views

    assert(sq->token == SQL_GRAPH_REACHES && "Invalid label in the AST, expected SQL_GRAPH_REACHES");

    // disable stmt caching for this query as WIP
    sql->caching = false;

    // let's see what we have got so far
    printf("[Semantic analysis] [reaches] Input relation: %s", dump_rel(sql, rel));

    lstoperands = sq->data.lval->h;
    sym_qfrom = lstoperands->data.sym; // first operand symbol( dlist( table, column ) )
    lstoperands = lstoperands->next; // move next
    sym_qto = lstoperands->data.sym; // second operand symbol( dlist( table, column ) )

    qfrom = rel_value_exp(sql, &rel, sym_qfrom, context, exp_kind_value);
    if(!qfrom) return NULL; // cannot refer to qfrom
    qto = rel_value_exp(sql, &rel, sym_qto, context, exp_kind_value);
    if(!qto) return NULL; // cannot refer to qto
    // TODO: to be handled with graph_select
    if(qfrom->card != CARD_MULTI || qto->card != CARD_MULTI){
    	return sql_error(sql, 42, "["__FILE__ ":%d] select/filter semantic not allowed for the time being", __LINE__);
    }

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
    graph_ptr = rel_graph_create(sql->sa);
    if(!graph_ptr) { return sql_error(sql, 03, "Cannot allocate rel_graph"); }
    result = (sql_rel*) graph_ptr;
    result->op = op_graph_select;
    result->l = rel;
    exp_ptr = exp_graph(sql, sa_list(sql->sa), sa_list(sql->sa));
    if(!exp_ptr) { return sql_error(sql, 03, "Cannot allocate sql_exp [e_graph] "); }
    list_append(exp_ptr->l, qfrom);
    list_append(exp_ptr->r, qto);
    result->exps = sa_list(sql->sa); // by convention exps has to be a list, even it contains only one item
    list_append(result->exps, exp_ptr);
    result->card = CARD_MULTI;
    result->nrcols = rel->nrcols;
    graph_ptr->edges = tbl_edges;
    graph_ptr->efrom = sa_list(sql->sa);
    list_append(graph_ptr->efrom, efrom);
    graph_ptr->eto = sa_list(sql->sa);
    list_append(graph_ptr->eto, eto);
    graph_ptr->spfw = sa_list(sql->sa); // empty list

    // let us see if what we are creating makes sense
    printf("[Semantic analysis] [reaches] Output relation: %s\n", dump_rel(sql, result));

    return result;
}

/*****************************************************************************
 *                                                                           *
 * CHEAPEST SUM, semantic phase                                              *
 *                                                                           *
 *****************************************************************************/

static bool error_reported(mvc* sql){ return (sql->session->status < 0); }

static list* bind_shortest_path_return(mvc *sql, list* bind1, list* bind2){
	if (error_reported(sql)){ // an error already occurred
		return NULL;
	} else if(list_length(bind1) && list_length(bind2)){
		return sql_error(sql, ERR_AMBIGUOUS, "Ambiguous expression for CHEAPEST SUM: %s, %s", exp_name(bind1->h->data), exp_name(bind2->h->data));
	} else if(list_length(bind1)){
		return bind1;
	} else {
		return bind2; // either if it has a value or it is null */
	}
}

static list* bind_shortest_path_graph(mvc *sql, sql_graph *graph, dlist *parse_tree, bool compute_path){
	const char* table_ref = NULL; // the table referred (optional)
	symbol* expr_weight = NULL; // the expression inside CHEAPEST SUM ( ... );
	sql_rel* edges = NULL; // the table expression representing the edges
	sql_exp* e = NULL; // the expression being bound
	exp_kind exp_kind_value = {type_value, card_column, TRUE}; // rel_value_exp parameters
	list* result = NULL; // the final result as list: [cost, path]

	// init
	table_ref = parse_tree->h->data.sval;
	expr_weight = parse_tree->h->next->data.sym;
	edges = graph->edges;

	if(table_ref != NULL){
		const char* tname = rel_name(edges);
		if(strcmp(tname, table_ref) == 0){
			e = rel_value_exp(sql, &edges, expr_weight, sql_sel, exp_kind_value);
			if(!e){ return sql_error(sql, 02, "Cannot bind the cheapest sum expression in the subquery `%s'", tname); }
		}
	} else { // table_ref == NULL
		// try to bind the expression a la `best effort'
		e = rel_value_exp(sql, &edges, expr_weight, sql_sel, exp_kind_value);
	}

	if(e){ // success
		node* duplicate = NULL;
		sql_exp* expr_csum = e; // the expression associated to the edge table to compute the weights
		sql_exp* column_cost = NULL; // the resulting expression for the cost
		sql_exp* column_path = NULL; // the resulting expression for the spfw
		bool is_bfs = false; // is this a BFS?

		// if this is an atom, then perform a simple BFS and multiply the final result by the original atom
		if(exp_is_atom(e)){
			e = exp_atom_lng(sql->sa, 1); // regardless of its value or type, the important part for the codegen is that this is an atom
			is_bfs = true;
		}

		// before creating a new spfw, search for duplicates in the list of expressions
		// already registered
		duplicate = list_find(graph->spfw, e, (fcmp) exp_match_exp_cmp);

		// we didn't find a duplicate, add to the list of expressions we need to compute
		// the shortest path
		if(!duplicate){
			e = exp_label(sql->sa, e, ++sql->label);
			list_append(graph->spfw, e);
		} else { // this is a duplicate indeed
			e = duplicate->data;
		}

		e->flag |= GRAPH_EXPR_COST;
		column_cost = exp_column(sql->sa, exp_relname(e), exp_name(e), exp_subtype(e), graph->relation.card, /* has_nil = */ FALSE, /* is_intern = */ FALSE);

		// if this is a bfs, we need to multiply the result by the actual value
		if(is_bfs){
			sql_subfunc* f_mult = NULL;

			// cast the result to the expected type
			if(type_cmp(exp_subtype(column_cost)->type, exp_subtype(expr_csum)->type) != 0 /* 0 = same type */){
				sql_type* column_cost_t = exp_subtype(column_cost)->type;
				sql_type* expr_csum_t = exp_subtype(expr_csum)->type;
				// currently a bfs returns a lng, so cast to lng unless it is even bigger
				if(expr_csum_t->eclass == EC_NUM && expr_csum_t->digits < column_cost_t->digits) {
					expr_csum = exp_convert(sql->sa, expr_csum, exp_subtype(expr_csum), exp_subtype(column_cost));
				} else {
					column_cost = exp_convert(sql->sa, column_cost, exp_subtype(column_cost), exp_subtype(expr_csum));
				}
			}

			f_mult = sql_bind_func(sql->sa, /*mvc_bind_schema(sql, "sys")*/ NULL, "sql_mul", exp_subtype(expr_csum), exp_subtype(column_cost), F_FUNC);
			assert(f_mult != NULL && "Cannot bind the multiply function");
			column_cost = exp_binop(sql->sa, expr_csum, column_cost, f_mult);
		}


		result = sa_list(sql->sa);
		list_append(result, column_cost);

		// shortest path
		if(compute_path){

			// first check whether we already requested to compute the path for the
			// associated cost expr
			if(duplicate && duplicate->next){
				// the convention is to always record the column for the path immediately
				// after the one for the cost
				sql_exp* candidate = duplicate->next->data;
				if (candidate->flag & GRAPH_EXPR_SHORTEST_PATH){
					column_path = candidate;
				}
			}

			// otherwise we need to create an hoc column & register in the list graph->spfw
			// to represent the path to compute
			if(!column_path) {
				const char* label = make_label(sql->sa, ++sql->label);
				sql_subtype* type = sql_bind_subtype(sql->sa, "nested_table", 0, 0);

				assert(type != NULL && "Cannot find the 'nested_table' type");
				// record the columns composing the nested_table type, from the edge table
				type->attributes = exps_copy(sql->sa, rel_projections(sql, edges, NULL, /* settname = */ true, /* intern = */ false));
				// set the relation name to label
				for(node* n = type->attributes->h; n; n = n->next){
					sql_exp* ne = n->data;
					ne->rname = label;
				}

				column_path = exp_column(sql->sa, label, label, type, CARD_MULTI, /* has_nil = */ FALSE, /* is_intern = */ FALSE);
				column_path->flag |= GRAPH_EXPR_SHORTEST_PATH;

				// record the column in the 'spfw' list
				if(duplicate){
					list_insert_after(graph->spfw, duplicate, column_path);
				} else {
					list_append(graph->spfw, column_path);
				}
			}

			column_path = exp_column(sql->sa, exp_relname(column_path), exp_name(column_path), exp_subtype(column_path), graph->relation.card, /* has_nil = */ FALSE, /* is_intern = */ FALSE);
			list_append(result, column_path);
		}

		return result;
	} else { // it did not bind to the associated edge table
		return NULL;
	}

}

// walk up in the relation tree and bind the given cheapest sum symbol `parse_tree' to a graph operator
// it returns a list having 1 or 2 members:
// 1- the expression representing the 'cost' of the path
// 2- when compute_path = true, the second member is a nested table object representing the computed path
static list* bind_shortest_path_recursion(mvc *sql, sql_rel* relation, dlist *parse_tree, bool compute_path){
	// edge case
	if(!relation || error_reported(sql)) return NULL;

	assert(relation->op != op_graph_join && "op_graph_join is not allowed in the semantic phase");

	switch(relation->op){
	case op_graph_select: { // base case
		list *lst1 = NULL, *lst2 = NULL;
		// base case
		lst1 = bind_shortest_path_graph(sql, (sql_graph*) relation, parse_tree, compute_path);
		// even if it bound the expression to this operator, we propagate up in the tree
		// to check and report ambiguities
		lst2 = bind_shortest_path_recursion(sql, relation->l, parse_tree, compute_path);
		return bind_shortest_path_return(sql, lst1, lst2);
	} break;
	case op_full:
	case op_left:
	case op_right:
	case op_join:
	case op_select: {
		list *lst1 = NULL, *lst2 = NULL;

		lst1 = bind_shortest_path_recursion(sql, relation->l, parse_tree, compute_path);
		lst2 = bind_shortest_path_recursion(sql, relation->r, parse_tree, compute_path);
		return bind_shortest_path_return(sql, lst1, lst2);
	} break;
	case op_semi:
	case op_anti:
	case op_groupby:
		// move up in the tree
		return bind_shortest_path_recursion(sql, relation->l, parse_tree, compute_path);
	default:
		return NULL;
	}

	return NULL; // silent the compiler warning
}


list* rel_graph_shortest_path(mvc *sql, sql_rel *rel, symbol *sym, int context, bool compute_path){
	list* result = NULL;

	assert(sym->data.lval != NULL && "CHEAPEST SUM: empty parse tree");

	// Check the context is the SELECT clause
	if(context != sql_sel){
		return sql_error(sql, 106, "CHEAPEST SUM is only allowed inside the SELECT clause");
	}

	// Find the relation where the sub the expression binds to
	assert(is_project(rel->op) && "Unexpected relation type");
	result = bind_shortest_path_recursion(sql, rel->l, sym->data.lval, compute_path);

	// If it didn't bind the exp~, prepare an error message if it was not already constructed
	if(!result && !error_reported(sql)){
		return sql_error(sql, 106, "Cannot bind the expression in CHEAPEST SUM");
	} else {
		return result; // this can be an exp~ or NULL + an error set
	}
}


/*****************************************************************************
 *                                                                           *
 * Bind an expression produced by a graph operator                           *
 *                                                                           *
 *****************************************************************************/
// Attempt to bind the column with table name `tbl_name' and column name `col_name'
// to one of the expression generated by the graph operator (i.e. a shortest path).
// It returns the expression bound if found, NULL otherwise.
sql_exp* graph_bind_spfw(sql_rel* rel, const char* tbl_name, const char* col_name){
	sql_graph* graph_ptr = NULL;

	assert(rel && is_graph(rel->op) && "The given operator is not a graph");
	graph_ptr = (sql_graph*) rel;

	// all expressions associated to a graph operator need to have a table name
	if(!tbl_name) return NULL;

	for(node* n = graph_ptr->spfw->h; n; n = n->next){
		sql_exp* candidate = n->data;

		if(candidate->flag & GRAPH_EXPR_COST){
			if(strcmp(tbl_name, candidate->rname) == 0 && strcmp(col_name, candidate->name) == 0){
				return candidate;
			}
		} else if (candidate->flag & GRAPH_EXPR_SHORTEST_PATH){
			assert(candidate->type == e_column && "Paths should be columns");
			if(strcmp(tbl_name, candidate->rname) == 0){
				assert(candidate->tpe.type->eclass == EC_NESTED_TABLE && "Expected a nested table (class)");
				assert(candidate->tpe.attributes != NULL && "Expected a nested table (attrs)");
				if(strcmp(col_name, candidate->name) == 0){ // fetch the nested table
					return candidate;
				}

				candidate = exps_bind_column(candidate->tpe.attributes, col_name, NULL);
				assert(candidate != NULL && "How was it able to match the table name and not the column name?");
				return candidate;
			}
		}
	}

	// not found
	return NULL;
}
