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
#include "sql_relation.h" // rel_graph

// DEBUG ONLY -- copy & paste from sql_gencode.c + decorate = TRUE
str
rel2str1( mvc *sql, sql_rel *rel)
{
	buffer *b;
	stream *s = buffer_wastream(b = buffer_create(1024), "rel_dump");
	list *refs = sa_list(sql->sa);
	char *res = NULL;

	rel_print_refs(sql, s, rel, 0, refs, TRUE);
	rel_print_(sql, s, rel, 0, refs, TRUE);
	mnstr_printf(s, "\n");
	res = buffer_get_buf(b);
	buffer_destroy(b);
	mnstr_destroy(s);
	return res;
}


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
    printf("[Semantic analysis] Input relation: %s", rel2str1(sql, rel));

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
    	return sql_error(sql, 42, "["__FILE__ ":%d] select/filter syntax not implemented yet", __LINE__);
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
    graph_ptr = (sql_graph*) sa_alloc(sql->sa, sizeof(sql_graph));
    if(!graph_ptr) { return sql_error(sql, 03, "Cannot allocate rel_graph"); }
    memset(graph_ptr, 0, sizeof(sql_graph));
    result = (sql_rel*) graph_ptr;
    sql_ref_init(&result->ref);
    result->op = op_graph_select;
    result->l = rel;
    exp_ptr = (sql_exp*) sa_alloc(sql->sa, sizeof(sql_exp));
    if(!exp_ptr) { return sql_error(sql, 03, "Cannot allocate sql_exp [e_graph] "); }
    memset(exp_ptr, 0, sizeof(sql_exp));
    exp_ptr->type = e_graph;
    exp_ptr->card = CARD_MULTI;
    exp_ptr->l = sa_list(sql->sa);
    list_append(exp_ptr->l, qfrom);
    exp_ptr->r = sa_list(sql->sa);
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
    printf("[Semantic analysis] Output relation: %s\n", rel2str1(sql, result));

    return rel;
}

/*****************************************************************************
 *                                                                           *
 * CHEAPEST SUM, semantic phase                                              *
 *                                                                           *
 *****************************************************************************/

//static bool error_reported(mvc* sql){ return (sql->session->status < 0); }
//
//static sql_exp* bindg_ret(mvc *sql, sql_exp* bind1, sql_exp* bind2){
//        if (error_reported(sql)){ // an error already occurred
//                return NULL;
//        } else if(bind1 && bind2){
//                return sql_error(sql, ERR_AMBIGUOUS, "Ambiguous expression for CHEAPEST SUM: %s, %s", exp_name(bind1), exp_name(bind2));
//        } else if(bind1){
//                return bind1;
//        } else {
//                return bind2; // either if it has a value or it is null */
//        }
//}
//
//static sql_exp* bindg_filter_graph(mvc *sql, sql_exp *exp, dlist *parse_tree){
//        const char* table_ref = NULL; // the table referred (optional)
//        symbol* expr_weight = NULL; // the expression inside CHEAPEST SUM ( ... );
//        graph_join* g = NULL;
//        sql_exp* e = NULL;
//        exp_kind exp_kind_value = {type_value, card_column, TRUE};
//
//        assert(exp && "Expected an expression");
//        assert(exp->type == e_cmp && get_cmp(exp) == cmp_filter_graph && "Expected a graph filter exp~");
//        assert(parse_tree && "The input argument parse_tree is NULL");
//        assert(parse_tree->cnt == 2 && "Expected two nodes in the root of the parse tree");
//
//        g = exp->f;
//
//        table_ref = parse_tree->h->data.sval;
//        expr_weight = parse_tree->h->next->data.sym;
//
//        if (table_ref){ // use the table name to refer to the edge table
//                const char* tname = rel_name(g->edges);
//
//                // TODO shall we consider the schema as well?
//                assert(tname != NULL);
//                if(strcmp(tname, table_ref) == 0){
//                        // force the binding against this relation
//                        e = rel_value_exp(sql, &(g->edges), expr_weight, sql_sel, exp_kind_value);
//                        if(!e){ return sql_error(sql, 02, "Cannot bind the cheapest sum expression in the subquery `%s'", tname); }
//                }
//        } else { // table name not given
//                // try to bind the expression a la `best effort'
//                e = rel_value_exp(sql, &(g->edges), expr_weight, sql_sel, exp_kind_value);
//        }
//
//        // did we bind our parse tree?
//        if(e){
//                if(g->cost){ // existing limitation, an expression has already been bound
//                        return sql_error(sql, 02, "TODO: At the moment you cannot bind multiple CHEAPEST SUM expression against the same join");
//                }
//
//                // found it!
//                g->cost = exp_label(sql->sa, e, ++sql->label);
//                return g->cost;
//
//        } else { // no, we didn't bind it
//                return NULL;
//        }
//}
//
//static sql_exp* bindg_exp(mvc *sql, sql_exp *exp, dlist *parse_tree){
//        if(exp->type == e_cmp && get_cmp(exp) == cmp_filter_graph){
//                // ok this is a graph join
//                return bindg_filter_graph(sql, exp, parse_tree);
//        } else {
//                // this is not a graph join, move along
//                return NULL;
//        }
//}
//// DEBUG ONLY -- copy & paste from sql_gencode.c + decorate = TRUE

//static sql_exp* bindg_exps(mvc *sql, list *exps, dlist *parse_tree){
//        sql_exp *result = NULL;
//
//        // edge case
//        if(!exps || error_reported(sql)) return NULL;
//
//        for(node* n = exps->h; n; n = n->next){
//                sql_exp *bound = bindg_exp(sql, n->data, parse_tree);
//                result = bindg_ret(sql, result, bound);
//                if(error_reported(sql)) return NULL; // ERROR! => stop processing
//        }
//
//        return result;
//}
//
//static sql_exp* bindg_rel(mvc *sql, sql_rel* relation, dlist *parse_tree){
//        // edge case
//        if(!relation || error_reported(sql)) return NULL;
//
//        switch(relation->op){
//        case op_full:
//        case op_left:
//        case op_right:
//        case op_semi:
//                assert("I haven't thought about these cases yet");
//                break;
//        case op_join: {
//                sql_exp *exp1 = NULL, *exp2 = NULL, *exp3 = NULL, *ret = NULL;
//
//                exp1 = bindg_rel(sql, relation->l, parse_tree);
//                exp2 = bindg_rel(sql, relation->r, parse_tree);
//                ret = bindg_ret(sql, exp1, exp2);
//                exp3 = bindg_exps(sql, relation->exps, parse_tree);
//                return bindg_ret(sql, ret, exp3);
//        } break;
//        case op_select: {
//                sql_exp* exp1 = bindg_exps(sql, relation->exps, parse_tree);
//                sql_exp* exp2 = bindg_rel(sql, relation->l, parse_tree);
//                return bindg_ret(sql, exp1, exp2);
//        } break;
//        case op_groupby:
//                // move up in the tree
//                return bindg_rel(sql, relation->l, parse_tree);
//        default:
//                return NULL;
//        }
//
//        return NULL; // silent the warning
//}
//
//sql_exp* rel_graph_cheapest_sum(mvc *sql, sql_rel **rel, symbol *sym, int context){
//        sql_exp* exp_bound = NULL;
//        sql_exp* result = NULL;
//
//        assert(sym->data.lval != NULL && "CHEAPEST SUM: empty parse tree");
//
//        // Check the context is the SELECT clause
//        if(context != sql_sel){
//                return sql_error(sql, 02, "CHEAPEST SUM is only allowed inside the SELECT clause");
//        }
//
//        // Find the relation where the sub the expression binds to
//        assert(is_project((*rel)->op) && "Unexpected relation type");
//        exp_bound = bindg_rel(sql, (*rel)->l, sym->data.lval);
//        if(!exp_bound){ return NULL; }
//
//        // Create the new column
//        result = exp_column(sql->sa, NULL, exp_bound->name, exp_subtype(exp_bound), (*rel)->card, /* has_nil = */ FALSE, /* is_intern = */ FALSE);
//        return result;
//}


sql_exp* rel_graph_cheapest_sum(mvc *sql, sql_rel **rel, symbol *sym, int context){
	(void) sql;
	(void) rel;
	(void) sym;
	(void) context;

	return NULL;
}
