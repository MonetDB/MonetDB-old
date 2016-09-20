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

sql_rel* rel_graph_reaches(mvc *sql, sql_rel *rel, symbol *sq){
	str dump = NULL;
	dnode* lstoperands = NULL; // temp to navigate over the operands
	symbol* sym_qfrom = NULL; // the `from' column in the ast
	symbol* sym_qto = NULL; // the `to' column in the ast
	sql_exp* qfrom = NULL; // reference to the `from' column
	sql_exp* qto = NULL; // reference to the `to' column
	exp_kind dummy = {0}; // dummy param, required by rel_value_exp
	symbol* sym_edges_tbl = NULL; // the table edges in the ast
	sql_rel* tbl_edges = NULL; // the edges table exp~
	symbol* sym_edges_from = NULL; // reference to the `edges from' column in the ast
	symbol* sym_edges_to = NULL; // ref to the `edges to' column in the ast
	sql_exp* efrom = NULL; // ref to the edges column `from'
	sql_exp* eto= NULL; // ref to the edges column `to'
	sql_subtype* exptype = NULL; // the expression type for all columns
	sql_rel* result = NULL; // final output operator

	assert(sq->token == SQL_GRAPH_REACHES && "Invalid label in the AST, expected SQL_GRAPH_REACHES");

	// disable stmt caching for this query as WIP
	sql->caching = false;

	// let's see what we have got so far
	dump = rel_to_str(sql, rel);
	printf("[Semantic analysis] Input relation: %s", dump);

	lstoperands = sq->data.lval->h;
	sym_qfrom = lstoperands->data.sym; // first operand symbol( dlist( table, column ) )
	lstoperands = lstoperands->next; // move next
	sym_qto = lstoperands->data.sym; // second operand symbol( dlist( table, column ) )

	qfrom = rel_value_exp(sql, &rel, sym_qfrom, sql_where, dummy);
	if(!qfrom) return NULL; // cannot refer to qfrom
	qto = rel_value_exp(sql, &rel, sym_qto, sql_where, dummy);
	if(!qto) return NULL; // cannot refer to qto

	// assume for the time being qfrom and qto come from the same table
	// if they are not properly joined => result explosion!

	// edges table
	lstoperands = lstoperands->next;
	sym_edges_tbl = lstoperands->data.sym;
	tbl_edges = table_ref(sql, NULL, sym_edges_tbl, /* lateral = */ 0);
	if(!tbl_edges) return NULL; // error

	// find the columns in tbl_edges
	lstoperands = lstoperands->next;
	sym_edges_from = lstoperands->data.sym;
	efrom = rel_value_exp(sql, &tbl_edges, sym_edges_from, sql_where, dummy);
	if(!efrom) return NULL; // error
	lstoperands = lstoperands->next;
	sym_edges_to = lstoperands->data.sym;
	eto = rel_value_exp(sql, &tbl_edges, sym_edges_to, sql_where, dummy);
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

	// build the new operator graphjoin operator
	result = rel_spfw(sql, rel, tbl_edges, exp_spfw(sql, qfrom, qto, efrom, eto));

	// let's if what we are creating makes sense
	dump = rel_to_str(sql, result);
	printf("[Semantic analysis] Output relation: %s\n", dump);

	return result;
}
