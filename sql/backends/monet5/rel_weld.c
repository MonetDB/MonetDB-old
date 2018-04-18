/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "monetdb_config.h"
 
#include "rel_bin.h"
#include "rel_rel.h"
#include "rel_exp.h"
#include "rel_psm.h"
#include "rel_prop.h"
#include "rel_select.h"
#include "rel_updates.h"
#include "rel_optimizer.h"
#include "rel_weld.h"
#include "sql_env.h"
#include "sql_statement.h"
#include "opt_prelude.h"
#include "mal_namespace.h"
#include "mal_builder.h"
#include "mal_weld.h"

#define STR_BUF_SIZE 4096

/* From sql_statement.c */
#define meta(Id, Tpe)                \
	q = newStmt(mb, batRef, newRef); \
	q = pushType(mb, q, Tpe);        \
	Id = getArg(q, 0);               \
	list = pushArgument(mb, list, Id);

#define metaInfo(Id, Tpe, Val)          \
	p = newStmt(mb, batRef, appendRef); \
	p = pushArgument(mb, p, Id);        \
	p = push##Tpe(mb, p, Val);          \
	Id = getArg(p, 0);

typedef struct {
	int next_var;
	int result_var;
	int num_parens; /* number of parentheses */
	int num_loops;
	str builder;
	str program;
	unsigned long program_max_len;
	list *stmt_list;
	list *input_col_list;
} weld_state;

/* In practice we don't need separate produce and consume functions. The consume phase
 * begins when the produce call ends */
typedef int (*produce_func)(backend*, sql_rel*, weld_state*);

static produce_func getproduce_func(sql_rel *rel);
static int exps_to_weld(backend*, str, int*, list*, list*, str);
static int exp_to_weld(backend*, str, int*, sql_exp*, list*);

static void dump_program(weld_state *wstate) {
	FILE *f = fopen(tmpnam(NULL), "w");
	dumpWeldProgram(wstate->program, f);
	fclose(f);
}

static void append_weld_stmt(weld_state *wstate, str weld_stmt) {
	if (strlen(wstate->program) + strlen(weld_stmt) >= wstate->program_max_len) {
		wstate->program_max_len = strlen(wstate->program) + strlen(weld_stmt) + 1;
		wstate->program = realloc(wstate->program, wstate->program_max_len * sizeof(char));
	}
	wstate->program = strcat(wstate->program, weld_stmt);
}

static int exps_to_weld(backend *be, str weld_stmt, int* len, list *exps, list *stmt_list, str delim) {
	node *en;
	for (en = exps->h; en; en = en->next) {
		if (exp_to_weld(be, weld_stmt, len, en->data, stmt_list) != 0) {
			return -1;
		}
		if (en->next != NULL) {
			*len += sprintf(weld_stmt + *len, "%s", delim);
		}
	}
	return 0;
}

static str get_weld_cmp(int cmp) {
	switch(cmp) {
	case cmp_gt:       return ">";
	case cmp_gte:      return ">=";
	case cmp_lte:      return "<=";
	case cmp_lt:       return "<";
	case cmp_equal:    return "==";
	case cmp_notequal: return "!="; 
	default: /* not implementable */ return NULL;
	}
}

static str get_weld_func(sql_subfunc *f) {
	if (strcmp(f->func->imp, "+") == 0)
		return "+";
	else if (strcmp(f->func->imp, "-") == 0)
		return "-";
	else if (strcmp(f->func->imp, "*") == 0)
		return "*";
	else if (strcmp(f->func->imp, "/") == 0)
		return "/";
	/* TODO check for others that we might support through UDFs */
	return NULL;
}

/* Check wether the relation return a BAT as opposed to a single value */
static int rel_returns_bat(sql_rel *rel) {
	switch (rel->op) {
	case op_select:
		return rel_returns_bat(rel->l);
		return rel_returns_bat(rel->l);
	case op_groupby:
		return 0;
	case op_basetable:
	case op_topn:
	case op_join:
		return 1;
	default:
		return -1;
	}
}

static int exp_has_column(sql_exp *exp) {
	node *en;
	int ret = 0;
	switch (exp->type) {
	case e_atom:
	case e_psm:
		ret = 0;
		break;
	case e_column:
		ret = 1;
		break;
	case e_convert:
		ret = exp_has_column(exp->l);
		break;
	case e_cmp:
		if (exp->l) ret |= exp_has_column(exp->l);
		if (exp->r) ret |= exp_has_column(exp->r);
		if (exp->f) ret |= exp_has_column(exp->f);
		break;
	case e_func:
	case e_aggr:
		for (en = ((list*)exp->l)->h; en; en = en->next) {
			ret |= exp_has_column(en->data);
		}
		break;
	}
	return ret;
}

/* Produce Weld code from an expression. If the expression doesn't involve a column, call `exp_bin` on it
 * to produce a stmt and then use the result variable in the Weld program. This way we let MonetDB evaluate
 * complex conversions or atom expansions. */
static int exp_to_weld(backend *be, str weld_stmt, int *len, sql_exp *exp, list *stmt_list) {
	if (!exp_has_column(exp)) {
		stmt* sub = exp_bin(be, exp, NULL, NULL, NULL, NULL, NULL, NULL);
		*len += sprintf(weld_stmt + *len, "in%d", sub->nr);
		list_append(stmt_list, sub);
		return 0;
	}
	switch (exp->type) {
	case e_convert: {
		*len += sprintf(weld_stmt + *len, "%s(", getWeldType(exp->tpe.type->localtype));
		if (exp_to_weld(be, weld_stmt, len, exp->l, stmt_list) < 0) return -1;
		*len += sprintf(weld_stmt + *len, ")");
		break;
	}
	case e_cmp: {
		if (is_anti(exp)) {
			*len += sprintf(weld_stmt + *len, "(");
		}
		if (exp->f) {
			if (get_weld_cmp(swap_compare(range2lcompare(exp->flag))) == NULL) return -1;
			if (exp_to_weld(be, weld_stmt, len, exp->r, stmt_list) != 0) return -1;
			*len += sprintf(weld_stmt + *len, " %s ", get_weld_cmp(swap_compare(range2lcompare(exp->flag))));
			if (exp_to_weld(be, weld_stmt, len, exp->l, stmt_list) != 0) return -1;
			*len += sprintf(weld_stmt + *len, " && ");
			if (exp_to_weld(be, weld_stmt, len, exp->f, stmt_list) != 0) return -1;
			*len += sprintf(weld_stmt + *len, " %s ", get_weld_cmp(range2lcompare(exp->flag)));
			if (exp_to_weld(be, weld_stmt, len, exp->l, stmt_list) != 0) return -1;
		} else {
			if (get_weld_cmp(get_cmp(exp)) == NULL) return -1;
			if (exp_to_weld(be, weld_stmt, len, exp->l, stmt_list) != 0) return -1;
			*len += sprintf(weld_stmt + *len, " %s ", get_weld_cmp(get_cmp(exp)));
			if (exp_to_weld(be, weld_stmt, len, exp->r, stmt_list) != 0) return -1;
		}
		if (is_anti(exp)) {
			*len += sprintf(weld_stmt + *len, ") == false ");
		}
		break;
	}
	case e_column: {
		char col_name[256];
		sprintf(col_name, "%s_%s", exp->l ? (str)exp->l : (str)exp->r, (str)exp->r);
		*len += sprintf(weld_stmt + *len, "%s", col_name);
		break;
	}
	case e_func: {
		str weld_func = get_weld_func(exp->f);
		int is_infix = 0;
		if (strcmp(weld_func, "+") == 0 || strcmp(weld_func, "-") == 0 ||
			strcmp(weld_func, "*") == 0 || strcmp(weld_func, "/") == 0) {
			is_infix = 1;
		}
		if (is_infix) {
			sql_exp *left = ((list*)exp->l)->h->data;
			sql_exp *right = ((list*)exp->l)->h->next->data;
			int left_type = exp_subtype(left)->type->localtype;
			int right_type = exp_subtype(right)->type->localtype;
			/* MonetDB bug or not, we might still have mismatching types here */
			/* Left operand */
			if (left_type < right_type)
				*len += sprintf(weld_stmt + *len, "%s(", getWeldType(right_type));
			if (exp_to_weld(be, weld_stmt, len, left, stmt_list) != 0) return -1;
			if (left_type < right_type)
				*len += sprintf(weld_stmt + *len, ")");
			/* Operator */
			*len += sprintf(weld_stmt + *len, " %s ", weld_func);
			/* Right operand */
			if (right_type < left_type)
				*len += sprintf(weld_stmt + *len, "%s(", getWeldType(left_type));
			if (exp_to_weld(be, weld_stmt, len, right, stmt_list) != 0) return -1;
			if (right_type < left_type)
				*len += sprintf(weld_stmt + *len, ")");
		} else {
			*len += sprintf(weld_stmt + *len, "%s(", weld_func);
			if (exps_to_weld(be, weld_stmt, len, exp->l, stmt_list, ", ") != 0) return -1;
			*len += sprintf(weld_stmt + *len, ")");
		}
		break;
	}
	default: return -1;
	}
	return 0;
}

static int
base_table_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	stmt *sub = subrel_bin(be, rel, NULL);
	node *en;
	sql_exp *exp;
	char weld_stmt[STR_BUF_SIZE], col_name[256];
	int count;
	int len = sprintf(weld_stmt, "for(zip(");
	for (en = rel->exps->h; en; en = en->next) {
		exp = en->data;
		stmt *col = bin_find_column(be, sub, exp->l, exp->r);
		len += sprintf(weld_stmt + len, "in%d", col->nr);
		if (en->next != NULL) {
			len += sprintf(weld_stmt + len, ", ");
		}
	}
	/* builder and function header */
	++wstate->num_loops;
	len += sprintf(weld_stmt + len, "), %s, |b%d, i%d, n%d|", wstate->builder, wstate->num_loops,
				   wstate->num_loops, wstate->num_loops);
	/* extract named values from the tuple */
	for (en = rel->exps->h, count = 0; en; en = en->next, count++) {
		exp = en->data;
		sprintf(col_name, "%s_%s", (str)exp->l, (str)exp->r);
		if (rel->exps->h->next == NULL) {
			/* just a single column so n.$0 doesn't work */
			len += sprintf(weld_stmt + len, "let %s = n%d", col_name, wstate->num_loops);
		} else {
			len += sprintf(weld_stmt + len, "let %s = n%d.$%d", col_name, wstate->num_loops, count);
		}
		len += sprintf(weld_stmt + len, ";");
	}
	++wstate->num_parens;
	append_weld_stmt(wstate, weld_stmt);
	list_append(wstate->stmt_list, sub);
	return 0;
}

static int
select_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	/* === Produce === */
	produce_func input_produce = getproduce_func(rel->l);
	if (input_produce == NULL) return -1;
	if (input_produce(be, rel->l, wstate) != 0) return -1;

	/* === Consume === */
	char weld_stmt[STR_BUF_SIZE * 2];
	int len = sprintf(weld_stmt, "if((");
	node *en;
	for (en = rel->exps->h; en; en = en->next) {
		len += sprintf(weld_stmt + len, "(");
		if (exp_to_weld(be, weld_stmt, &len, en->data, wstate->stmt_list) < 0) return -1;
		len += sprintf(weld_stmt + len, ")");
		if (en->next != NULL) {
			len += sprintf(weld_stmt + len, " && ");
		}
	}
	/* negate the condition so that we have "true" on the right side and we can continue to append */
	len += sprintf(weld_stmt + len, ") == false, b%d, ", wstate->num_loops);
	++wstate->num_parens;
	append_weld_stmt(wstate, weld_stmt);
	return 0;
}

static int
project_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	/* === Produce === */
	produce_func input_produce = getproduce_func(rel->l);
	if (input_produce == NULL) return -1;
	if (input_produce(be, rel->l, wstate) != 0) return -1;

	/* === Consume === */
	char weld_stmt[STR_BUF_SIZE * 2];
	char col_name[256];
	int len = 0;
	node *en;
	sql_exp *exp;
	for (en = rel->exps->h; en; en = en->next) {
		exp = en->data;
		sprintf(col_name, "%s_%s", exp->rname ? exp->rname : exp->name, exp->name);
		len += sprintf(weld_stmt + len, "let %s = ", col_name);
		if (exp_to_weld(be, weld_stmt, &len, en->data, wstate->stmt_list) < 0) return -1;
		len += sprintf(weld_stmt + len, ";");
	}
	append_weld_stmt(wstate, weld_stmt);
	return 0;
}

static void
push_args(MalBlkPtr mb, InstrPtr *weld_instr, list *stmt_list, int *arg_names, int *idx) {
	node *en;
	for (en = stmt_list->h; en; en = en->next) {
		stmt* s = en->data;
		if (s->type == st_list) {
			push_args(mb, weld_instr, s->op4.lval, arg_names, idx);
		} else {
			*weld_instr = pushArgument(mb, *weld_instr, s->nr);
			arg_names[*idx] = s->nr;
			++*idx;
		}
	}
}

/* Mostly code from sql_statement.c */
static stmt *create_result_instr(backend *be, stmt *weld_program_stmt, list *exps) {
	node *n;
	InstrPtr q = NULL;
	MalBlkPtr mb = be->mb;
	// gather the meta information
	int tblId, nmeId, tpeId, lenId, scaleId, k, i;
	InstrPtr p = NULL, list;

	list = newInstruction(be->mb, sqlRef, resultSetRef);
	getArg(list, 0) = newTmpVariable(be->mb, TYPE_int);
	k = list->argc;
	meta(tblId, TYPE_str);
	meta(nmeId, TYPE_str);
	meta(tpeId, TYPE_str);
	meta(lenId, TYPE_int);
	meta(scaleId, TYPE_int);

	for (n = exps->h, i = 0; n; n = n->next, i++) {
		sql_exp *exp = n->data;
		sql_subtype *subtype = exp_subtype(exp);
		const char *tname = exp->rname ? exp->rname : exp->name;
		const char *sname = "sys";
		const char *_empty = "";
		const char *tn = (tname) ? tname : _empty;
		const char *sn = (sname) ? sname : _empty;
		const char *cn = exp->name;
		const char *ntn = sql_escape_ident(tn);
		const char *nsn = sql_escape_ident(sn);
		size_t fqtnl;
		char *fqtn = NULL;

		if (ntn && nsn && (fqtnl = strlen(ntn) + 1 + strlen(nsn) + 1)) {
			fqtn = NEW_ARRAY(char, fqtnl);
			if (fqtn) {
				snprintf(fqtn, fqtnl, "%s.%s", nsn, ntn);
				metaInfo(tblId, Str, fqtn);
				metaInfo(nmeId, Str, cn);
				metaInfo(tpeId, Str, (subtype->type->localtype == TYPE_void ? "char" : subtype->type->sqlname));
				metaInfo(lenId, Int, subtype->digits);
				metaInfo(scaleId, Int, subtype->scale);
				list = pushArgument(mb, list, weld_program_stmt->q->argv[i]);
				_DELETE(fqtn);
			} else
				q = NULL;
		} else
			q = NULL;
		c_delete(ntn);
		c_delete(nsn);
		if (q == NULL) return NULL;
	}
	// add the correct variable ids
	getArg(list, k++) = tblId;
	getArg(list, k++) = nmeId;
	getArg(list, k++) = tpeId;
	getArg(list, k++) = lenId;
	getArg(list, k) = scaleId;
	pushInstruction(mb, list);

	stmt *s = stmt_create(be->mvc->sa, st_output);
	s->op1 = weld_program_stmt;
	s->nr = getDestVar(list);
	s->q = list;
	return s;
}

static stmt *
root_produce(backend *be, sql_rel *rel)
{
	if (rel->op != op_project && rel->op != op_topn) return NULL;
	/* === Produce === */
	weld_state *wstate = calloc(1, sizeof(weld_state));
	wstate->program = calloc(1, 1);
	wstate->stmt_list = sa_list(be->mvc->sa);

	/* TODO handle TOPN */
	sql_rel *root = rel;
	node *en;
	char weld_stmt[STR_BUF_SIZE], col_name[256];
	int i, count, len = 0;
	int result_is_bat = rel_returns_bat(root);
	/* Save the builders in a variable */
	int result_var = wstate->next_var++;
	sprintf(weld_stmt, "let v%d = ", result_var);
	append_weld_stmt(wstate, weld_stmt);
	/* Prepare the builders */
	if (result_is_bat) {
		len += sprintf(weld_stmt + len, "{");
		for (en = root->exps->h; en; en = en->next) {
			sql_subtype *subtype = exp_subtype(en->data);
			len += sprintf(weld_stmt + len, "appender[%s]", getWeldType(subtype->type->localtype));
			if (en->next != NULL) {
				len += sprintf(weld_stmt + len, ", ");
			}
		}
		len += sprintf(weld_stmt + len, "}");
		wstate->builder = weld_stmt;
	}
	produce_func input_produce = getproduce_func(rel);
	if (input_produce == NULL) return NULL;
	if (input_produce(be, rel, wstate) != 0) {
		/* Can't convert this query */
		free(wstate->program);
		free(wstate);
		return NULL;
	}

	/* === Consume === */
	/* Append the results to the builders */
	len = 0;
	len += sprintf(weld_stmt + len, "{");
	for (en = root->exps->h, count = 0; en; en = en->next, count++) {
		sql_exp *exp = en->data;
		sprintf(col_name, "%s_%s", exp->rname ? exp->rname : exp->name, exp->name);
		if (result_is_bat) {
			len += sprintf(weld_stmt + len, "merge(b%d.$%d, %s)", wstate->num_loops, count, col_name);
		} else {
			len += sprintf(weld_stmt + len, "%s", col_name);
		}
		if (en->next != NULL) {
			len += sprintf(weld_stmt + len, ", ");
		}
	}
	len += sprintf(weld_stmt + len, "}");
	/* Close the parentheses */
	for (i = 0; i < wstate->num_parens; i++) {
		len += sprintf(weld_stmt + len, ")");
	}
	len += sprintf(weld_stmt + len, ";");
	/* Final result statement */
	if (result_is_bat) {
		len += sprintf(weld_stmt + len, "{");
		for (en = root->exps->h, count = 0; en; en = en->next, count++) {
			len += sprintf(weld_stmt + len, "result(v%d.$%d)", result_var, count);
			if (en->next != NULL) {
				len += sprintf(weld_stmt + len, ", ");
			}
		}
		len += sprintf(weld_stmt + len, "}");
	} else {
		/* Just the top variable */
		len += sprintf(weld_stmt + len, "v%d", result_var);
	}

	append_weld_stmt(wstate, weld_stmt);

	/* Build the Weld MAL instruction */
	InstrPtr weld_instr = newInstruction(be->mb, weldRef, weldRunRef);
	for (en = root->exps->h; en; en = en->next) {
		sql_subtype *subtype = exp_subtype(en->data);
		int type = subtype->type->localtype;
		if (result_is_bat) {
			type = newBatType(type);
		}
		weld_instr = pushReturn(be->mb, weld_instr, newTmpVariable(be->mb, type));
	}
	/* Push the arguments, first arg: the weld program, second arg: array of arg names */
	stmt *program_stmt = stmt_atom_string(be, wstate->program);
	int *arg_names = (int*)sa_alloc(be->mvc->sa, 1000 * sizeof(int)); /* Should be enough */
	weld_instr = pushArgument(be->mb, weld_instr, program_stmt->nr);
	weld_instr = pushPtr(be->mb, weld_instr, arg_names);
	int idx = 0;
	push_args(be->mb, &weld_instr, wstate->stmt_list, arg_names, &idx);
	pushInstruction(be->mb, weld_instr);

	list_append(wstate->stmt_list, program_stmt);
	stmt* weld_program_stmt = stmt_list(be, wstate->stmt_list);
	weld_program_stmt->q = weld_instr;

	dump_program(wstate);
	free(wstate->program);
	free(wstate);

	return create_result_instr(be, weld_program_stmt, root->exps);
}

produce_func getproduce_func(sql_rel *rel)
{
	switch (rel->op) {
		case op_basetable:
			return &base_table_produce;
		case op_select:
			return &select_produce;
		case op_project:
			return &project_produce;
		default:
			return NULL;
	}
}

stmt *
output_rel_weld(backend *be, sql_rel *rel )
{
	stmt* weld_program = root_produce(be, rel);
	if (weld_program == NULL) {
		fprintf(stderr, "output_rel_weld FAILED\n");
		return NULL;
	}
	be->mvc->type = Q_TABLE;
	return weld_program;
}
