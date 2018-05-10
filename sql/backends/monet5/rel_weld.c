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

/* The code here generates a Weld program by parsing the relational algebra tree and using the
 * produce-consume model as described in the "Efficiently Compiling Efficient Query Plans for Modern
 * Hardware" paper. Some notes on the implementation:
 * -  In practice we don't need to separate produce and consume functions. The consume phase begins
 *    when the produce function call ends
 * -  We still produce a MAL program with one big "weld.run" instruction which implements the query logic
 * -  We need to generate textual Weld code, which is error prone and a pain in C. Each operator generates
 *    code on the fly, i.e. we append Weld statements to a string buffer which will eventually result in a
 *    complete program.
 * -  op_basetable is also handled by MonetDB, we rely on the existing code for reading the BATs
 * -  When expressions don't involve columns, we let MonetDB handle them. This is useful for evaluating complex 
 *    atoms
 * -  String BATs are backed by 2 arrays: one with the strings and the other with the offsets. At the end of the Weld
 *    program we need to also return the strings array so that we can later build a string BAT, so special care is
 *    need to ensure that the strings array is referenced correctly throughout the program.
 * -  Weld only supports sorting in ascending order for now
 * */

#define STR_BUF_SIZE 4096

#define REL 0
#define ALIAS 1
#define ANY 2

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
	int num_parens; /* number of parentheses */
	int num_loops;
	str builder;
	str program;
	unsigned long program_len;
	unsigned long program_max_len;
	char global_init[STR_BUF_SIZE * 3]; /* global stmts such as vheap.base renaming and udfs inits */
	char global_cleanup[STR_BUF_SIZE * 3];
	list *stmt_list;
	sql_allocator *sa;
	int error;
} weld_state;

typedef void (*produce_func)(backend*, sql_rel*, weld_state*);

static produce_func getproduce_func(sql_rel *rel);
static void exps_to_weld(backend*, weld_state*, list*, str);
static void exp_to_weld(backend*, weld_state*, sql_exp*);

static void
dump_program(weld_state *wstate) {
	FILE *f = fopen(tmpnam(NULL), "w");
	dumpWeldProgram(wstate->program, f);
	fclose(f);
}

#ifdef __GNUC__
static void
wprintf(weld_state *wstate, char *format, ...) __attribute__ ((format(printf, 2, 3)));
#else
static
#endif

void
wprintf(weld_state *wstate, char *format, ...) {
	if (wstate->program_max_len == 0) {
		wstate->program_max_len = STR_BUF_SIZE * 2;
		wstate->program = sa_alloc(wstate->sa, wstate->program_max_len);
	} else if (wstate->program_len + STR_BUF_SIZE > wstate->program_max_len) {
		unsigned long old_size = wstate->program_max_len;
		wstate->program_max_len += STR_BUF_SIZE;
		wstate->program =
			sa_realloc(wstate->sa, wstate->program, wstate->program_max_len, old_size);
	}
	va_list args;
	va_start(args, format);
	wstate->program_len += vsprintf(wstate->program + wstate->program_len, format, args);
	va_end(args);
}

static void exps_to_weld(backend *be, weld_state *wstate, list *exps, str delim) {
	node *en;
	for (en = exps->h; en; en = en->next) {
		exp_to_weld(be, wstate, en->data);
		if (en->next != NULL && delim != NULL) {
			wprintf(wstate, "%s", delim);
		}
	}
}

static str
get_weld_cmp(int cmp) {
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

static str 
get_func_name(sql_subfunc *f) {
	return f->func->imp ? f->func->imp : f->func->base.name;
}

static str
get_weld_func(sql_subfunc *f) {
	str name = get_func_name(f);
	if (strcmp(name, "+") == 0 || strcmp(name, "sum") == 0 || strcmp(name, "count") == 0 || strcmp(name, "avg") == 0)
		return "+";
	else if (strcmp(name, "-") == 0)
		return "-";
	else if (strcmp(name, "*") == 0 || strcmp(name, "prod") == 0)
		return "*";
	else if (strcmp(name, "/") == 0)
		return "/";
	else if (strcmp(name, "&") == 0 || strcmp(name, "and") == 0)
		return "&";
	else if (strcmp(name, "|") == 0 || strcmp(name, "or") == 0)
		return "|";
	else if (strcmp(name, "<>") == 0 || strcmp(name, "!=") == 0)
		return "!=";
	else if (strcmp(name, "min") == 0)
		return "min";
	else if (strcmp(name, "max") == 0)
		return "max";
	else if (strcmp(name, "like") == 0)
		return "like";
	else if (strcmp(name, "year") == 0)
		return "year";
	else if (strcmp(name, "identity") == 0)
		return "identity";
	else if (strcmp(name, "ifthenelse") == 0)
		return "ifthenelse";
	else if (strcmp(name, "=") == 0)
		return "=";
	else if (strcmp(name, "isnil") == 0)
		return "isnil";
	return NULL;
}

static str
get_col_name(sql_allocator *sa, sql_exp *exp, int name_type) {
	char col_name[256];
	size_t i;
	if (name_type == REL) {
		sprintf(col_name, "%s_%s", exp->l ? (str)exp->l : (str)exp->r, (str)exp->r);
	} else if (name_type == ALIAS) {
		sprintf(col_name, "%s_%s", exp->rname ? exp->rname : exp->name, exp->name);
	} else if (name_type == ANY) {
		if (exp->name) {
			return get_col_name(sa, exp, ALIAS);
		} else {
			return get_col_name(sa, exp, REL);
		}
	}
	for (i = 0; i < strlen(col_name); i++) {
		if (!isalnum(col_name[i])) {
			col_name[i] = '_';
		}
	}
	return sa_strdup(sa, col_name);
}

/* Check wether the relation return a BAT as opposed to a single value */
static int
rel_returns_bat(sql_rel *rel) {
	switch (rel->op) {
	case op_project:
	case op_select:
		return rel_returns_bat(rel->l);
	case op_groupby:
		return ((list*)rel->r)->h != NULL ? 1 : 0;
	case op_basetable:
	case op_topn:
	case op_join:
		return 1;
	default:
		return -1;
	}
}

static int
exp_has_column(sql_exp *exp) {
	node *en;
	int ret = 0;
	switch (exp->type) {
	case e_atom:
	case e_psm:
		ret = 0;
		break;
	case e_aggr:
	case e_column:
		ret = 1;
		break;
	case e_convert:
		ret = exp_has_column(exp->l);
		break;
	case e_cmp:
		if (exp->flag == cmp_filter || exp->flag == cmp_or) {
			for (en = ((list*)exp->l)->h; en; en = en->next) {
				ret |= exp_has_column(en->data);
			}
		} else if (exp->l) {
			ret |= exp_has_column(exp->l);
		}
		if (exp->flag == cmp_filter || exp->flag == cmp_or || exp->flag == cmp_in || exp->flag == cmp_notin) {
			for (en = ((list*)exp->r)->h; en; en = en->next) {
				ret |= exp_has_column(en->data);
			}
		} else if (exp->r) {
			ret |= exp_has_column(exp->r);
		}
		if (exp->f) ret |= exp_has_column(exp->f);
		break;
	case e_func:
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
static void
exp_to_weld(backend *be, weld_state *wstate, sql_exp *exp) {
	if (!exp_has_column(exp)) {
		stmt* sub = exp_bin(be, exp, NULL, NULL, NULL, NULL, NULL, NULL);
		wprintf(wstate, "in%d", sub->nr);
		list_append(wstate->stmt_list, sub);
		return;
	}
	switch (exp->type) {
	case e_convert: {
		sql_subtype *from = ((list*)exp->r)->h->data;
		sql_subtype *to = ((list*)exp->r)->h->next->data;
		str conv_to = getWeldType(to->type->localtype);
		if (strcmp(conv_to, "vec[i8]") == 0) {
			/* Do nothing */
			exp_to_weld(be, wstate, exp->l);
		} else {
			wprintf(wstate, "%s(", conv_to);
			exp_to_weld(be, wstate, exp->l);
			int scale_diff = to->scale - from->scale;
			if (scale_diff != 0) {
				if (scale_diff < 0) {
					wprintf(wstate, " / 1");
					scale_diff = -scale_diff;
				} else {
					wprintf(wstate, " * 1");
				}
				while (scale_diff--) {
					wprintf(wstate, "0");
				}
				str conv_from = getWeldType(from->type->localtype);
				if (strcmp(conv_from, "f32") == 0 || strcmp(conv_from, "f64") == 0) {
					wprintf(wstate, ".0");
				}
				wprintf(wstate, "%s", getWeldTypeSuffix(from->type->localtype));
			}
			wprintf(wstate, ")");
		}
		break;
	}
	case e_cmp: {
		if (is_anti(exp)) {
			wprintf(wstate, "(");
		}
		if (exp->flag == cmp_in || exp->flag == cmp_notin) {
			if (exp->flag == cmp_notin) {
				wprintf(wstate, "(");
			}
			node *en;
			for (en = ((list*)exp->r)->h; en; en = en->next) {
				exp_to_weld(be, wstate, exp->l);
				wprintf(wstate, " == ");
				exp_to_weld(be, wstate, en->data);
				if (en->next != NULL) {
					wprintf(wstate, " || ");
				}
			}
			if (exp->flag == cmp_notin) {
				wprintf(wstate, ") == false");
			}
		} else if (get_cmp(exp) == cmp_or) {
			wprintf(wstate, "(");
			exps_to_weld(be, wstate, exp->l, " && ");
			wprintf(wstate, ") || ( ");
			exps_to_weld(be, wstate, exp->r, " && ");
			wprintf(wstate, ")");
		} else if (get_cmp(exp) == cmp_filter) {
			str udf = get_weld_func(exp->f);
			if (strcmp(udf, "like") == 0) {
				int state_ptr = wstate->next_var++;
				sprintf(wstate->global_init + strlen(wstate->global_init),
						"let v%d = cudf[like_pattern_init, i64](", state_ptr);
				/* Process the pattern and the escape */
				unsigned long old_len = wstate->program_len;
				exps_to_weld(be, wstate, exp->r, ", ");
				/* Add the pattern and the escape to the init udf */
				sprintf(wstate->global_init + strlen(wstate->global_init), "%s);",
						wstate->program + old_len);
				sprintf(wstate->global_cleanup + strlen(wstate->global_cleanup),
						"let v%d = cudf[like_pattern_cleanup, i64](v%d);", state_ptr, state_ptr);
				wstate->program_len = old_len;

				wprintf(wstate, "cudf[%s, bool](v%d,", udf, state_ptr);
				exps_to_weld(be, wstate, exp->l, ", ");
				wprintf(wstate, ")");
			} else {
				wstate->error = 1;
				return;
			}
		} else if (exp->f) {
			if (get_weld_cmp(swap_compare(range2lcompare(exp->flag))) == NULL) {
				wstate->error = 1;
				return;
			}
			exp_to_weld(be, wstate, exp->r);
			wprintf(wstate, " %s ", get_weld_cmp(swap_compare(range2lcompare(exp->flag))));
			exp_to_weld(be, wstate, exp->l);
			wprintf(wstate, " && ");
			exp_to_weld(be, wstate, exp->l);
			wprintf(wstate, " %s ", get_weld_cmp(range2rcompare(exp->flag)));
			exp_to_weld(be, wstate, exp->f);
		} else {
			if (get_weld_cmp(get_cmp(exp)) == NULL) {
				wstate->error = 1;
				return;
			}
			exp_to_weld(be, wstate, exp->l);
			wprintf(wstate, " %s ", get_weld_cmp(get_cmp(exp)));
			exp_to_weld(be, wstate, exp->r);
		}
		if (is_anti(exp)) {
			wprintf(wstate, ") == false ");
		}
		break;
	}
	case e_column: {
		wprintf(wstate, "%s", get_col_name(wstate->sa, exp, REL));
		break;
	}
	case e_func: {
		str weld_func = get_weld_func(exp->f);
		if (weld_func == NULL) {
			wstate->error = 1;
			return;
		}
		int is_infix = 0;
		if (strcmp(weld_func, "+") == 0 || strcmp(weld_func, "-") == 0 ||
			strcmp(weld_func, "*") == 0 || strcmp(weld_func, "/") == 0 ||
			strcmp(weld_func, "&") == 0 || strcmp(weld_func, "|") == 0) {
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
				wprintf(wstate, "%s(", getWeldType(right_type));
			exp_to_weld(be, wstate, left);
			if (left_type < right_type)
				wprintf(wstate, ")");
			/* Operator */
			wprintf(wstate, " %s ", weld_func);
			/* Right operand */
			if (right_type < left_type)
				wprintf(wstate, "%s(", getWeldType(left_type));
			exp_to_weld(be, wstate, right);
			if (right_type < left_type)
				wprintf(wstate, ")");
		} else {
			if (strcmp(weld_func, "year") == 0) {
				wprintf(wstate, "cudf[%s, i32](", weld_func);
				exps_to_weld(be, wstate, exp->l, ", ");
				wprintf(wstate, ")");
			} else if (strcmp(weld_func, "identity") == 0) {
				wprintf(wstate, "i_%d", wstate->num_loops);
			} else if (strcmp(weld_func, "ifthenelse") == 0) {
				/* if(x == i8nil, TYPEnil, if(x != 0, y, z)) */
				node *en = ((list*)exp->l)->h;
				sql_exp *cond_exp = en->data;
				sql_exp *then_exp = en->next->data;
				sql_exp *else_exp = en->next->next->data;
				str then_type = getWeldType(exp_subtype(then_exp)->type->localtype);
				wprintf(wstate, "if(");
				exp_to_weld(be, wstate, cond_exp);
				wprintf(wstate, " == i8nil, %snil, if(", then_type);
				exp_to_weld(be, wstate, cond_exp);
				wprintf(wstate, " > 0c, ");
				exp_to_weld(be, wstate, then_exp);
				wprintf(wstate, ", ");
				exp_to_weld(be, wstate, else_exp);
				wprintf(wstate, "))");
			} else if (strcmp(weld_func, "=") == 0) {
				wprintf(wstate, "i8(");
				exps_to_weld(be, wstate, exp->l, " == ");
				wprintf(wstate, ")");
			} else if (strcmp(weld_func, "!=") == 0) {
				wprintf(wstate, "i8(");
				exps_to_weld(be, wstate, exp->l, " != ");
				wprintf(wstate, ")");
			} else if (strcmp(weld_func, "isnil") == 0) {
				sql_exp *e = ((list*)exp->l)->h->data;
				str type = getWeldType(exp_subtype(e)->type->localtype);
				wprintf(wstate, "i8(");
				exp_to_weld(be, wstate, e);
				wprintf(wstate, " == %snil)", type);
			} else if (strcmp(weld_func, "like") == 0) {
				/* Same as `like` from cmp but not quite ... */
				int state_ptr = wstate->next_var++;
				sprintf(wstate->global_init + strlen(wstate->global_init),
						"let v%d = cudf[like_pattern_init, i64](", state_ptr);
				/* Process the pattern and the escape */
				node *en = ((list*)exp->l)->h;
				unsigned long old_len = wstate->program_len;
				exp_to_weld(be, wstate, en->next->data);
				/* Add the pattern and the escape to the init udf */
				sprintf(wstate->global_init + strlen(wstate->global_init), "%s, "");",
						wstate->program + old_len);
				sprintf(wstate->global_cleanup + strlen(wstate->global_cleanup),
						"let v%d = cudf[like_pattern_cleanup, i64](v%d);", state_ptr, state_ptr);
				wstate->program_len = old_len;

				wprintf(wstate, "i8(cudf[%s, bool](v%d,", weld_func, state_ptr);
				exp_to_weld(be, wstate, en->data);
				wprintf(wstate, "))");
			} else {
				wprintf(wstate, "%s(", weld_func);
				exps_to_weld(be, wstate, exp->l, ", ");
				wprintf(wstate, ")");
			}
		}
		break;
	}
	case e_aggr: {
		if (exp->l) {
			exps_to_weld(be, wstate, exp->l, NULL);
		} else {
			if (strcmp(((sql_subfunc*)exp->f)->func->imp, "count") == 0) {
				wprintf(wstate, "1L");
			} else {
				wstate->error = 1;
				return;
			}
		}
		break;
	}
	default: wstate->error = 1;
	}
}

static void
base_table_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	stmt *sub = subrel_bin(be, rel, NULL);
	node *en;
	sql_exp *exp;
	char struct_mbr[64];
	int count;
	wprintf(wstate, "for(zip(");
	for (en = rel->exps->h; en; en = en->next) {
		exp = en->data;
		stmt *col;
		if (exp->name) {
			col = bin_find_column(be, sub, exp->rname ? exp->rname : exp->name, exp->name);
		} else {
			col = bin_find_column(be, sub, exp->l, exp->r);
		}
		wprintf(wstate, "in%d", col->nr);
		if (en->next != NULL) {
			wprintf(wstate, ", ");
		}
		if (exp_subtype(exp)->type->localtype == TYPE_str) {
			/* Save the vheap and stroffset names */
			sprintf(wstate->global_init + strlen(wstate->global_init), "let %s_strcol = in%dstr;",
					get_col_name(wstate->sa, exp, ANY), col->nr);
			sprintf(wstate->global_init + strlen(wstate->global_init), "let %s_stroffset = in%dstroffset;",
					get_col_name(wstate->sa, exp, ANY), col->nr);
		}
	}
	/* builder and function header */
	++wstate->num_loops;
	wprintf(wstate, "), %s, |b%d, i_%d, n%d|", wstate->builder, wstate->num_loops,
				   wstate->num_loops, wstate->num_loops);
	/* extract named values from the tuple */
	for (en = rel->exps->h, count = 0; en; en = en->next, count++) {
		exp = en->data;
		str col_name = get_col_name(wstate->sa, exp, ANY);
		if (rel->exps->h->next == NULL) {
			/* just a single column so n.$0 doesn't work */
			sprintf(struct_mbr, "n%d", wstate->num_loops);
		} else {
			sprintf(struct_mbr, "n%d.$%d", wstate->num_loops, count);
		}
		if (exp_subtype(exp)->type->localtype == TYPE_str) {
			wprintf(wstate, "let %s = strslice(%s_strcol, i64(%s) + %s_stroffset);",
						   col_name, col_name, struct_mbr, col_name);
			wprintf(wstate, "let %s_stridx = %s;", col_name, struct_mbr);
		} else {
			wprintf(wstate, "let %s = %s;", col_name, struct_mbr);
		}
	}
	++wstate->num_parens;
	list_append(wstate->stmt_list, sub);
}

static void
select_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	/* === Produce === */
	produce_func input_produce = getproduce_func(rel->l);
	if (input_produce == NULL) {
		wstate->error = 1;
		return;
	}
	input_produce(be, rel->l, wstate);

	/* === Consume === */
	wprintf(wstate, "if((");
	node *en;
	for (en = rel->exps->h; en; en = en->next) {
		wprintf(wstate, "(");
		exp_to_weld(be, wstate, en->data);
		wprintf(wstate, ")");
		if (en->next != NULL) {
			wprintf(wstate, " && ");
		}
	}
	/* negate the condition so that we have "true" on the right side and we can continue to append */
	wprintf(wstate, ") == false, b%d, ", wstate->num_loops);
	++wstate->num_parens;
}

static void
project_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	char new_builder[STR_BUF_SIZE], struct_mbr[64];
	str col_name;
	int len = 0, i, count;
	node *en;
	sql_exp *exp;
	list *appender_cols = sa_list(wstate->sa);
	list *appender_col_types = sa_list(wstate->sa);
	list *orderby_exps = rel->r;

	/* === Produce === */
	int old_num_parens = wstate->num_parens;
	int old_num_loops = wstate->num_loops;
	str old_builder = wstate->builder;
	int result_var = 0;
	if (orderby_exps) {
		/* Order by statement */
		wstate->num_parens = wstate->num_loops = 0;
		result_var = wstate->next_var++;
		wstate->num_parens++;
		wprintf(wstate, "let v%d = (", result_var);

		/* New builder */
		for (en = rel->exps->h; en; en = en->next) {
			col_name = get_col_name(wstate->sa, en->data, ALIAS);
			list_append(appender_cols, col_name);
			list_append(appender_col_types, exp_subtype(en->data)->type);
		}
		for (en = orderby_exps->h; en; en = en->next) {
			col_name = get_col_name(wstate->sa, en->data, ALIAS);
			if (!list_find(appender_cols, col_name, (fcmp)strcmp)) {
				list_append(appender_cols, col_name);
				list_append(appender_col_types, exp_subtype(en->data)->type);
			}
		}

		len = sprintf(new_builder, "appender[");
		if (list_length(appender_col_types) > 1) {
			/* We have a struct */
			len += sprintf(new_builder + len, "{");
		}
		for (en = appender_col_types->h; en; en = en->next) {
			int type = ((sql_type*)en->data)->localtype;
			if (type == TYPE_str) {
				len += sprintf(new_builder + len, "?");
			} else {
				len += sprintf(new_builder + len, "%s", getWeldType(type));
			}
			if (en->next != NULL) {
				len += sprintf(new_builder + len, ", ");
			}
		}
		if (list_length(appender_col_types) > 1) {
			/* We have a struct */
			len += sprintf(new_builder + len, "}");
		}
		len += sprintf(new_builder + len, "]");
		wstate->builder = new_builder;
	}

	produce_func input_produce = getproduce_func(rel->l);
	if (input_produce == NULL) {
		wstate->error = 1;
		goto cleanup;
	}
	input_produce(be, rel->l, wstate);

	/* === Consume === */
	for (en = rel->exps->h; en; en = en->next) {
		exp = en->data;
		col_name = get_col_name(wstate->sa, exp, ALIAS);
		if (exp_subtype(exp)->type->localtype == TYPE_str) {
			if (exp->type != e_column) {
				/* We can only handle string column renaming. If this is something else, like producing
				 * a new string, we don't support it yet. */
				wstate->error = 1;
			}
			str old_col_name = get_col_name(wstate->sa, exp, REL);
			wprintf(wstate, "let %s = %s;", col_name, old_col_name);
			wprintf(wstate, "let %s_stridx = %s_stridx;", col_name, old_col_name);
			/* Save the vheap and stroffset names */
			sprintf(wstate->global_init + strlen(wstate->global_init), "let %s_strcol = %s_strcol;",
					col_name, old_col_name);
			sprintf(wstate->global_init + strlen(wstate->global_init), "let %s_stroffset = %s_stroffset;",
					col_name, old_col_name);
		} else {
			wprintf(wstate, "let %s = ", col_name);
			exp_to_weld(be, wstate, exp);
			wprintf(wstate, ";");
		}
	}
	if (rel->r) {
		/* Sorting phase - begin by materializing the columns */
		wprintf(wstate, "merge(b%d, ", wstate->num_loops);
		if (list_length(appender_cols) > 1) {
			wprintf(wstate, "{");
		}
		for (en = appender_cols->h, count = 0; en; en = en->next, count++) {
			col_name = en->data;
			int type = ((sql_type*)list_fetch(appender_col_types, count))->localtype;
			wprintf(wstate, "%s", col_name);
			if (type == TYPE_str) {
				wprintf(wstate, "_stridx");
			}
			if (en->next != NULL) {
				wprintf(wstate, ", ");
			}
		}
		if (list_length(appender_cols) > 1) {
			wprintf(wstate, "}");
		}
		wprintf(wstate, ")");
		for (i = 0; i < wstate->num_parens; i++) {
			wprintf(wstate, ")");
		}
		wprintf(wstate, ";");
		/* Sort the array */
		wstate->next_var++;
		wprintf(wstate, "let v%d = sort(result(v%d), |n| ", wstate->next_var, result_var);
		for (en = orderby_exps->h; en; en = en->next) {
			exp = en->data;
			col_name = get_col_name(wstate->sa, exp, ALIAS);
			node *col_list_node = list_find(appender_cols, col_name, (fcmp)strcmp);
			int idx = list_position(appender_cols, col_list_node->data);
			if (exp_subtype(exp)->type->localtype == TYPE_str) {
				wprintf(wstate, "let %s = strslice(%s_strcol, i64(n.$%d) + %s_stroffset);",
					col_name, col_name, idx, col_name);
			} else {
				wprintf(wstate, "let %s = n.$%d;", col_name, idx);
			}
		}
		if (list_length(orderby_exps) > 1) {
			wprintf(wstate, "{");
		}
		for (en = orderby_exps->h; en; en = en->next) {
			exp = en->data;
			col_name = get_col_name(wstate->sa, exp, ALIAS);
			wprintf(wstate, "%s", col_name);
			if (en->next != NULL) {
				wprintf(wstate, ", ");
			}
		}
		if (list_length(orderby_exps) > 1) {
			wprintf(wstate, "}");
		}
		wprintf(wstate, ");");
		/* Resume the pipeline */
		wstate->num_parens = old_num_parens;
		wstate->num_loops = old_num_loops;
		wstate->builder = old_builder;
		wstate->num_loops++;
		wstate->num_parens++;
		wprintf(wstate, "for(v%d, %s, |b%d, i_%d, n%d|", wstate->next_var,
					   wstate->builder, wstate->num_loops, wstate->num_loops, wstate->num_loops);
		for (en = rel->exps->h, count = 0; en; en = en->next, count++) {
			if (list_length(appender_cols) > 1) {
				sprintf(struct_mbr, "n%d.$%d", wstate->num_loops, count);
			} else {
				sprintf(struct_mbr, "n%d", wstate->num_loops);
			}
			exp = en->data;
			col_name = get_col_name(wstate->sa, exp, ALIAS);
			if (exp_subtype(exp)->type->localtype == TYPE_str) {
				wprintf(wstate, "let %s = strslice(%s_strcol, i64(%s) + %s_stroffset);",
						col_name, col_name, struct_mbr, col_name);
				wprintf(wstate, "let %s_stridx = %s;", col_name, struct_mbr);
			} else {
				wprintf(wstate, "let %s = %s;", col_name, struct_mbr);
			}
		}
	}
cleanup:
	list_destroy(appender_cols);
	list_destroy(appender_col_types);
}

static void
groupby_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	char new_builder[STR_BUF_SIZE];
	str col_name;
	int len = 0, i, col_count, aggr_count, result_var, have_avg = 0;
	node *en;
	sql_exp *exp;
	str aggr_func = NULL;
	produce_func input_produce;
	list *group_by_exps = sa_list(wstate->sa);
	list *aggr_exps = sa_list(wstate->sa);
	list *project_exps = sa_list(wstate->sa);

	/* === Produce === */
	int old_num_parens = wstate->num_parens;
	int old_num_loops = wstate->num_loops;
	str old_builder = wstate->builder;

	for (en = rel->exps->h; en; en = en->next) {
		exp = en->data;
		if (exp->type == e_aggr) {
			list_append(aggr_exps, exp);
			if (aggr_func == NULL) {
				aggr_func = get_weld_func(exp->f);
			} else if (aggr_func != get_weld_func(exp->f)) {
				/* Currently Weld only supports a single operation for mergers */
				wstate->error = 1;
				goto cleanup;
			}
			if (strcmp(get_func_name(exp->f), "avg") == 0) {
				have_avg = 1;
			}
		} else {
			if (exp->type == e_column && exp->r && exp->name && strcmp((str)exp->r, exp->name) != 0) {
				list_append(project_exps, exp);
			} else {
				list_append(group_by_exps, exp);
			}
		}
	}
	/* Create a new builder */
	wstate->num_parens = wstate->num_loops = 0;
	result_var = wstate->next_var++;
	wprintf(wstate, "let v%d = (", result_var);
	wstate->num_parens++;
	len = 0;
	if (list_length(group_by_exps) > 0) {
		len += sprintf(new_builder + len, "dictmerger[");
		if (list_length(group_by_exps) > 1) {
			len += sprintf(new_builder + len, "{"); /* key is a struct */
		}
		for (en = group_by_exps->h; en; en = en->next) {
			exp = en->data;
			int type = exp_subtype(exp)->type->localtype;
			if (type == TYPE_str) {
				len += sprintf(new_builder + len, "?");
			} else {
				len += sprintf(new_builder + len, "%s", getWeldType(type));
			}
			if (en->next != NULL) {
				len += sprintf(new_builder + len, ", ");
			}
		}
		if (list_length(group_by_exps) > 1) {
			len += sprintf(new_builder + len, "}"); /* key is a struct */
		}
		len += sprintf(new_builder + len, ", ");
	} else {
		len += sprintf(new_builder + len, "merger[");
	}
	if (list_length(aggr_exps) > 1 || have_avg) {
		len += sprintf(new_builder + len, "{"); /* value is a struct */
	}
	for (en = aggr_exps->h; en; en = en->next) {
		exp = en->data;
		int type = exp_subtype(exp)->type->localtype;
		len += sprintf(new_builder + len, "%s", getWeldType(type));
		if (strcmp(get_func_name(exp->f), "avg") == 0) {
			len += sprintf(new_builder + len, ", i64");
		}
		if (en->next != NULL) {
			len += sprintf(new_builder + len, ", ");
		}
	}
	if (list_length(aggr_exps) > 1 || have_avg) {
		len += sprintf(new_builder + len, "}"); /* value is a struct */
	}
	len += sprintf(new_builder + len, ", %s]", aggr_func);
	wstate->builder = new_builder;
	input_produce = getproduce_func(rel->l);
	if (input_produce == NULL) {
		wstate->error = 1;
		goto cleanup;
	}
	input_produce(be, rel->l, wstate);

	/* === Consume === */
	wprintf(wstate, "merge(b%d, ", wstate->num_loops);
	if (list_length(group_by_exps) > 0) { /* {key, value} */
		/* Build the key */
		wprintf(wstate, "{");
		if (list_length(group_by_exps) > 1) {
			wprintf(wstate, "{"); /* key is a struct */
		}
		for (en = group_by_exps->h; en; en = en->next) {
			exp = en->data;
			exp_to_weld(be, wstate, exp);
			int type = exp_subtype(exp)->type->localtype;
			if (type == TYPE_str) {
				wprintf(wstate, "_stridx");
			}
			if (en->next != NULL) {
				wprintf(wstate, ", ");
			}
		}
		if (list_length(group_by_exps) > 1) {
			wprintf(wstate, "}"); /* key is a struct */
		}
		wprintf(wstate, ", ");
	}
	if (list_length(aggr_exps) > 1 || have_avg) {
		wprintf(wstate, "{"); /* value is a struct */
	}
	for (en = aggr_exps->h; en; en = en->next) {
		exp = en->data;
		/* We might have different types when doing the aggregation so we need to cast */
		str weld_type = getWeldType(exp_subtype(exp)->type->localtype);
		wprintf(wstate, "%s(", weld_type);
		exp_to_weld(be, wstate, exp);
		wprintf(wstate, ")");
		if (strcmp(get_func_name(exp->f), "avg") == 0) {
			wprintf(wstate, ", 1L");
		}
		if (en->next != NULL) {
			wprintf(wstate, ", ");
		}
	}
	if (list_length(aggr_exps) > 1 || have_avg) {
		wprintf(wstate, "}"); /* value is a struct */
	}
	if (list_length(group_by_exps) > 0) {
		wprintf(wstate, "}"); /* {key, value} */
	}
	wprintf(wstate, ")");
	for (i = 0; i < wstate->num_parens; i++) {
		wprintf(wstate, ")");
	}
	wprintf(wstate, ";");

	/* Resume the pipeline */
	wstate->num_parens = old_num_parens;
	wstate->num_loops = old_num_loops;
	wstate->builder = old_builder;
	char struct_mbr[128];
	col_count = aggr_count = 0;
	if (group_by_exps->h) {
		wstate->num_loops++;
		wstate->num_parens++;
		wprintf(wstate, "for(tovec(result(v%d)), %s, |b%d, i_%d, n%d|", result_var,
					   wstate->builder, wstate->num_loops, wstate->num_loops, wstate->num_loops);
	} else {
		wstate->next_var++;
		wprintf(wstate, "let v%d = result(v%d);", wstate->next_var, result_var);
	}
	/* Column renaming */
	for (en = rel->exps->h; en; en = en->next) {
		exp = en->data;
		col_name = get_col_name(wstate->sa, exp, ALIAS);
		if (list_find(project_exps, exp, NULL)) {
			/* Just use the old name */
			sprintf(struct_mbr, "%s", get_col_name(wstate->sa, exp, REL));
		} else if (list_find(group_by_exps, exp, NULL)) {
			/* Set the name of a group by column */
			len = sprintf(struct_mbr, "n%d.$0", wstate->num_loops);
			if (list_length(group_by_exps) > 1) {
				len += sprintf(struct_mbr + len, ".$%d", col_count++);
			}
		} else if (list_find(aggr_exps, exp, NULL) && list_length(group_by_exps) > 0) {
			/* Set the name of an aggregate if we used a dictmerger */
			len = sprintf(struct_mbr, "n%d.$1", wstate->num_loops);
			if (list_length(aggr_exps) > 1 || have_avg) {
				len += sprintf(struct_mbr + len, ".$%d", aggr_count++);
			}
			if (strcmp(get_func_name(exp->f), "avg") == 0) {
				/* Divide by the count */
				len += sprintf(struct_mbr + len, " / f64(n%d.$1.$%d)", wstate->num_loops, aggr_count++);
			}
		} else {
			/* Set the name of an aggregate if we used a merger */
			len = sprintf(struct_mbr, "v%d", wstate->next_var);
			if (list_length(aggr_exps) > 1 || have_avg) {
				len += sprintf(struct_mbr + len, ".$%d", col_count++);
			}
			if (strcmp(get_func_name(exp->f), "avg") == 0) {
				/* Divide by the count */
				len += sprintf(struct_mbr + len, " / f64(v%d.$%d)", wstate->next_var, aggr_count++);
			}
		}
		if (exp_subtype(exp)->type->localtype == TYPE_str) {
			wprintf(wstate, "let %s = strslice(%s_strcol, i64(%s) + %s_stroffset);", 
						   col_name, col_name, struct_mbr, col_name);
			wprintf(wstate, "let %s_stridx = %s;", col_name, struct_mbr);
			/* Global string col renaming */
			str old_col_name = get_col_name(wstate->sa, exp, REL);
			sprintf(wstate->global_init + strlen(wstate->global_init), "let %s_strcol = %s_strcol;",
					col_name, old_col_name);
			sprintf(wstate->global_init + strlen(wstate->global_init), "let %s_stroffset = %s_stroffset;",
					col_name, old_col_name);
		} else {
			wprintf(wstate, "let %s = %s;", col_name, struct_mbr);
		}
	}
cleanup:
	list_destroy(group_by_exps);
	list_destroy(aggr_exps);
	list_destroy(project_exps);
}

static void
join_produce(backend *be, sql_rel *rel, weld_state *wstate)
{
	char new_builder[STR_BUF_SIZE], struct_mbr[64];
	str col_name;
	int len = 0, i, count;
	node *en;
	sql_exp *exp;
	sql_rel *right = rel->r;
	list *right_cols = sa_list(wstate->sa);
	list *right_cmp_cols = sa_list(wstate->sa);
	list *left_cmp_cols = sa_list(wstate->sa);
	produce_func left_produce, right_produce;

	/* === Produce === */
	int old_num_parens = wstate->num_parens;
	int old_num_loops = wstate->num_loops;
	str old_builder = wstate->builder;

	/* Create a new builder */
	wstate->num_parens = wstate->num_loops = 0;
	int result_var = wstate->next_var++;
	wprintf(wstate, "let v%d = (", result_var);
	wstate->num_parens++;

	/* Find the operator that produces the columns */
	while (right != NULL && right->op != op_project && right->op != op_basetable) {
		right = right->l;
	}
	if (right == NULL) {
		wstate->error = 1;
		goto cleanup;
	}
	for (en = right->exps->h; en; en = en->next) {
		list_append(right_cols, get_col_name(wstate->sa, en->data, ANY));
	}

	len = 0;
	len += sprintf(new_builder + len, "groupmerger[");
	if (list_length(rel->exps) > 1) {
		len += sprintf(new_builder + len, "{"); /* key is a struct */
	}
	for (en = rel->exps->h; en; en = en->next) {
		/* left cmp */
		exp = ((sql_exp*)en->data)->l;
		col_name = get_col_name(wstate->sa, exp, REL);
		if (list_find(right_cols, col_name, (fcmp)strcmp)) {
			list_append(right_cmp_cols, col_name);
		} else {
			list_append(left_cmp_cols, col_name);
		}
		/* right cmp */
		exp = ((sql_exp*)en->data)->r;
		col_name = get_col_name(wstate->sa, exp, REL);
		if (list_find(right_cols, col_name, (fcmp)strcmp)) {
			list_append(right_cmp_cols, col_name);
		} else {
			list_append(left_cmp_cols, col_name);
		}

		/* both have the same type */
		int type = exp_subtype(exp)->type->localtype;
		len += sprintf(new_builder + len, "%s", getWeldType(type));
		if (en->next != NULL) {
			len += sprintf(new_builder + len, ", ");
		}
	}
	if (list_length(rel->exps) > 1) {
		len += sprintf(new_builder + len, "}"); /* key is a struct */
	}
	len += sprintf(new_builder + len, ", ");
	if (list_length(right->exps) > 1) {
		len += sprintf(new_builder + len, "{"); /* value is a struct */
	}
	for (en = right->exps->h; en; en = en->next) {
		exp = en->data;
		int type = exp_subtype(exp)->type->localtype;
		if (type == TYPE_str) {
			len += sprintf(new_builder + len, "?");
		} else {
			len += sprintf(new_builder + len, "%s", getWeldType(type));
		}
		if (en->next != NULL) {
			len += sprintf(new_builder + len, ", ");
		}
	}
	if (list_length(right->exps) > 1) {
		len += sprintf(new_builder + len, "}"); /* value is a struct */
	}
	len += sprintf(new_builder + len, "]");

	wstate->builder = new_builder;
	right_produce = getproduce_func(rel->r);
	left_produce = getproduce_func(rel->l);
	if (right_produce == NULL || left_produce == NULL) {
		wstate->error = 1;
		goto cleanup;
	}
	right_produce(be, rel->r, wstate);

	/* === Consume === */
	wprintf(wstate, "merge(b%d, {", wstate->num_loops); /* {key, value} */
	/* Build the key */
	if (list_length(right_cmp_cols) > 1) {
		wprintf(wstate, "{"); /* key is a struct */
	}
	for (en = right_cmp_cols->h; en; en = en->next) {
		wprintf(wstate, "%s", (str)en->data);
		if (en->next != NULL) {
			wprintf(wstate, ", ");
		}
	}
	if (list_length(right_cmp_cols) > 1) {
		wprintf(wstate, "}"); /* key is a struct */
	}
	wprintf(wstate, ", ");
	if (list_length(right->exps) > 1) {
		wprintf(wstate, "{"); /* value is a struct */
	}
	/* Build the value */
	for (en = right->exps->h, count = 0; en; en = en->next, count++) {
		exp = en->data;
		wprintf(wstate, "%s", (str)list_fetch(right_cols, count));
		if (exp_subtype(exp)->type->localtype == TYPE_str) {
			wprintf(wstate, "_stridx");
		}
		if (en->next != NULL) {
			wprintf(wstate, ", ");
		}
	}
	if (list_length(right->exps) > 1) {
		wprintf(wstate, "}"); /* value is a struct */
	}

	wprintf(wstate, "})");
	for (i = 0; i < wstate->num_parens; i++) {
		wprintf(wstate, ")");
	}
	wprintf(wstate, ";");
	/* Materialize the hashtable */
	wprintf(wstate, "let v%d = result(v%d);", result_var, result_var);

	/* Resume the pipeline */
	wstate->num_parens = old_num_parens;
	wstate->num_loops = old_num_loops;
	wstate->builder = old_builder;
	left_produce(be, rel->l, wstate);

	/* === 2nd Consume === */
	wstate->num_loops++;
	wstate->num_parens++;
	wprintf(wstate, "for(");
	if (rel->op == op_semi) {
		wprintf(wstate, "slice(");
	}
	wprintf(wstate, "lookup(v%d, ", result_var);
	if (list_length(left_cmp_cols) > 1) {
		wprintf(wstate, "{"); /* key is a struct */
	}
	for (en = left_cmp_cols->h; en; en = en->next) {
		/* Hashtable key */
		wprintf(wstate, "%s", (str)en->data);
		if (en->next != NULL) {
			wprintf(wstate, ", ");
		}
	}
	if (list_length(left_cmp_cols) > 1) {
		wprintf(wstate, "}"); /* key is a struct */
	}
	wprintf(wstate, ")");
	if (rel->op == op_semi) {
		wprintf(wstate, ", 0L, 1L)"); /* Just the first element of the vector */
	}
	wprintf(wstate, ", b%d, |b%d, i_%d, n%d|", wstate->num_loops - 1, wstate->num_loops,
			wstate->num_loops, wstate->num_loops);
	for (en = right->exps->h, count = 0; en; en = en->next, count++) {
		len = sprintf(struct_mbr, "n%d", wstate->num_loops);
		if (list_length(right->exps) > 1) {
			len += sprintf(struct_mbr + len, ".$%d", count);
		}
		exp = en->data;
		col_name = list_fetch(right_cols, count);
		if (exp_subtype(exp)->type->localtype == TYPE_str) {
			wprintf(wstate, "let %s = strslice(%s_strcol, i64(%s) + %s_stroffset);", 
						   col_name, col_name, struct_mbr, col_name);
			wprintf(wstate, "let %s_stridx = %s;", col_name, struct_mbr);
		} else {
			wprintf(wstate, "let %s = %s;", col_name, struct_mbr);
		}
	}
cleanup:
	list_destroy(right_cols);
	list_destroy(right_cmp_cols);
	list_destroy(left_cmp_cols);
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
	wstate->sa = sa_create();
	wstate->stmt_list = sa_list(wstate->sa);

	stmt *final_stmt = NULL, *program_stmt, *weld_program_stmt;
	InstrPtr weld_instr;
	sql_rel *root = rel->op == op_topn ? rel->l : rel;
	node *en;
	char builder[STR_BUF_SIZE];
	str col_name;
	int i, count, *arg_names, len = 0, idx;
	int result_is_bat = rel_returns_bat(root);
	/* Save the builders in a variable */
	int result_var = wstate->next_var++;
	wprintf(wstate, "let v%d = (", result_var);
	wstate->num_parens++;
	/* Prepare the builders */
	if (result_is_bat) {
		len += sprintf(builder + len, "{");
		for (en = root->exps->h; en; en = en->next) {
			int type = exp_subtype(en->data)->type->localtype;
			if (type == TYPE_str) {
				/* We'll append just the offset in vheap, we don't know the type yet */
				len += sprintf(builder + len, "appender[?]");
			} else {
				len += sprintf(builder + len, "appender[%s]", getWeldType(type));
			}
			if (en->next != NULL) {
				len += sprintf(builder + len, ", ");
			}
		}
		len += sprintf(builder + len, "}");
		wstate->builder = builder;
	}
	produce_func input_produce = getproduce_func(root);
	if (input_produce == NULL) return NULL;
	input_produce(be, root, wstate);
	if (wstate->error) {
		/* Can't convert this query */
		goto cleanup;
	}

	/* === Consume === */
	/* Append the results to the builders */
	wprintf(wstate, "{");
	for (en = root->exps->h, count = 0; en; en = en->next, count++) {
		sql_exp *exp = en->data;
		col_name = get_col_name(wstate->sa, exp, ALIAS);
		if (result_is_bat) {
			int type = exp_subtype(en->data)->type->localtype;
			if (type == TYPE_str) {
				wprintf(wstate, "merge(b%d.$%d, %s_stridx)", wstate->num_loops, count, col_name);
			} else {
				wprintf(wstate, "merge(b%d.$%d, %s)", wstate->num_loops, count, col_name);
			}
		} else {
			wprintf(wstate, "%s", col_name);
		}
		if (en->next != NULL) {
			wprintf(wstate, ", ");
		}
	}
	wprintf(wstate, "}");
	/* Close the parentheses */
	for (i = 0; i < wstate->num_parens; i++) {
		wprintf(wstate, ")");
	}
	wprintf(wstate, ";");
	wprintf(wstate, "%s", wstate->global_cleanup); /* TODO - this gets removed by Weld */
	/* Final result statement */
	if (result_is_bat) {
		wprintf(wstate, "{");
		char limit[64], offset[64];
		if (rel->op == op_topn) {
			sql_exp *limit_exp = rel->exps->h->data;
			stmt *limit_stmt = exp_bin(be, limit_exp, NULL, NULL, NULL, NULL, NULL, NULL);
			list_append(wstate->stmt_list, limit_stmt);
			sprintf(limit, "i64(in%d)", limit_stmt->nr);
			if (rel->exps->h->next != NULL) {
				sql_exp *offset_exp = rel->exps->h->next->data;
				stmt *offset_stmt = exp_bin(be, offset_exp, NULL, NULL, NULL, NULL, NULL, NULL);
				list_append(wstate->stmt_list, offset_stmt);
				sprintf(offset, "i64(in%d)", offset_stmt->nr);
			} else {
				sprintf(offset, "0L");
			}
		}
		for (en = root->exps->h, count = 0; en; en = en->next, count++) {
			if (rel->op == op_topn)
				wprintf(wstate, "slice(");
			wprintf(wstate, "result(v%d.$%d)", result_var, count);
			if (rel->op == op_topn)
				wprintf(wstate, ", %s, %s)", offset, limit);
			sql_exp *exp = en->data;
			int type = exp_subtype(en->data)->type->localtype;
			if (type == TYPE_str) {
				wprintf(wstate, ", %s_strcol", get_col_name(wstate->sa, exp, ALIAS));
			}
			if (en->next != NULL) {
				wprintf(wstate, ", ");
			}
		}
		wprintf(wstate, "}");
	} else {
		/* Just the top variable */
		wprintf(wstate, "v%d", result_var);
	}
	/* Combine the string cols renamings with the weld program */
	wstate->program = sa_strconcat(wstate->sa, wstate->global_init, wstate->program);

	/* Build the Weld MAL instruction */
	weld_instr = newInstruction(be->mb, weldRef, weldRunRef);
	for (en = root->exps->h; en; en = en->next) {
		sql_subtype *subtype = exp_subtype(en->data);
		int type = subtype->type->localtype;
		if (result_is_bat) {
			type = newBatType(type);
		}
		weld_instr = pushReturn(be->mb, weld_instr, newTmpVariable(be->mb, type));
	}
	/* Push the arguments, first arg: the weld program, second arg: array of arg names */
	idx = 0;
	program_stmt = stmt_atom_string(be, wstate->program);
	arg_names = (int*)sa_alloc(be->mvc->sa, 1000 * sizeof(int)); /* Should be enough */
	weld_instr = pushArgument(be->mb, weld_instr, program_stmt->nr);
	weld_instr = pushPtr(be->mb, weld_instr, arg_names);
	push_args(be->mb, &weld_instr, wstate->stmt_list, arg_names, &idx);
	pushInstruction(be->mb, weld_instr);

	list_append(wstate->stmt_list, program_stmt);
	weld_program_stmt = stmt_list(be, wstate->stmt_list);
	weld_program_stmt->q = weld_instr;
	final_stmt = create_result_instr(be, weld_program_stmt, root->exps);

cleanup:
	dump_program(wstate);
	list_destroy(wstate->stmt_list);
	sa_destroy(wstate->sa);
	free(wstate);

	return final_stmt;
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
		case op_groupby:
			return &groupby_produce;
		case op_join:
		case op_semi:
			return &join_produce;
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
