/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _SQL_STATEMENT_H_
#define _SQL_STATEMENT_H_

#include "sql_mem.h"
#include "sql_types.h"
#include "sql_atom.h"
#include "sql_string.h"
#include "sql_mvc.h"
#include "mal_backend.h"

typedef union stmtdata {
	struct atom *aval;
	struct list *lval;
	struct stmt *stval;

	struct sql_column *cval;
	struct sql_idx *idxval;
	struct sql_table *tval;

	sql_subtype typeval;
	struct sql_subaggr *aggrval;
	struct sql_subfunc *funcval;
	sql_rel *rel;
} stmtdata;

typedef enum stmt_type {
	st_none,
	st_var,			/* use and/or declare variable */

	st_table,		/* some functions return a table */
	st_temp,		/* temporal bat */
	st_single,		/* single value bat */
	st_rs_column,
	st_tid,
	st_bat,
	st_idxbat,
	st_const,
	st_gen_group,
	st_mirror,
	st_result,		/* get nth result of a statement */

	st_limit,
	st_limit2,
	st_sample,
	st_order,
	st_reorder,

	st_output,
	st_affected_rows,

	st_atom,
	st_uselect,
	st_uselect2,
	st_tunion,
	st_tdiff,
	st_tinter,

	st_join,
	st_join2,
	st_joinN,

	st_export,
	st_append,
	st_table_clear,
	st_exception,
	st_trans,
	st_catalog,

	st_append_col,
	st_append_idx,
	st_update_col,
	st_update_idx,
	st_delete,

	st_group,
	st_convert,
	st_Nop,
	st_func,
	st_aggr,

	st_alias,

	/* used internally only */
	st_list,

	/* flow control statements */
	st_cond,
	st_control_end,
	st_return,
	st_assign
} st_type;

/* flag to indicate anti join/select */
#define SWAPPED 16
#define ANTI ANTISEL
#define GRP_DONE 32

typedef struct stmt {
	st_type type;
	struct stmt *op1;
	struct stmt *op2;
	struct stmt *op3;
	stmtdata op4;		/* only op4 will hold other types */

	char nrcols;
	char key;		/* key (aka all values are unique) */ // TODO make this thing a bool
	char aggr;		/* aggregated */
	char partition;		/* selected as mitosis candidate */

	int flag;

	int nr;			/* variable assignment */

	const char *tname;
	const char *cname;
	InstrPtr q;
} stmt;

sql_extern void create_merge_partitions_accumulator(backend *be);
sql_extern int add_to_merge_partitions_accumulator(backend *be, int nr);

sql_extern int stmt_key(stmt *s);

sql_extern stmt *stmt_none(backend *be);

//#define VAR_DECLARE 1
#define VAR_GLOBAL(f) ((f>>1)==1)
sql_extern stmt *stmt_var(backend *be, const char *varname, sql_subtype *t, int declare, int level);
sql_extern stmt *stmt_vars(backend *be, const char *varname, sql_table *t, int declare, int level);
sql_extern stmt *stmt_varnr(backend *be, int nr, sql_subtype *t);

sql_extern stmt *stmt_table(backend *be, stmt *cols, int temp);
sql_extern stmt *stmt_rs_column(backend *be, stmt *result_set, int i, sql_subtype *tpe);

sql_extern stmt *stmt_bat(backend *be, sql_column *c, int access, int partition);
sql_extern stmt *stmt_idxbat(backend *be, sql_idx *i, int access, int partition);
sql_extern stmt *stmt_tid(backend *be, sql_table *t, int partition);

sql_extern stmt *stmt_append_col(backend *be, sql_column *c, stmt *b, int locked);
sql_extern stmt *stmt_append_idx(backend *be, sql_idx *i, stmt *b);
sql_extern stmt *stmt_update_col(backend *be, sql_column *c, stmt *tids, stmt *upd);
sql_extern stmt *stmt_update_idx(backend *be, sql_idx *i, stmt *tids, stmt *upd);
sql_extern stmt *stmt_delete(backend *be, sql_table *t, stmt *b);

sql_extern stmt *stmt_append(backend *be, stmt *c, stmt *values);
sql_extern stmt *stmt_table_clear(backend *be, sql_table *t);

sql_extern stmt *stmt_export(backend *be, stmt *t, const char *sep, const char *rsep, const char *ssep, const char *null_string, int onclient, stmt *file);
sql_extern stmt *stmt_trans(backend *b, int type, stmt *chain, stmt *name);
sql_extern stmt *stmt_catalog(backend *be, int type, stmt *args);

sql_extern stmt *stmt_temp(backend *be, sql_subtype *t);
sql_extern stmt *stmt_atom(backend *be, atom *a);
sql_extern stmt *stmt_atom_string(backend *be, const char *s);
sql_extern stmt *stmt_atom_string_nil(backend *be);
sql_extern stmt *stmt_atom_int(backend *be, int i);
sql_extern stmt *stmt_atom_lng(backend *be, lng i);
sql_extern stmt *stmt_atom_lng_nil(backend *be);
sql_extern stmt *stmt_bool(backend *be, int b);

sql_extern stmt *stmt_uselect(backend *be, stmt *op1, stmt *op2, comp_type cmptype, stmt *sub, int anti);
/* cmp
       0 ==   l <  x <  h
       1 ==   l <  x <= h
       2 ==   l <= x <  h
       3 ==   l <= x <= h
       */
sql_extern stmt *stmt_uselect2(backend *be, stmt *op1, stmt *op2, stmt *op3, int cmp, stmt *sub, int anti);
sql_extern stmt *stmt_genselect(backend *be, stmt *lops, stmt *rops, sql_subfunc *f, stmt *sub, int anti);

sql_extern stmt *stmt_tunion(backend *be, stmt *op1, stmt *op2);
sql_extern stmt *stmt_tdiff(backend *be, stmt *op1, stmt *op2);
sql_extern stmt *stmt_tdiff2(backend *be, stmt *op1, stmt *op2);
sql_extern stmt *stmt_tinter(backend *be, stmt *op1, stmt *op2);

sql_extern stmt *stmt_join(backend *be, stmt *op1, stmt *op2, int anti, comp_type cmptype);
sql_extern stmt *stmt_join2(backend *be, stmt *l, stmt *ra, stmt *rb, int cmp, int anti, int swapped);
/* generic join operator, with a left and right statement list */
sql_extern stmt *stmt_genjoin(backend *be, stmt *l, stmt *r, sql_subfunc *op, int anti, int swapped);

sql_extern stmt *stmt_project(backend *be, stmt *op1, stmt *op2);
sql_extern stmt *stmt_project_delta(backend *be, stmt *col, stmt *upd, stmt *ins);
sql_extern stmt *stmt_left_project(backend *be, stmt *op1, stmt *op2, stmt *op3);

sql_extern stmt *stmt_list(backend *be, list *l);
sql_extern void stmt_set_nrcols(stmt *s);

sql_extern stmt *stmt_group(backend *be, stmt *op1, stmt *grp, stmt *ext, stmt *cnt, int done);

/* raise exception incase the condition (cond) holds, continue with stmt res */
sql_extern stmt *stmt_exception(backend *be, stmt *cond, const char *errstr, int errcode);

sql_extern stmt *stmt_const(backend *be, stmt *s, stmt *val);

sql_extern stmt *stmt_gen_group(backend *be, stmt *gids, stmt *cnts);	/* given a gid,cnt blowup to full groups */
sql_extern stmt *stmt_mirror(backend *be, stmt *s);
sql_extern stmt *stmt_result(backend *be, stmt *s, int nr);

/* 
 * distinct: compute topn on unique groups
 * dir:      direction of the ordering, ie 1 Ascending, 0 decending
 * last:     intermediate step or last step 
 * order:    is order important or not (firstn vs slice)
 */ 
sql_extern stmt *stmt_limit(backend *sa, stmt *c, stmt *piv, stmt *gid, stmt *offset, stmt *limit, int distinct, int dir, int nullslast, int last, int order);
sql_extern stmt *stmt_sample(backend *be, stmt *s, stmt *sample, stmt *seed);
sql_extern stmt *stmt_order(backend *be, stmt *s, int direction, int nullslast);
sql_extern stmt *stmt_reorder(backend *be, stmt *s, int direction, int nullslast, stmt *orderby_ids, stmt *orderby_grp);

sql_extern stmt *stmt_convert(backend *sa, stmt *v, sql_subtype *from, sql_subtype *to);
sql_extern stmt *stmt_unop(backend *be, stmt *op1, sql_subfunc *op);
sql_extern stmt *stmt_binop(backend *be, stmt *op1, stmt *op2, sql_subfunc *op);
sql_extern stmt *stmt_Nop(backend *be, stmt *ops, sql_subfunc *op);
sql_extern stmt *stmt_func(backend *be, stmt *ops, const char *name, sql_rel *imp, int f_union);
sql_extern stmt *stmt_aggr(backend *be, stmt *op1, stmt *grp, stmt *ext, sql_subaggr *op, int reduce, int no_nil, int nil_if_empty);

sql_extern stmt *stmt_alias(backend *be, stmt *op1, const char *tname, const char *name);

sql_extern stmt *stmt_output(backend *be, stmt *l);
sql_extern stmt *stmt_affected_rows(backend *be, stmt *l);

/* flow control statements */
sql_extern stmt *stmt_cond(backend *be, stmt *cond, stmt *outer, int loop, int anti);
sql_extern stmt *stmt_control_end(backend *be, stmt *cond);
sql_extern stmt *stmt_return(backend *be, stmt *val, int nr_of_declared_tables);
sql_extern stmt *stmt_assign(backend *be, const char *varname, stmt *val, int level);

sql_extern sql_subtype *tail_type(stmt *st);
sql_extern int stmt_has_null(stmt *s);

sql_extern const char *column_name(sql_allocator *sa, stmt *st);
sql_extern const char *table_name(sql_allocator *sa, stmt *st);
sql_extern const char *schema_name(sql_allocator *sa, stmt *st);

sql_extern stmt *const_column(backend *ba, stmt *val);
sql_extern stmt *stmt_fetch(backend *ba, stmt *val);

#endif /* _SQL_STATEMENT_H_ */
