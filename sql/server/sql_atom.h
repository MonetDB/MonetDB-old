/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _SQL_ATOM_H_
#define _SQL_ATOM_H_

#include "sql_mem.h"
#include "sql_types.h"

typedef struct atom {
	int isnull;
	sql_subtype tpe;
	ValRecord data;
	dbl d;
	int varid;/* used during code generation only */
} atom;

#define atom_null(a) (((atom*)a)->isnull)

sql_extern void atom_init( atom *a );
sql_extern atom *atom_bool( sql_allocator *sa, sql_subtype *tpe, bit t);
#ifdef HAVE_HGE
sql_extern atom *atom_int( sql_allocator *sa, sql_subtype *tpe, hge val);
#else
sql_extern atom *atom_int( sql_allocator *sa, sql_subtype *tpe, lng val);
#endif
sql_extern atom *atom_float( sql_allocator *sa, sql_subtype *tpe, double val);
sql_extern atom *atom_string( sql_allocator *sa, sql_subtype *tpe, const char *val);
sql_extern atom *atom_general( sql_allocator *sa, sql_subtype *tpe, const char *val);
#ifdef HAVE_HGE
sql_extern atom *atom_dec( sql_allocator *sa, sql_subtype *tpe, hge val, double dval);
#else
sql_extern atom *atom_dec( sql_allocator *sa, sql_subtype *tpe, lng val, double dval);
#endif
sql_extern atom *atom_ptr( sql_allocator *sa, sql_subtype *tpe, void *v);

sql_extern int atom_neg( atom *a );
sql_extern unsigned int atom_num_digits( atom *a );

/* duplicate atom */
sql_extern atom *atom_dup( sql_allocator *sa, atom *a);

/* cast atom a to type tp (success == 1, fail == 0) */
sql_extern int atom_cast(sql_allocator *sa, atom *a, sql_subtype *tp);

sql_extern char *atom2string(sql_allocator *sa, atom *a);
sql_extern char *atom2sql(atom *a);
sql_extern sql_subtype *atom_type(atom *a);

#ifdef HAVE_HGE
sql_extern hge atom_get_int(atom *a);
#else
sql_extern lng atom_get_int(atom *a);
#endif

sql_extern int atom_cmp(atom *a1, atom *a2);

sql_extern atom *atom_add(atom *a1, atom *a2);
sql_extern atom *atom_sub(atom *a1, atom *a2);
sql_extern atom *atom_mul(atom *a1, atom *a2);
sql_extern int atom_inc(atom *a);
sql_extern int atom_is_true(atom *a);
sql_extern int atom_is_zero(atom *a);

#ifdef HAVE_HGE
sql_extern hge scales[39];
#else
sql_extern lng scales[19];
#endif

sql_extern atom* atom_absolute_min(sql_allocator *sa, sql_subtype* tpe);
sql_extern atom* atom_absolute_max(sql_allocator *sa, sql_subtype* tpe);
sql_extern atom* atom_zero_value(sql_allocator *sa, sql_subtype* tpe);

#endif /* _SQL_ATOM_H_ */

