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
