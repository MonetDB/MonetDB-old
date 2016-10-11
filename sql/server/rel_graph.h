/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

#ifndef _REL_GRAPH_H_
#define _REL_GRAPH_H_

#include "rel_semantic.h"
#include "sql_semantic.h"

sql_rel* rel_graph_reaches(mvc *sql, sql_rel *rel, symbol *sq, int context);
sql_exp* rel_graph_cheapest_sum(mvc *sql, sql_rel **rel, symbol *sq, int context);


#endif /* _REL_GRAPH_H_ */
