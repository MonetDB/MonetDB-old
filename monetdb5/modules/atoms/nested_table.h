/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */
#ifndef _MONET_NESTED_TABLE_H_
#define _MONET_NESTED_TABLE_H_

#include "monetdb_config.h"
#include "mal.h"


// index in the array BATatoms
extern int TYPE_nested_table;

typedef struct {
	oid count;
	oid values[FLEXIBLE_ARRAY_MEMBER];
} nested_table;


#endif /* _MONET_NESTED_TABLE_H_ */
