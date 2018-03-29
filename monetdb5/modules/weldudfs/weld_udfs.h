/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */
#ifndef _WELD_UDFS_H_
#define _WELD_UDFS_H_

#include "monetdb_config.h"
#include "gdk.h"
#include "mal.h"
#include "mal_interpreter.h"

typedef struct {
	void* data;
	lng length;
} WeldVec;

#define joinStructAndFunc(CTYPE, WTYPE)                                                       \
	typedef struct {                                                                          \
		CTYPE *data;                                                                          \
		lng length;                                                                           \
	} WeldVec##WTYPE;                                                                         \
	mal_export void weldJoinNoCandList##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r,           \
											  bit *nilmatches, lng *estimate, void *result);  \
	mal_export void weldJoin##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r, WeldVec##WTYPE *sl, \
									WeldVec##WTYPE *sr, bit *nilmatches, lng *estimate,       \
									void *result);
joinStructAndFunc(bte, i8);
joinStructAndFunc(int, i32);
joinStructAndFunc(void, i64);
joinStructAndFunc(flt, f32);
joinStructAndFunc(dbl, f64);

#endif
