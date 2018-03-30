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

#define structDef(CTYPE, WTYPE) \
	typedef struct {            \
		CTYPE *data;            \
		lng length;             \
	} WeldVec##WTYPE;

#define funcDef(WTYPE)                                                                             \
	mal_export void weldJoinNoCandList##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r,                \
											  bit *nilmatches, lng *estimate, void *result);       \
	mal_export void weldJoin##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r, WeldVec##WTYPE *sl,      \
									WeldVec##WTYPE *sr, bit *nilmatches, lng *estimate,            \
									void *result);                                                 \
	mal_export void weldDifferenceNoCandList##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r,          \
													bit *nilmatches, lng *estimate, void *result); \
	mal_export void weldDifference##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r,                    \
										  WeldVec##WTYPE *sl, WeldVec##WTYPE *sr, bit *nilmatches, \
										  lng *estimate, void *result);                            \
	mal_export void weldIntersectNoCandList##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r,           \
												   bit *nilmatches, lng *estimate, void *result);  \
	mal_export void weldIntersect##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r, WeldVec##WTYPE *sl, \
										 WeldVec##WTYPE *sr, bit *nilmatches, lng *estimate,       \
										 void *result);

structDef(bte, i8);
structDef(int, i32);
structDef(lng, i64);
structDef(flt, f32);
structDef(dbl, f64);

funcDef(i8);
funcDef(i32);
funcDef(i64);
funcDef(f32);
funcDef(f64);

#endif
