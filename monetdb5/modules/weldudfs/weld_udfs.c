/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "gdk.h"
#include "mal.h"
#include "mtime.h"
#include "pcre_pub.h"

typedef struct {
	char *data;
	int64_t len;
} i8vec;

MT_Lock initLock MT_LOCK_INITIALIZER("udfs_init");

mal_export void state_init(i8vec *op, int64_t *state_ptr);
mal_export void like(int64_t *state_ptr, i8vec *col, i8vec *pattern, i8vec *exc, int8_t *result);
mal_export void year(int32_t *col, int32_t *result);

void state_init(i8vec *op, int64_t *state_ptr) {
	(void)op;
	void *ptr = calloc(0, sizeof(void*));
	*state_ptr = (int64_t)ptr;
}

void like(int64_t *state_ptr, i8vec *col, i8vec *pattern, i8vec *exc, int8_t *result) {
	(void)exc;
	int64_t *adr = (void*)*state_ptr;
	RE *re = (RE*)(*adr);
	if (re == NULL) {
		MT_lock_set(&initLock);
		if (re == NULL) {
			/* Create a RE struct and save it in the given mem location */
			int nr = re_simple(pattern->data);
			re = re_create(pattern->data, nr);
			*adr = (int64_t)re;
		}
		MT_lock_unset(&initLock);
	}
	*result = (int8_t)re_match_no_ignore(col->data, re);
}

void year(int32_t *col, int32_t *result) {
	(void)MTIMEdate_extract_year(result, col);
}
