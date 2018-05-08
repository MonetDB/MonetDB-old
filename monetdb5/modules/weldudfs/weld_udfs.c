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

mal_export void like_pattern_init(i8vec *pattern, i8vec *exc, int64_t *state_ptr);
mal_export void like(int64_t *state_ptr, i8vec *col, int8_t *result);
mal_export void like_pattern_cleanup(int64_t *state_ptr, int64_t *);
mal_export void year(int32_t *col, int32_t *result);

void like_pattern_init(i8vec *pattern, i8vec *exc, int64_t *state_ptr) {
	(void)exc;
	int nr = re_simple(pattern->data);
	RE *re = re_create(pattern->data, nr);
	*state_ptr = (int64_t)re;
}

void like(int64_t *state_ptr, i8vec *col, int8_t *result) {
	RE *re = (RE*)(*state_ptr);
	*result = (int8_t)re_match_no_ignore(col->data, re);
}

void like_pattern_cleanup(int64_t *state_ptr, int64_t *result) {
	RE *re = (RE*)(*state_ptr);
	re_destroy(re);
	*result = 0;
}

void year(int32_t *col, int32_t *result) {
	(void)MTIMEdate_extract_year(result, col);
}
