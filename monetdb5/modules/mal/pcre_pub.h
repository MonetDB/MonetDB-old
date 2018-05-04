/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

/* current implementation assumes simple %keyword% [keyw%]* */
typedef struct RE {
	char *k;
	int search;
	int skip;
	int len;
	struct RE *n;
} RE;

int re_simple(const char *pat);
RE *re_create(const char *pat, int nr);
int re_match_no_ignore(const char *s, RE *pattern);
