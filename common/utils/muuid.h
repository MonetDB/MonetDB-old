/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef MT_SEEN_MUUID_H
#define MT_SEEN_MUUID_H 1

/* this function is (currently) only used in msabaoth and sql;
 * msabaoth is part of monetdb5 and we want this function to be
 * exported so that the call in sql can be satisfied by the version
 * that is included in monetdb5 */

#ifdef HAVE_EMBEDDED
#define muuid_export extern __attribute__((__visibility__("hidden")))
#elif defined(WIN32)
#if !defined(LIBMUUID)
#define muuid_export extern __declspec(dllimport)
#else
#define muuid_export extern __declspec(dllexport)
#endif
#else
#define muuid_export extern
#endif

muuid_export char* generateUUID(void);

#endif

/* vim:set ts=4 sw=4 noexpandtab: */
