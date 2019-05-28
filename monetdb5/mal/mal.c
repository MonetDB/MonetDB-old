/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

/* (author) M. Kersten */
#include "monetdb_config.h"
#include "mal.h"

char 	monet_cwd[FILENAME_MAX] = { 0 };
size_t 	monet_memory = 0;
char 	monet_characteristics[4096];

#ifdef HAVE_HGE
int have_hge;
#endif

MT_Lock     mal_contextLock = MT_LOCK_INITIALIZER("mal_contextLock");
MT_Lock     mal_remoteLock = MT_LOCK_INITIALIZER("mal_remoteLock");
MT_Lock     mal_profileLock = MT_LOCK_INITIALIZER("mal_profileLock");
MT_Lock     mal_copyLock = MT_LOCK_INITIALIZER("mal_copyLock");
MT_Lock     mal_delayLock = MT_LOCK_INITIALIZER("mal_delayLock");
MT_Lock     mal_oltpLock = MT_LOCK_INITIALIZER("mal_oltpLock");
