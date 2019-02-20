/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _MAL_EMBEDDED_H
#define _MAL_EMBEDDED_H

#include "monetdb_config.h"

#include "mal_client.h"
#include "mal_import.h"

/* #define MAL_EMBEDDED_DEBUG  */

#define MAXMODULES  128

typedef struct {
	str modnme, source;
} malSignatures;

mal_export malSignatures malModules[];

mal_export str malEmbeddedBoot(Client c);
mal_export str malExtraModulesBoot(Client c, malSignatures extraMalModules[]);
mal_export str malEmbeddedStop(void);
mal_export str malEmbeddedRestart(Client c);

#endif /*  _MAL_EMBEDDED_H*/
