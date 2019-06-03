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

#define MAXMODULES  128

mal5_export str malModules[];

mal5_export str malEmbeddedBoot(void);
mal5_export str malExtraModulesBoot(Client c, str extraMalModules[], char* mal_scripts);
mal5_export void malEmbeddedReset(void);
mal5_export _Noreturn void malEmbeddedStop(int status);

#endif /*  _MAL_EMBEDDED_H*/
