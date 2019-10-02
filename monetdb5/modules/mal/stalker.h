/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _STLKR_H
#define _STLKR_H

#include "mal.h"
#include "mal_interpreter.h"

mal_export str STLKRflush_buffer(void);
mal_export str STLKRset_log_level(void *ret, int *lvl);
mal_export str STLKRreset_log_level(void);
mal_export str STLKRset_flush_level(void *ret, int *lvl);
mal_export str STLKRreset_flush_level(void);

#endif
