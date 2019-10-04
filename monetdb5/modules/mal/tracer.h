/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _TRACER_H
#define _TRACER_H

#include "mal.h"
#include "mal_interpreter.h"

mal_export str TRCRflush_buffer(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p);
mal_export str TRCRset_log_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p);
mal_export str TRCRreset_log_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p);
mal_export str TRCRset_flush_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p);
mal_export str TRCRreset_flush_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p);

#endif
