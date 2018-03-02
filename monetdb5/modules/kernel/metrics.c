
/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "metrics.h"
#include "gdk.h"
#include "mal.h"
#include "mal_client.h"
#include "mal_instruction.h"
#include "mal_interpreter.h"

#include "gdk_posix.h"

mal_export str METRICSrsssize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) cntxt;
    (void) mb;
    lng *res = getArgReference_lng(stk, pci, 0);
    *res = (lng)MT_getrss();

    return MAL_SUCCEED;
}

mal_export str METRICSgetvmsize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    (void) cntxt;
    (void) mb;
    lng *res = getArgReference_lng(stk, pci, 0);

#ifdef WIN32
	MEMORYSTATUSEX psvmemCounters;
	psvmemCounters.dwLength = sizeof(MEMORYSTATUSEX); 
	
	GlobalMemoryStatusEx(&psvmemCounters);

	*res = psvmemCounters.ullTotalVirtual - psvmemCounters.ullAvailVirtual;
#else
    *res = 0;
#endif

    return MAL_SUCCEED;
}
