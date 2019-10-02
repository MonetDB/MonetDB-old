/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 *
 * This module contains the MAL primitives to control the logging system.
 * If the stalker fails then there is no easy way to report this.
 * This is marked with the STALKER marker.
 *
 */

#include "monetdb_config.h"
#include "stalker.h"
#include "gdk_stalker.h"


str 
STLKRflush_buffer(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;

    if ( GDKstalker_flush_buffer() == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED;
}


str
STLKRset_log_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	int *lvl;
	(void) cntxt;
	(void) mb;

	lvl = (int*) getArgReference_str(stk,pci,1);
    if( GDKstalker_set_log_level(lvl) == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED; 
}


str
STLKRreset_log_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
    if( GDKstalker_reset_log_level())
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED;
}


str
STLKRset_flush_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	int *lvl;
	(void) cntxt;
	(void) mb;

	lvl = (int*) getArgReference_str(stk,pci,1);
    if( GDKstalker_set_flush_level(lvl))
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED;
}


str
STLKRreset_flush_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
    if( GDKstalker_reset_flush_level() == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED;
}
