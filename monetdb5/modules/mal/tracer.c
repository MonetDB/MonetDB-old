/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 *
 * This module contains the MAL primitives to control the logging system.
 * If the tracer fails then there is no easy way to report this.
 * This is marked with the tracer marker.
 *
 */

#include "monetdb_config.h"
#include "tracer.h"
#include "gdk_tracer.h"


str 
TRCRflush_buffer(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;

    if ( GDKtracer_flush_buffer() == GDK_FAIL)
        throw(TRACER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED;
}


str
TRCRset_log_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	
    int *lvl;
	(void) cntxt;
	(void) mb;

	lvl = (int*) getArgReference_str(stk,pci,1);
    if( GDKtracer_set_log_level(lvl) == GDK_FAIL)
        throw(TRACER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED; 
}


str
TRCRreset_log_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;

    if( GDKtracer_reset_log_level())
        throw(TRACER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED;
}


str
TRCRset_flush_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{	
    int *lvl;
	(void) cntxt;
	(void) mb;

	lvl = (int*) getArgReference_str(stk,pci,1);
    if( GDKtracer_set_flush_level(lvl))
        throw(TRACER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    return MAL_SUCCEED;
}


str
TRCRreset_flush_level(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;

    if( GDKtracer_reset_flush_level() == GDK_FAIL)
        throw(TRACER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED;
}
