/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "stalker.h"
#include "gdk_stalker.h"


int GDK_result = 0;

str 
STLKRflush_buffer(void)
{
    GDK_result = GDKstalker_flush_buffer();
    if(GDK_result == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);
    
    return MAL_SUCCEED;
}


str
STLKRset_log_level(void *ret, int *lvl)
{
    (void) ret;
    GDK_result = GDKstalker_set_log_level(lvl);
    if(GDK_result == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED; 
}


str
STLKRreset_log_level(void)
{
    GDK_result = GDKstalker_reset_log_level();
    if(GDK_result == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED;
}


str
STLKRset_flush_level(void *ret, int *lvl)
{
    (void) ret;
    GDK_result = GDKstalker_set_flush_level(lvl);
    if(GDK_result == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED;
}


str
STLKRreset_flush_level()
{
    GDK_result = GDKstalker_reset_flush_level();
    if(GDK_result == GDK_FAIL)
        throw(STALKER, __FILE__, "%s:%s", __func__, OPERATION_FAILED);

    return MAL_SUCCEED;
}
