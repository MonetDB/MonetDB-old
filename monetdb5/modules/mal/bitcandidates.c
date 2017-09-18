/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "bitcandidates.h"
#include "mal.h"
#include "mal_instruction.h"
#include "mal_interpreter.h"

str
BCLcompress(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    bat *ret = getArgReference_bat(stk,pci,0);
    bat *val = getArgReference_bat(stk,pci,1);
	BAT *b;
    (void) cntxt;
    (void) mb;
	b = BATdescriptor(*val);
	if( b == NULL)
		throw(MAL,"compress",INTERNAL_BAT_ACCESS);
    BBPkeepref(*ret = *val);
	return MAL_SUCCEED;
}

str
BCLdecompress(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    bat *ret = getArgReference_bat(stk,pci,0);
    bat *val = getArgReference_bat(stk,pci,1);
	BAT *b;
    (void) cntxt;
    (void) mb;
	b = BATdescriptor(*val);
	if( b == NULL)
		throw(MAL,"decompress",INTERNAL_BAT_ACCESS);
    BBPkeepref(*ret = *val);
	return MAL_SUCCEED;
}
