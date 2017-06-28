/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/* monetdb_config.h must be the first include in each .c file */

#include "monetdb_config.h"
#include "sample.h"

#ifdef notdefined //!!!TODO

/* MAL wrapper */
char *
UDFBATweightedsample(bat *ret, const bat *arg, const lng *cnt)
{//bat = identifier, BAT is actual bat, BATdescriptor turns ID into BAT
	BAT *res = NULL, *src = NULL;
	char *msg = NULL;

	/* assert calling sanity */
	assert(ret != NULL && arg != NULL);

	/* bat-id -> BAT-descriptor */
	if ((src = BATdescriptor(*arg)) == NULL)
		throw(MAL, "batudf.reverse", RUNTIME_OBJECT_MISSING);
	printf("Count: %lld\n", *cnt);

	//TODO Type checking
	/* do the work */
	//msg = UDFBATreverse_ ( &res, src );//TODO
	throw(MAL, "batudf.reverse", "LOLFAIL");//TODO
	res = _BATsample(arg, *cnt, BAT *cdf)
	
	/* release input BAT-descriptor */
	//BBPunfix(src->batCacheid);

	//if (msg == MAL_SUCCEED) {
		/* register result BAT in buffer pool */
	//	BBPkeepref((*ret = res->batCacheid));
	//}
	return msg;
}

#endif
