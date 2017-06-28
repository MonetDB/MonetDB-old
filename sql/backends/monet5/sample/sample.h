/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/* In your own module, replace "UDF" & "udf" by your module's name */

#ifndef _SQL_WEIGHTEDSAMPLE_H_
#define _SQL_WEIGHTEDSAMPLE_H_
#include "sql.h"

/* export MAL wrapper functions */

extern char * UDFBATweightedsample(bat *ret, const bat *arg, const lng *cnt);

#endif

