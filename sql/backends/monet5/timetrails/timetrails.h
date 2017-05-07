/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

/* In your own module, replace "timetrails" & "udf" by your module's name */

#ifndef _SQL_TIMETRAILS_H_
#define _SQL_TIMETRAILS_H_
#include "sql.h"
#include <string.h>

/* This is required as-is (except from renaming "timetrails" & "udf" as suggested
 * above) for all modules for correctly exporting function on Unix-like and
 * Windows systems. */

#ifdef WIN32
#ifndef LIBtimetrails
#define udf_export extern __declspec(dllimport)
#else
#define udf_export extern __declspec(dllexport)
#endif
#else
#define udf_export extern
#endif

/* export MAL wrapper functions */

#endif /* _SQL_TIMETRAILS_H_ */
