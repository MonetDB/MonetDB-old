/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * H. Muehleisen, M. Raasveldt
 * Inverse RAPI
 */
#ifndef _INVERSE_RAPI_LIB_
#define _INVERSE_RAPI_LIB_

int monetdb_startup(char* dir, char silent);
char* monetdb_query(char* query, void** result);
void monetdb_cleanup_result(void* output);

#endif
