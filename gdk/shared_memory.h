/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains a number of functions for acquiring/releasing shared memory
 */

#ifndef _SHAREDMEMORY_LIB_
#define _SHAREDMEMORY_LIB_

#include "monetdb_config.h"
#include "gdk.h"

#include <stddef.h>

//! Initialize the shared memory module
str initialize_shared_memory(void);
//! Not thread safe
str create_shared_memory(int id, size_t size, void **return_ptr);
//! This is thread safe
str release_shared_memory(void *ptr);
//! Not thread safe
int get_unique_shared_memory_id(int offset);
//! This is thread safe
str get_shared_memory(int id, size_t size, void **return_ptr);

#endif /* _SHAREDMEMORY_LIB_ */
