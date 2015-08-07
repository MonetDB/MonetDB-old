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

//! Returns semaphore ID, id = unique id within the program, count = amount of semaphores
int create_process_semaphore(int id, int count);
//! Returns semaphore ID
int get_process_semaphore(int id, int count);
//! Returns value of semaphore <number> at semaphore id <sem_id>
int get_semaphore_value(int sem_id, int number);
//! Change the semaphore <number> at semaphore id <sem_id> value by <change> (change = 1 means +1, not set the value to 1)
int change_semaphore_value(int sem_id, int number, int change);
//! Release semaphore at sem_id
int release_process_semaphore(int sem_id);



#endif /* _SHAREDMEMORY_LIB_ */
