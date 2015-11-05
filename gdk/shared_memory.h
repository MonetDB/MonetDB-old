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
 
#ifndef _WIN32
#include "monetdb_config.h"
#include "gdk.h"

#include <stddef.h>

#define SHM_SHARED 1
#define SHM_MEMMAP 2
#define SHM_EITHER 3

//! Initialize the shared memory module
str initialize_shared_memory(void);
//! Not thread safe if 'reg=true', otherwise thread safe
str create_shared_memory(int id, size_t size, bool reg, void **return_ptr, lng *return_shmid);
//! This is thread safe
str release_shared_memory(void *ptr);
//! Only release the pointer of this process, doesn't actually delete the memory
str release_shared_memory_ptr(void *ptr);
//! Thread safe
str release_shared_memory_shmid(int memory_id, void *ptr);
//! Not thread safe
int get_unique_shared_memory_id(int offset);
//! Not thread safe if 'reg=true', otherwise thread safe
str get_shared_memory(int id, size_t size, bool reg, void **return_ptr, lng *return_shmid);

str create_process_semaphore(int id, int count, int *semid);
str get_process_semaphore(int sem_id, int count, int *semid);
str get_semaphore_value(int sem_id, int number, int *semval);
str change_semaphore_value(int sem_id, int number, int change);
str change_semaphore_value_timeout(int sem_id, int number, int change, int timeout_mseconds, bool *succeed);
str release_process_semaphore(int sem_id);

extern int memtype;
#endif

#endif /* _SHAREDMEMORY_LIB_ */
