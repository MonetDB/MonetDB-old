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

#include <stdbool.h>
#include <stdio.h>

/*
//! Returns the ID of the shared memory, returns -1 on failure
int init_shared_memory(char *keystring, int id, size_t size, int flags);
//! Returns the pointer to the shared memory, argument is the return value of init_shared_memory
void *get_shared_memory_address(int memory_id);
//! Release the chunk of shared memory
bool release_shared_memory(int memory_id, void *ptr);*/

//! Initialize the shared memory module
void initialize_shared_memory(void);
//! Not thread safe
void* create_shared_memory(int id, size_t size);
//! Not thread safe
bool release_shared_memory(void *ptr);
//! Not thread safe
int get_unique_shared_memory_id(int offset);
//! This is thread safe
void *get_shared_memory(int id, size_t size);

//! Returns semaphore ID, id = unique id within the program, count = amount of semaphores
int create_process_semaphore(int id, int count);
//! Returns semaphore ID
int get_process_semaphore(int id, int count);
//! Returns value of semaphore <number> at semaphore id <sem_id>
int get_semaphore_value(int sem_id, int number);
//! Change the semaphore <number> at semaphore id <sem_id> value by <change> (change = 1 means +1, not set the value to 1)
bool change_semaphore_value(int sem_id, int number, int change);
//! Release semaphore at sem_id
bool release_process_semaphore(int sem_id);



#endif /* _SHAREDMEMORY_LIB_ */
