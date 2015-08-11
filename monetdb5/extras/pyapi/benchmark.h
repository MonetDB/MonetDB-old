/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raaasveldt
 * Contains hooks for malloc/free as well as timer 
 */

#ifndef _PYAPI_BENCHMARK_LIB_
#define _PYAPI_BENCHMARK_LIB_

#include "monetdb_config.h"
#include "pyapi.h"

#ifdef _PYAPI_TESTING_

#ifdef HAVE_TIME_H
#include <time.h>
typedef struct timespec time_storage;
#else
typedef int time_storage;
#endif

//returns the current time
void timer(time_storage*);
double GET_ELAPSED_TIME(time_storage start_time, time_storage end_time);
//sets up malloc hooks, not thread safe, do not use in thread context
void init_hook (void);
void reset_hook(void);
//detaches malloc hooks, not thread safe
void revert_hook (void);
//gets peak memory usage between init_hook() and revert_hook() calls
unsigned long long GET_MEMORY_PEAK(void);
//get current memory usage (note that this only measures the malloc calls between init_hook() and revert_hook() calls)
unsigned long long GET_MEMORY_USAGE(void);


#endif

#endif
