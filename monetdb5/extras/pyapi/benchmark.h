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

//starts a performance timer with clock()
void start_timer(void);
//ends the timer
void end_timer(void);
//sets up malloc hooks, not thread safe, do not use in thread context
void init_hook (void);
void reset_hook(void);
//detaches malloc hooks, not thread safe
void revert_hook (void);
//gets peak memory usage between init_hook() and revert_hook() calls
unsigned long long GET_MEMORY_PEAK(void);
//get current memory usage (note that this only measures the malloc calls between init_hook() and revert_hook() calls)
unsigned long long GET_MEMORY_USAGE(void);
//gets the elapsed time of a timer started with start_timer() and end_timer()
double GET_ELAPSED_TIME(void);

#endif
