/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#include "benchmark.h"

#include "monetdb_config.h"

static unsigned long long memtrace_current_memory_bytes = 0;
static unsigned long long memtrace_memory_peak = 0;

#if defined(HAVE_MALLOC_H) && defined(HAVE_STRING_H)
#include <string.h>
#include <malloc.h>

#ifdef __MALLOC_DEPRECATED //if this isn't defined MALLOC_HOOKS aren't supported, probably
// We are using malloc/free hooks which are deprecated, so we have to ignore the warnings
// (This is obviously bad practice, but the alternative is having to recompile Python and then tracing both PyMemAlloc/Realloc and GDKmalloc/realloc calls, this is much easier, and we aren't using them in a thread context and no thread safety is why they are deprecated in the first place) 

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
/* Prototypes for our hooks.  */
static void *my_malloc_hook (size_t, const void *);
static void my_free_hook (void*, const void *);
static void add_ptr(void *ptr, size_t size);
static void remove_ptr(void *ptr);
static void* (*old_malloc_hook)(size_t, const void*)=NULL;
static void (*old_free_hook)(void*, const void*)=NULL;

//we keep a datastore of pointers and the amount of size that was malloced when the pointer was created
static void **memtrace_pointers = NULL;   //the pointers
static size_t *memtrace_sizes = NULL;     //the sizes
static long memtrace_max_size = 100;    //the max size of the _pointers and _sizes arrays
static long memtrace_current_size = -1; //the current index

void add_ptr(void *ptr, size_t size)
{
	memtrace_current_size++;
	if (memtrace_current_size >= memtrace_max_size)
	{
		//if the max_size is exceeded extend the array
		void **new_ptrs = malloc(sizeof(void*) * memtrace_max_size * 2);
		size_t *new_sizes = malloc(sizeof(size_t) * memtrace_max_size * 2);
		memcpy(new_ptrs, memtrace_pointers, memtrace_max_size * sizeof(void*));
		memcpy(new_sizes, memtrace_sizes, memtrace_max_size * sizeof(size_t));
		free(memtrace_pointers); free(memtrace_sizes);
		memtrace_pointers = new_ptrs; memtrace_sizes = new_sizes;
		memtrace_max_size = memtrace_max_size * 2;
	}

	memtrace_pointers[memtrace_current_size] = ptr;
	memtrace_sizes[memtrace_current_size] = size;
	memtrace_current_memory_bytes += size;
	if (memtrace_current_memory_bytes > memtrace_memory_peak) memtrace_memory_peak = memtrace_current_memory_bytes;
}

void remove_ptr(void *ptr)
{
	//because malloc hooks inherently aren't thread safe we don't care to make this thread safe either
	long i;
	for(i = 0; i <= memtrace_current_size; i++)
	{
		if (memtrace_pointers[i] == ptr)
		{
			memtrace_current_memory_bytes -= memtrace_sizes[i];
			memtrace_pointers[i] = memtrace_pointers[memtrace_current_size];
			memtrace_sizes[i] = memtrace_sizes[memtrace_current_size];
			memtrace_current_size--;
			return;
		}
	}
}

void init_hook (void)
{
	if (memtrace_pointers == NULL) {
		memtrace_pointers = malloc(memtrace_max_size * sizeof(void*));
		memtrace_sizes = malloc(memtrace_max_size * sizeof(size_t));
		memtrace_current_size = -1;
	}
	memtrace_current_memory_bytes = 0;
	memtrace_memory_peak = 0;

	old_malloc_hook = __malloc_hook;
	old_free_hook = __free_hook;
	__malloc_hook = my_malloc_hook;
	__free_hook = my_free_hook;
}

void revert_hook (void)
{
	__malloc_hook = old_malloc_hook;
	__free_hook = old_free_hook;
	free(memtrace_pointers);
	free(memtrace_sizes);
	memtrace_current_size = -1;
	memtrace_max_size = 100;
	memtrace_pointers = NULL; memtrace_sizes = NULL;
}

static void *my_malloc_hook (size_t size, const void *caller)
{
	void *result; (void) caller;
	/* Restore all old hooks */
	__malloc_hook = old_malloc_hook;
	__free_hook = old_free_hook;
	/* Call recursively */
	result = malloc (size);
	add_ptr(result, size);
	/* Restore our own hooks */
	__malloc_hook = my_malloc_hook;
	__free_hook = my_free_hook;
	return result;
}

static void my_free_hook (void *ptr, const void *caller)
{
	(void) caller;
	/* Restore all old hooks */
	__malloc_hook = old_malloc_hook;
	__free_hook = old_free_hook;
	/* Call recursively */
	free (ptr);
	remove_ptr(ptr);
	/* Restore our own hooks */
	__malloc_hook = my_malloc_hook;
	__free_hook = my_free_hook;
}
#else
void init_hook (void) {}
void revert_hook (void) {}
#endif
#else
void init_hook (void) {}
void revert_hook (void) {}
#endif

#ifdef HAVE_TIME_H
#include <time.h>

double GET_ELAPSED_TIME(double start_time, double end_time)
{
	return (double)(end_time - start_time) / CLOCKS_PER_SEC;
}

double timer(void)
{
	return clock();
}
#else
double GET_ELAPSED_TIME(double start_time, double end_time) { return 0; }
double timer(void) { return 0; }
#endif

unsigned long long GET_MEMORY_PEAK(void)
{
	return memtrace_memory_peak;
}

unsigned long long GET_MEMORY_USAGE(void)
{
	return memtrace_current_memory_bytes;
}

void reset_hook(void)
{
	memtrace_current_memory_bytes = 0;
	memtrace_memory_peak = 0;
}
