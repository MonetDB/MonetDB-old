/* 
 * mtwist.h - Mersenne Twister functions
 *
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <http://unlicense.org/>
 *
 */

#ifndef _SEEN_MTWIST_H
#define _SEEN_MTWIST_H 1

#ifdef NATIVE_WIN32
#if !defined(LIBMTWIST) && !defined(LIBMUTILS) && !defined(LIBGDK) && !defined(LIBMEROUTIL)
#define mtwist_export extern __declspec(dllimport)
#else
#define mtwist_export extern __declspec(dllexport)
#endif
#else
#define mtwist_export extern
#endif


#define MTWIST_N 624
#define MTWIST_M 397

/* Mersenne Twister library state */
struct mtwist_s {
	/* MT buffer holding N 32 bit unsigned integers */
	unsigned int state[MTWIST_N];

	/* Pointer into above - next int to use */
	unsigned int* next;

	/* Number of remaining integers in state before an update is needed */
	unsigned int remaining;

	/* 1 if a seed was given */
	unsigned int seeded : 1;
};

/* Mersenne Twister state */
typedef struct mtwist_s mtwist;

/* constructor */
mtwist_export mtwist* mtwist_new(void);

/* destructor */
mtwist_export void mtwist_free(mtwist* mt);

/* methods */
mtwist_export void mtwist_seed(mtwist* mt, unsigned int seed);
mtwist_export unsigned int mtwist_u32rand(mtwist* mt);
mtwist_export double mtwist_drand(mtwist* mt);
mtwist_export int mtwist_uniform_int(mtwist* mt, int a, int b);

#endif
