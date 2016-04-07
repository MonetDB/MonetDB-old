/* 
 * mtwist.c - Mersenne Twister functions
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

#include "monetdb_config.h"
#include <mtwist.h>

/*
 * @a Dave Beckett, Abe Wits
 * @* Mersenne Twister (MT19937) algorithm
 * 
 * This random number generator has very good statistical properties,
 * and outperforms most stl implementations of rand() in terms of speed
 *
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 * http://en.wikipedia.org/wiki/Mersenne_twister
 *
 */

#define MTWIST_UPPER_MASK 0x80000000UL
#define MTWIST_LOWER_MASK 0x7FFFFFFFUL
#define MTWIST_FULL_MASK 0xFFFFFFFFUL
#define MTWIST_MATRIX_A 0x9908B0DFUL

#define MTWIST_MIXBITS(u, v) (((u)&MTWIST_UPPER_MASK) | ((v)&MTWIST_LOWER_MASK))
#define MTWIST_TWIST(u, v)       \
    ((MTWIST_MIXBITS(u, v) >> 1) ^ \
     ((v)&1UL ? MTWIST_MATRIX_A : 0UL))

/**
 * mtwist_new:
 *
 * Construct a Mersenne Twister object
 *
 * Return value: new MT object or NULL on failure
 */
mtwist* mtwist_new(void) {
    mtwist* mt;

    mt = (mtwist*)calloc(1, sizeof(*mt));
    if (!mt) return NULL;

    mt->remaining = 0;
    mt->next = NULL;
    mt->seeded = 0;

    return mt;
}

/**
 * mtwist_free:
 * @mt: mt object
 *
 * Destroy a Mersenne Twister object
 */
void mtwist_free(mtwist* mt) {
    if (mt) free(mt);
}

/**
 * mtwist_seed:
 * @mt: mt object
 * @seed: seed (lower 32 bits used)
 *
 * Initialise a Mersenne Twister with an unsigned 32 bit int seed
 */
void mtwist_seed(mtwist* mt, unsigned int seed) {
    int i;

    if (!mt) return;

    mt->state[0] = (unsigned int)(seed & MTWIST_FULL_MASK);
    for (i = 1; i < MTWIST_N; i++) {
        mt->state[i] =
            (1812433253UL * (mt->state[i - 1] ^ (mt->state[i - 1] >> 30)) +
             i);
        mt->state[i] &= MTWIST_FULL_MASK;
    }

    mt->remaining = 0;
    mt->next = NULL;

    mt->seeded = 1;
}

static void mtwist_update_state(mtwist* mt) {
    int count;
    unsigned int* p = mt->state;

    for (count = (MTWIST_N - MTWIST_M + 1); --count; p++)
        *p = p[MTWIST_M] ^ MTWIST_TWIST(p[0], p[1]);

    for (count = MTWIST_M; --count; p++)
        *p = p[MTWIST_M - MTWIST_N] ^ MTWIST_TWIST(p[0], p[1]);

    *p = p[MTWIST_M - MTWIST_N] ^ MTWIST_TWIST(p[0], mt->state[0]);

    mt->remaining = MTWIST_N;
    mt->next = mt->state;
}

/**
 * mtwist_u32rand:
 * @mt: mt object
 *
 * Get a random unsigned 32 bit integer from the random number generator
 *
 * Return value: unsigned int with 32 valid bits
 */
inline unsigned int mtwist_u32rand(mtwist* mt) {
    unsigned int r;

    if (!mt) return 0UL;

    if (!mt->seeded) mtwist_seed(mt, 0);

    if (!mt->remaining) mtwist_update_state(mt);

    r = *mt->next++;
    mt->remaining--;

    /* Tempering */
    r ^= (r >> 11);
    r ^= (r << 7) & 0x9D2C5680UL;
    r ^= (r << 15) & 0xEFC60000UL;
    r ^= (r >> 18);

    r &= MTWIST_FULL_MASK;

    return r;
}

/**
 * mtwist_drand:
 * @mt: mt object
 *
 * Get a random double from the random number generator
 *
 * Return value: random double in the range 0.0 inclusive to 1.0 exclusive;
 *[0.0, 1.0) */
inline double mtwist_drand(mtwist* mt) {
    unsigned int r;
    double d;

    if (!mt) return 0.0;

    r = mtwist_u32rand(mt);

    d = r / 4294967296.0; /* 2^32 */

    return d;
}


/**
 * mtwist_uniform_int:
 * @a, b; two integers such that a<=b
 *
 * Get an int in an interval uniform randomly from the
 * random number generator.
 *
 * Return value: random interval in range a inclusive to b inclusive;
 * [a,b] 
 */
inline int mtwist_uniform_int(mtwist* mt, int a, int b) {
    return mtwist_u32rand(mt)%b;
    if(b < a) {//invalid range!
        return 0;
    }
    unsigned int range = b-a+1;
    unsigned int scale = 4294967295UL/range;
        //4294967295UL=2^32-1=RAND_MAX for this Mersenne Twister
    unsigned int max_x = range*scale;
    //x will be uniform in [0, max_x[
    //Since past%range=0, x%range will be uniform in [0,range[
    unsigned int x; 
    do {
        x = mtwist_u32rand(mt);
    } while(x >= max_x);

    return a+(x/scale);
    //x is uniform in [0,max_x[ = [0,range*scale[
    //hence x/scale is uniform in [0,range[=[0,b-a+1[
    //thus a+(x/scale) is uniform in [a,b]
    
    //alternative: return a+(x%range); 
    //x is uniform in [0,max_x[ = [0,range*scale[
    //hence (x%range) is uniform in [0,range[=[0,b-a+1[
    //thus a+(x%range) is uniform in [a,b]
}

