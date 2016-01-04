/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

typedef struct {
	sht type;		/* type of index entity */
	sht width;		/* width of hash entries */
	sht pieces;		/* number of hash chunks */
	BUN chunk;		/* chunk size (last may be different) */
	BUN nil;		/* nil representation */
	BUN cap;		/* collision list size */
	BUN mask;		/* number of hash buckets-1 (power of 2) */
	void *Hash;		/* hash table */
	void *Link;		/* collision list */
	Heap *heap;		/* heap where the hash is stored */
} Hash;

#define HASHnil(H)	(H)->nil

#define mix_bte(X)	((unsigned int) (X))
#define mix_sht(X)	((unsigned int) (X))
#define mix_int(X)	(((X)>>7)^((X)>>13)^((X)>>21)^(X))
#define mix_lng(X)	mix_int((unsigned int) ((X) ^ ((X) >> 32)))
#ifdef HAVE_HGE
#define mix_hge(X)	mix_int((unsigned int) ((X) ^ ((X) >> 32) ^ \
						((X) >> 64) ^ ((X) >> 96)))
#endif
#define hash_loc(H,V)	hash_any(H,V)
#define hash_var(H,V)	hash_any(H,V)
#define hash_any(H,V)	(ATOMhash((H)->type, (V)) & (H)->mask)
#define hash_bte(H,V)	(assert(((H)->mask & 0xFF) == 0xFF), (BUN) mix_bte(*(const unsigned char*) (V)))
#define hash_sht(H,V)	(assert(((H)->mask & 0xFFFF) == 0xFFFF), (BUN) mix_sht(*(const unsigned short*) (V)))
#define hash_int(H,V)	((BUN) mix_int(*(const unsigned int *) (V)) & (H)->mask)
/* XXX return size_t-sized value for 8-byte oid? */
#define hash_lng(H,V)	((BUN) mix_lng(*(const ulng *) (V)) & (H)->mask)
#ifdef HAVE_HGE
#define hash_hge(H,V)	((BUN) mix_hge(*(const uhge *) (V)) & (H)->mask)
#endif
#if SIZEOF_OID == SIZEOF_INT
#define hash_oid(H,V)	hash_int(H,V)
#else
#define hash_oid(H,V)	hash_lng(H,V)
#endif
#if SIZEOF_WRD == SIZEOF_INT
#define hash_wrd(H,V)	hash_int(H,V)
#else
#define hash_wrd(H,V)	hash_lng(H,V)
#endif

#define hash_flt(H,V)	hash_int(H,V)
#define hash_dbl(H,V)	hash_lng(H,V)

#define HASHget2(h, pcs, prb) ((BUN) ((const BUN2type *) h->Hash)[pcs * (h->mask + 1) + prb])
#define HASHget4(h, pcs, prb) ((BUN) ((const BUN4type *) h->Hash)[pcs * (h->mask + 1) + prb])
#ifdef BUN8
#define HASHget8(h, pcs, prb) ((BUN) ((const BUN8type *) h->Hash)[pcs * (h->mask + 1) + prb])
#define HASHget(h, pcs, prb)				\
	((h)->width == BUN4 ? HASHget4(h, pcs, prb) :	\
	 (h)->width == BUN8 ? HASHget8(h, pcs, prb) :	\
	 HASHget2(h, pcs, prb))
#else
#define HASHget(h, pcs, prb)				\
	((h)->width == BUN4 ? HASHget4(h, pcs, prb) : HASHget2(h, pcs, prb))
#endif


#define HASHgetlink2(h, hb) ((BUN) ((BUN2type *) h->Link)[hb])
#define HASHgetlink4(h, hb) ((BUN) ((BUN4type *) h->Link)[hb])
#ifdef BUN8
#define HASHgetlink8(h, hb) ((BUN) ((BUN8type *) h->Link)[hb])
#define HASHgetlink(h, hb)				\
	((h)->width == BUN4 ? HASHgetlink4(h, hb) :	\
	 (h)->width == BUN8 ? HASHgetlink8(h, hb) :	\
	 HASHgetlink2(h, hb))
#else
#define HASHgetlink(h, hb)				\
	((h)->width == BUN4 ? HASHgetlink4(h, hb) : HASHgetlink2(h, hb))
#endif

/* input parameters:
 * BATiter *bi -- the iterator for the BAT being searched
 * Hash *h -- the pointer to the Hash structure under scrutiny
 * const BUN prb -- the calculated hash value, taking hash mask into account
 * const void *v -- pointer to the value being searched
 * (prb = HASHprobe(h, v))
 *
 * scratch variables:
 * int pcs -- must be a signed type
 * BUN hb
 */
#define HASHloop(bi, h, prb, v, hb, pcs)				\
	for (pcs = h->pieces - 1; pcs >= 0; pcs--)			\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (ATOMcmp(h->type, v, BUNtail(bi, hb)) == 0)
#define HASHlooploc(bi, h, prb, v, hb, pcs)				\
	for (pcs = h->pieces - 1; pcs >= 0; pcs--)			\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (ATOMcmp(h->type, v, BUNtloc(bi, hb)) == 0)
#define HASHloopvar(bi, h, prb, v, hb, pcs)				\
	for (pcs = h->pieces - 1; pcs >= 0; pcs--)			\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (ATOMcmp(h->type, v, BUNtvar(bi, hb)) == 0)
#define HASHloop_str(bi, h, prb, v, hb, pcs)				\
	for (pcs = h->pieces - 1; pcs >= 0; pcs--)			\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (GDK_STREQ(v, BUNtvar(bi, hb)))
#define HASHloop_TYPE(bi, h, prb, v, hb, pcs, TYPE)			\
	for (pcs = h->pieces - 1; pcs >= 0; pcs--)			\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (simple_EQ(v, BUNtloc(bi, hb), TYPE))
#define HASHloop_bte(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, bte)
#define HASHloop_sht(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, sht)
#define HASHloop_int(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, int)
#define HASHloop_lng(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, lng)
#ifdef HAVE_HGE
#define HASHloop_hge(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, hge)
#endif
#define HASHloop_flt(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, flt)
#define HASHloop_dbl(bi, h, prb, v, hb, pcs)	HASHloop_TYPE(bi, h, prb, v, hb, pcs, dbl)

/* input parameters:
 * BATiter *bi -- the iterator for the BAT being searched
 * Hash *h -- the pointer to the Hash structure under scrutiny
 * const BUN prb -- the calculated hash value, taking hash mask into account
 * const void *v -- pointer to the value being searched
 * BUN start -- lowest boundary of region we're searching (inclusive)
 * BUN end -- upper boundary of region we're searching (exclusive)
 * (prb = HASHprobe(h, v))
 *
 * scratch variables:
 * int pcs -- must be a signed type
 * BUN hb
 */
#define HASHloop_bound(bi, h, prb, v, start, end, hb, pcs)		\
	for (pcs = (int) ((end + h->chunk -  1) / h->chunk),		\
		     pcs = pcs >= h->pieces ? h->pieces - 1 : pcs;	\
	     pcs >= (int) (start / h->chunk) || pcs == h->pieces - 1;	\
	     pcs--)							\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (hb >= start && hb < end &&			\
			    ATOMcmp(h->type, v, BUNtail(bi, hb)) == 0)
#define HASHloop_bound_EQ(bi, h, prb, v, start, end, hb, pcs, EQ)	\
	for (pcs = (int) ((end + h->chunk -  1) / h->chunk),		\
		     pcs = pcs >= h->pieces ? h->pieces - 1 : pcs;	\
	     pcs >= (int) (start / h->chunk) || pcs == h->pieces - 1;	\
	     pcs--)							\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (hb >= start && hb < end &&			\
			    EQ(v, BUNtail(bi, hb)))
#define HASHloop_bound_TYPE(bi, h, prb, v, start, end, hb, pcs, TYPE)	\
	for (pcs = (int) ((end + h->chunk -  1) / h->chunk), 		\
		     pcs = pcs >= h->pieces ? h->pieces - 1 : pcs;	\
	     pcs >= (int) (start / h->chunk) || pcs == h->pieces - 1;	\
	     pcs--)							\
		for (hb = HASHget(h, pcs, prb);				\
		     hb != HASHnil(h);					\
		     hb = HASHgetlink(h, hb))				\
			if (hb >= start && hb < end &&			\
			    simple_EQ(v, BUNtloc(bi, hb), TYPE))
