/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "gdk.h"
#include "gdk_analytic.h"
#include "gdk_calc_private.h"

#define ANALYTICAL_DIFF_IMP(TPE)              \
	do {                                      \
		TPE *bp = (TPE*)Tloc(b, 0);           \
		TPE prev = *bp, *end = bp + cnt;      \
		if(np) {                              \
			for(; bp<end; bp++, rb++, np++) { \
				*rb = *np;                    \
				if (*bp != prev) {            \
					*rb = TRUE;               \
					prev = *bp;               \
				}                             \
			}                                 \
		} else {                              \
			for(; bp<end; bp++, rb++) {       \
				if (*bp != prev) {            \
					*rb = TRUE;               \
					prev = *bp;               \
				} else {                      \
					*rb = FALSE;              \
				}                             \
			}                                 \
		}                                     \
	} while(0);

gdk_return
GDKanalyticaldiff(BAT *r, BAT *b, BAT *p, int tpe)
{
	BUN i, cnt = BATcount(b);
	bit *restrict rb = (bit*)Tloc(r, 0), *restrict np = p ? (bit*)Tloc(p, 0) : NULL;

	switch(tpe) {
		case TYPE_bit:
			ANALYTICAL_DIFF_IMP(bit)
			break;
		case TYPE_bte:
			ANALYTICAL_DIFF_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_DIFF_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_DIFF_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_DIFF_IMP(lng)
			break;
#ifdef HAVE_HGE
		case TYPE_hge:
			ANALYTICAL_DIFF_IMP(hge)
			break;
#endif
		case TYPE_flt:
			ANALYTICAL_DIFF_IMP(flt)
			break;
		case TYPE_dbl:
			ANALYTICAL_DIFF_IMP(dbl)
			break;
		default: {
			BATiter it = bat_iterator(b);
			ptr v = BUNtail(it, 0), next;
			int (*atomcmp)(const void *, const void *) = ATOMcompare(tpe);
			if(np) {
				for (i=0; i<cnt; i++, rb++, np++) {
					*rb = *np;
					next = BUNtail(it, i);
					if (atomcmp(v, next) != 0) {
						*rb = TRUE;
						v = next;
					}
				}
			} else {
				for(i=0; i<cnt; i++, rb++) {
					next = BUNtail(it, i);
					if (atomcmp(v, next) != 0) {
						*rb = TRUE;
						v = next;
					} else {
						*rb = FALSE;
					}
				}
			}
		}
	}
	BATsetcount(r, cnt);
	r->tnonil = true;
	r->tnil = false;
	return GDK_SUCCEED;
}

#undef ANALYTICAL_DIFF_IMP

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_ROWS_START(TPE) \
	do {                                               \
		TPE *bs = MIN(pbp + limit, bp);                \
		int curval = 0;                                \
		for(; pbp<bs; pbp++, rb++)                     \
			*rb = curval++;                            \
		for(; pbp<bp; pbp++, rb++)                     \
			*rb = curval;                              \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_ROWS_END(TPE) \
	do {                                             \
		int curval = MIN(ncnt, limit);               \
		TPE *bs = bp - curval;                       \
		curval++;                                    \
		for(; pbp<bs; pbp++, rb++)                   \
			*rb = curval;                            \
		for(; pbp<bp; pbp++, rb++)                   \
			*rb = --curval;                          \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_RANGE_START(TPE) \
	do {                                                \
		TPE *bl = pbp-1, *bs, v, blimit = (TPE) limit;  \
		int curval;                                     \
		for(; pbp<bp; pbp++, rb++) {                    \
			curval = 0;                                 \
			v = *pbp;                                   \
			for(bs=pbp-1; bs>bl; bs--, curval++) {      \
				if (ABSOLUTE(v - *bs) > blimit)         \
					break;                              \
			}                                           \
			*rb = curval;                               \
		}                                               \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_RANGE_END(TPE) \
	do {                                              \
		TPE *bs, v, blimit = (TPE) limit;             \
		int curval;                                   \
		for(; pbp<bp; pbp++, rb++) {                  \
			curval = 1;                               \
			v = *pbp;                                 \
			for(bs=pbp+1; bs<bp; bs++, curval++) {    \
				if (ABSOLUTE(v - *bs) > blimit)       \
					break;                            \
			}                                         \
			*rb = curval;                             \
		}                                             \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_GROUPS_START(TPE) \
	do {                                                 \
		TPE *bl = pbp-1, *bs, v;                         \
		int curval;                                      \
		BUN rlimit;                                      \
		for(; pbp<bp; pbp++, rb++) {                     \
			curval = 0;                                  \
			rlimit = limit;                              \
			v = *pbp;                                    \
			for(bs=pbp-1; bs>bl; bs--, curval++) {       \
				if(v != *bs) {                           \
					if(rlimit == 0)                      \
						break;                           \
					rlimit--;                            \
					v = *bs;                             \
				}                                        \
			}                                            \
			*rb = curval;                                \
		}                                                \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_GROUPS_END(TPE) \
	do {                                               \
		TPE *bs, v;                                    \
		int curval;                                    \
		BUN rlimit;                                    \
		for(; pbp<bp; pbp++, rb++) {                   \
			curval = 1;                                \
			rlimit = limit;                            \
			v = *pbp;                                  \
			for(bs=pbp+1; bs<bp; bs++, curval++) {     \
				if(v != *bs) {                         \
					if(rlimit == 0)                    \
						break;                         \
					rlimit--;                          \
					v = *bs;                           \
				}                                      \
			}                                          \
			*rb = curval;                              \
		}                                              \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_ALL_START(TPE) \
	do {                                              \
		int curval = 0;                               \
		for(; pbp<bp; pbp++, rb++)                    \
			*rb = curval++;                           \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_FIXED_ALL_END(TPE) \
	do {                                            \
		int curval = ncnt + 1;                      \
		for(; pbp<bp; pbp++, rb++)                  \
			*rb = --curval;                         \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(TPE, IMP) \
	do {                                   \
		TPE *pbp, *bp;                     \
		pbp = bp = (TPE*)Tloc(b, 0);       \
		if(start) {                        \
			if(np) {                       \
				nend += cnt;               \
				for(; np<nend; np++) {     \
					if (*np) {             \
						ncnt = (np - pnp); \
						bp += ncnt;        \
						IMP##_START(TPE)   \
						pnp = np;          \
						pbp = bp;          \
					}                      \
				}                          \
				ncnt = (np - pnp);         \
				bp += ncnt;                \
				IMP##_START(TPE)           \
			} else {                       \
				ncnt = cnt;                \
				bp += ncnt;                \
				IMP##_START(TPE)           \
			}                              \
		} else if(np) {                    \
			nend += cnt;                   \
			for(; np<nend; np++) {         \
				if (*np) {                 \
				ncnt = (np - pnp);         \
					bp += ncnt;            \
					IMP##_END(TPE)         \
					pnp = np;              \
					pbp = bp;              \
				}                          \
			}                              \
			ncnt = (np - pnp);             \
			bp += ncnt;                    \
			IMP##_END(TPE)                 \
		} else {                           \
			ncnt = cnt;                    \
			bp += ncnt;                    \
			IMP##_END(TPE)                 \
		}                                  \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ROWS_START \
	do {                                             \
		int curval = 0;                              \
		BUN l = MIN(k + limit, i);                   \
		for(; k<l; k++, rb++)                        \
			*rb = curval++;                          \
		for(; k<i; k++, rb++)                        \
			*rb = curval;                            \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ROWS_END \
	do {                                           \
		int curval = MIN(ncnt, limit);             \
		BUN l = i - curval;                        \
		curval++;                                  \
		for(; k<l; k++, rb++)                      \
			*rb = curval;                          \
		for(; k<i; k++, rb++)                      \
			*rb = --curval;                        \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_RANGE_START \
	do {                                              \
		void *v;                                      \
		int curval, llimit = (int)limit;              \
		BUN j;                                        \
		*rb = 0; /* the first element's window size is hardcoded to avoid overflow in BUN */ \
		rb++;                                         \
		k++;                                          \
		j = k - 1;                                    \
		for(; k<i; k++, rb++) {                       \
			curval = 1;                               \
			v = BUNtail(bpi, k);                      \
			for(BUN l=k-1; l>j; l--, curval++) {      \
				if (ABSOLUTE(atomcmp(v, BUNtail(bpi, l))) > llimit) \
					break;                            \
			}                                         \
			*rb = curval;                             \
		}                                             \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_RANGE_END \
	do {                                            \
		void *v;                                    \
		int curval, llimit = (int)limit;            \
		for(; k<i; k++, rb++) {                     \
			curval = 1;                             \
			v = BUNtail(bpi, k);                    \
			for(BUN l=k+1; l<i; l++, curval++) {    \
				if (ABSOLUTE(atomcmp(v, BUNtail(bpi, l))) > llimit) \
					break;                          \
			}                                       \
			*rb = curval;                           \
		}                                           \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_GROUPS_START \
	do {                                               \
		void *v, *next;                                \
		int curval;                                    \
		BUN j, rlimit;                                 \
		*rb = 0; /* the first element's window size is hardcoded to avoid overflow in BUN */ \
		rb++;                                          \
		k++;                                           \
		j = k - 1;                                     \
		for(; k<i; k++, rb++) {                        \
			curval = 1;                                \
			rlimit = limit;                            \
			v = BUNtail(bpi, k);                       \
			for(BUN l=k-1; l>j; l--, curval++) {       \
				next = BUNtail(bpi, l);                \
				if(atomcmp(v, next)) {                 \
					if(rlimit == 0)                    \
						break;                         \
					rlimit--;                          \
					v = next;                          \
				}                                      \
			}                                          \
			*rb = curval;                              \
		}                                              \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_GROUPS_END \
	do {                                             \
		void *v, *next;                              \
		int curval;                                  \
		BUN rlimit;                                  \
		for(; k<i; k++, rb++) {                      \
			curval = 1;                              \
			rlimit = limit;                          \
			v = BUNtail(bpi, k);                     \
			for(BUN l=k+1; l<i; l++, curval++) {     \
				next = BUNtail(bpi, l);              \
				if(atomcmp(v, next)) {               \
					if(rlimit == 0)                  \
						break;                       \
					rlimit--;                        \
					v = next;                        \
				}                                    \
			}                                        \
			*rb = curval;                            \
		}                                            \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ALL_START \
	do {                                            \
		int curval = 0;                             \
		for(; k<i; k++, rb++)                       \
			*rb = curval++;                         \
	} while(0);

#define ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ALL_END \
	do {                                          \
		int curval = ncnt + 1;                    \
		for(; k<i; k++, rb++)                     \
			*rb = --curval;                       \
	} while(0);

#ifdef HAVE_HGE
#define ANALYTICAL_WINDOW_BOUNDS_LIMIT(FRAME) \
	case TYPE_hge: \
		ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(hge, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
		break;
#else
#define ANALYTICAL_WINDOW_BOUNDS_LIMIT(FRAME)
#endif

#define ANALYTICAL_WINDOW_BOUNDS_BRANCHES(FRAME) \
	do { \
		switch(tpe) { \
			case TYPE_bit: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(bit, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			case TYPE_bte: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(bte, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			case TYPE_sht: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(sht, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			case TYPE_int: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(int, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			case TYPE_lng: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(lng, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			ANALYTICAL_WINDOW_BOUNDS_LIMIT(FRAME) \
			case TYPE_flt: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(flt, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			case TYPE_dbl: \
				ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED(dbl, ANALYTICAL_WINDOW_BOUNDS_FIXED##FRAME) \
				break; \
			default: { \
				if(start) { \
					if (p) { \
						pnp = np = (bit*)Tloc(p, 0); \
						nend = np + cnt; \
						for(; np<nend; np++) { \
							if (*np) { \
								ncnt = (np - pnp); \
								i += ncnt; \
								ANALYTICAL_WINDOW_BOUNDS_VARSIZED##FRAME##_START \
								pnp = np; \
							} \
						} \
						ncnt = (np - pnp); \
						i += ncnt; \
						ANALYTICAL_WINDOW_BOUNDS_VARSIZED##FRAME##_START \
					} else { \
						ncnt = cnt; \
						i += ncnt; \
						ANALYTICAL_WINDOW_BOUNDS_VARSIZED##FRAME##_START \
					} \
				} else if (p) { \
					pnp = np = (bit*)Tloc(p, 0); \
					nend = np + cnt; \
					for(; np<nend; np++) { \
						if (*np) { \
							ncnt = (np - pnp); \
							i += ncnt; \
							ANALYTICAL_WINDOW_BOUNDS_VARSIZED##FRAME##_END \
							pnp = np; \
						} \
					} \
					ncnt = (np - pnp); \
					i += ncnt; \
					ANALYTICAL_WINDOW_BOUNDS_VARSIZED##FRAME##_END \
				} else { \
					ncnt = cnt; \
					i += ncnt; \
					ANALYTICAL_WINDOW_BOUNDS_VARSIZED##FRAME##_END \
				} \
			} \
		} \
	} while(0);

gdk_return
GDKanalyticalwindowbounds(BAT *r, BAT *b, BAT *p, int tpe, int unit, BUN limit, bool start)
{
	BUN i = 0, k = 0, ncnt, cnt = BATcount(b);
	int *restrict rb = (int*)Tloc(r, 0);
	bit *np = p ? (bit*)Tloc(p, 0) : NULL, *pnp = np, *nend = np;
	BATiter bpi = bat_iterator(b);
	int (*atomcmp)(const void *, const void *) = ATOMcompare(tpe);

	if(unit == 0) {
		ANALYTICAL_WINDOW_BOUNDS_BRANCHES(_ROWS)
	} else if(unit == 1) {
		ANALYTICAL_WINDOW_BOUNDS_BRANCHES(_RANGE)
	} else if(unit == 2) {
		ANALYTICAL_WINDOW_BOUNDS_BRANCHES(_GROUPS)
	} else if(unit == 3) {
		ANALYTICAL_WINDOW_BOUNDS_BRANCHES(_ALL)
	} else {
		assert(0);
	}
	BATsetcount(r, cnt);
	r->tnonil = true;
	r->tnil = false;
	return GDK_SUCCEED;
}

#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_ROWS_START
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_ROWS_END
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_RANGE_START
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_RANGE_END
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_GROUPS_START
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_GROUPS_END
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_ALL_START
#undef ANALYTICAL_WINDOW_BOUNDS_FIXED_ALL_END
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ROWS_START
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ROWS_END
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_RANGE_START
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_RANGE_END
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_GROUPS_START
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_GROUPS_END
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ALL_START
#undef ANALYTICAL_WINDOW_BOUNDS_VARSIZED_ALL_END
#undef ANALYTICAL_WINDOW_BOUNDS_CALC_FIXED
#undef ANALYTICAL_WINDOW_BOUNDS_BRANCHES
#undef ANALYTICAL_WINDOW_BOUNDS_LIMIT

#define NTILE_CALC(TPE)               \
	do {                              \
		if((BUN)val >= ncnt) {        \
			i = 1;                    \
			for(; rb<rp; i++, rb++)   \
				*rb = i;              \
		} else if(ncnt % val == 0) {  \
			buckets = ncnt / val;     \
			for(; rb<rp; i++, rb++) { \
				if(i == buckets) {    \
					j++;              \
					i = 0;            \
				}                     \
				*rb = j;              \
			}                         \
		} else {                      \
			buckets = ncnt / val;     \
			for(; rb<rp; i++, rb++) { \
				*rb = j;              \
				if(i == buckets) {    \
					j++;              \
					i = 0;            \
				}                     \
			}                         \
		}                             \
	} while(0);

#define ANALYTICAL_NTILE_IMP(TPE)            \
	do {                                     \
		TPE i = 0, j = 1, *rp, *rb, buckets; \
		TPE val =  *(TPE *)ntile;            \
		rb = rp = (TPE*)Tloc(r, 0);          \
		if(is_##TPE##_nil(val)) {            \
			TPE *end = rp + cnt;             \
			has_nils = true;                 \
			for(; rp<end; rp++)              \
				*rp = TPE##_nil;             \
		} else if(p) {                       \
			pnp = np = (bit*)Tloc(p, 0);     \
			end = np + cnt;                  \
			for(; np<end; np++) {            \
				if (*np) {                   \
					i = 0;                   \
					j = 1;                   \
					ncnt = np - pnp;         \
					rp += ncnt;              \
					NTILE_CALC(TPE)          \
					pnp = np;                \
				}                            \
			}                                \
			i = 0;                           \
			j = 1;                           \
			ncnt = np - pnp;                 \
			rp += ncnt;                      \
			NTILE_CALC(TPE)                  \
		} else {                             \
			rp += cnt;                       \
			NTILE_CALC(TPE)                  \
		}                                    \
		goto finish;                         \
	} while(0);

gdk_return
GDKanalyticalntile(BAT *r, BAT *b, BAT *p, BAT *o, int tpe, const void* restrict ntile)
{
	BUN cnt = BATcount(b), ncnt = cnt;
	bit *np, *pnp, *end;
	bool has_nils = false;
	gdk_return gdk_res = GDK_SUCCEED;

	assert(ntile);

	(void) o;
	switch (tpe) {
		case TYPE_bte:
			ANALYTICAL_NTILE_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_NTILE_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_NTILE_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_NTILE_IMP(lng)
			break;
#ifdef HAVE_HGE
		case TYPE_hge:
			ANALYTICAL_NTILE_IMP(hge)
			break;
#endif
		default:
			goto nosupport;
	}
nosupport:
	GDKerror("ntile: type %s not supported.\n", ATOMname(tpe));
	return GDK_FAIL;
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return gdk_res;
}

#undef ANALYTICAL_NTILE_IMP
#undef NTILE_CALC

#define FIRST_CALC(TPE)            \
	do {                           \
		for (;rb < rp; rb++)       \
			*rb = curval;          \
		if(is_##TPE##_nil(curval)) \
			has_nils = true;       \
	} while(0);

#define ANALYTICAL_FIRST_IMP(TPE)           \
	do {                                    \
		TPE *rp, *rb, *restrict bp, curval; \
		rb = rp = (TPE*)Tloc(r, 0);         \
		bp = (TPE*)Tloc(b, 0);              \
		curval = *bp;                       \
		if (p) {                            \
			pnp = np = (bit*)Tloc(p, 0);    \
			end = np + cnt;                 \
			for(; np<end; np++) {           \
				if (*np) {                  \
					ncnt = (np - pnp);      \
					rp += ncnt;             \
					bp += ncnt;             \
					FIRST_CALC(TPE)         \
					curval = *bp;           \
					pnp = np;               \
				}                           \
			}                               \
			ncnt = (np - pnp);              \
			rp += ncnt;                     \
			bp += ncnt;                     \
			FIRST_CALC(TPE)                 \
		} else {                            \
			rp += cnt;                      \
			FIRST_CALC(TPE)                 \
		}                                   \
	} while(0);

#define ANALYTICAL_FIRST_OTHERS                                         \
	do {                                                                \
		curval = BUNtail(bpi, j);                                       \
		if((*atomcmp)(curval, nil) == 0)                                \
			has_nils = true;                                            \
		for (;j < i; j++) {                                             \
			if ((gdk_res = BUNappend(r, curval, false)) != GDK_SUCCEED) \
				goto finish;                                            \
		}                                                               \
	} while(0);

gdk_return
GDKanalyticalfirst(BAT *r, BAT *b, BAT *p, BAT *o, int tpe)
{
	int (*atomcmp)(const void *, const void *);
	const void* restrict nil;
	bool has_nils = false;
	BUN i = 0, j = 0, ncnt, cnt = BATcount(b);
	bit *np, *pnp, *end;
	gdk_return gdk_res = GDK_SUCCEED;

	(void) o;
	switch(tpe) {
		case TYPE_bit:
			ANALYTICAL_FIRST_IMP(bit)
			break;
		case TYPE_bte:
			ANALYTICAL_FIRST_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_FIRST_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_FIRST_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_FIRST_IMP(lng)
			break;
#ifdef HAVE_HUGE
		case TYPE_hge:
			ANALYTICAL_FIRST_IMP(hge)
			break;
#endif
		case TYPE_flt:
			ANALYTICAL_FIRST_IMP(flt)
			break;
		case TYPE_dbl:
			ANALYTICAL_FIRST_IMP(dbl)
			break;
		default: {
			BATiter bpi = bat_iterator(b);
			void *restrict curval;
			nil = ATOMnilptr(tpe);
			atomcmp = ATOMcompare(tpe);
			if (p) {
				pnp = np = (bit*)Tloc(p, 0);
				end = np + cnt;
				for(; np<end; np++) {
					if (*np) {
						i += (np - pnp);
						ANALYTICAL_FIRST_OTHERS
						pnp = np;
					}
				}
				i += (np - pnp);
				ANALYTICAL_FIRST_OTHERS
			} else { /* single value, ie no ordering */
				i += cnt;
				ANALYTICAL_FIRST_OTHERS
			}
		}
	}
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return gdk_res;
}

#undef ANALYTICAL_FIRST_IMP
#undef FIRST_CALC
#undef ANALYTICAL_FIRST_OTHERS

#define LAST_CALC(TPE)             \
	do {                           \
		curval = *(bp - 1);        \
		if(is_##TPE##_nil(curval)) \
			has_nils = true;       \
		for (;rb < rp; rb++)       \
			*rb = curval;          \
	} while(0);

#define ANALYTICAL_LAST_IMP(TPE)            \
	do {                                    \
		TPE *rp, *rb, *restrict bp, curval; \
		rb = rp = (TPE*)Tloc(r, 0);         \
		bp = (TPE*)Tloc(b, 0);              \
		if (p) {                            \
			pnp = np = (bit*)Tloc(p, 0);    \
			end = np + cnt;                 \
			for(; np<end; np++) {           \
				if (*np) {                  \
					ncnt = (np - pnp);      \
					rp += ncnt;             \
					bp += ncnt;             \
					LAST_CALC(TPE)          \
					pnp = np;               \
				}                           \
			}                               \
			ncnt = (np - pnp);              \
			rp += ncnt;                     \
			bp += ncnt;                     \
			LAST_CALC(TPE)                  \
		} else {                            \
			rp += cnt;                      \
			bp += cnt;                      \
			LAST_CALC(TPE)                  \
		}                                   \
	} while(0);

#define ANALYTICAL_LAST_OTHERS                                          \
	do {                                                                \
		curval = BUNtail(bpi, i - 1);                                   \
		if((*atomcmp)(curval, nil) == 0)                                \
			has_nils = true;                                            \
		for (;j < i; j++) {                                             \
			if ((gdk_res = BUNappend(r, curval, false)) != GDK_SUCCEED) \
				goto finish;                                            \
		}                                                               \
	} while(0);

gdk_return
GDKanalyticallast(BAT *r, BAT *b, BAT *p, BAT *o, int tpe)
{
	int (*atomcmp)(const void *, const void *);
	const void* restrict nil;
	bool has_nils = false;
	BUN i = 0, j = 0, ncnt, cnt = BATcount(b);
	bit *np, *pnp, *end;
	gdk_return gdk_res = GDK_SUCCEED;

	(void) o;
	switch(tpe) {
		case TYPE_bit:
			ANALYTICAL_LAST_IMP(bit)
			break;
		case TYPE_bte:
			ANALYTICAL_LAST_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_LAST_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_LAST_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_LAST_IMP(lng)
			break;
#ifdef HAVE_HUGE
		case TYPE_hge:
			ANALYTICAL_LAST_IMP(hge)
			break;
#endif
		case TYPE_flt:
			ANALYTICAL_LAST_IMP(flt)
			break;
		case TYPE_dbl:
			ANALYTICAL_LAST_IMP(dbl)
			break;
		default: {
			BATiter bpi = bat_iterator(b);
			void *restrict curval;
			nil = ATOMnilptr(tpe);
			atomcmp = ATOMcompare(tpe);
			if (p) {
				pnp = np = (bit*)Tloc(p, 0);
				end = np + cnt;
				for(; np<end; np++) {
					if (*np) {
						i += (np - pnp);
						ANALYTICAL_LAST_OTHERS
						pnp = np;
					}
				}
				i += (np - pnp);
				ANALYTICAL_LAST_OTHERS
			} else { /* single value, ie no ordering */
				i += cnt;
				ANALYTICAL_LAST_OTHERS
			}
		}
	}
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return gdk_res;
}

#undef ANALYTICAL_LAST_IMP
#undef LAST_CALC
#undef ANALYTICAL_LAST_OTHERS

#define NTHVALUE_CALC(TPE)         \
	do {                           \
		if(nth > (BUN) (bp - pbp)) \
			curval = TPE##_nil;    \
		else                       \
			curval = *(pbp + nth); \
		if(is_##TPE##_nil(curval)) \
			has_nils = true;       \
		for(; rb<rp; rb++)         \
			*rb = curval;          \
	} while(0);

#define ANALYTICAL_NTHVALUE_IMP(TPE)     \
	do {                                 \
		TPE *rp, *rb, *pbp, *bp, curval; \
		pbp = bp = (TPE*)Tloc(b, 0);     \
		rb = rp = (TPE*)Tloc(r, 0);      \
		if(nth == BUN_NONE) {            \
			TPE* rend = rp + cnt;        \
			has_nils = true;             \
			for(; rp<rend; rp++)         \
				*rp = TPE##_nil;         \
		} else if(p) {                   \
			pnp = np = (bit*)Tloc(p, 0); \
			end = np + cnt;              \
			for(; np<end; np++) {        \
				if (*np) {               \
					ncnt = (np - pnp);   \
					rp += ncnt;          \
					bp += ncnt;          \
					NTHVALUE_CALC(TPE)   \
					pbp = bp;            \
					pnp = np;            \
				}                        \
			}                            \
			ncnt = (np - pnp);           \
			rp += ncnt;                  \
			bp += ncnt;                  \
			NTHVALUE_CALC(TPE)           \
		} else {                         \
			rp += cnt;                   \
			bp += cnt;                   \
			NTHVALUE_CALC(TPE)           \
		}                                \
		goto finish;                     \
	} while(0);

#define ANALYTICAL_NTHVALUE_OTHERS                                      \
	do {                                                                \
		if(nth > (i - j))                                               \
			curval = nil;                                               \
		else                                                            \
			curval = BUNtail(bpi, nth);                                 \
		if((*atomcmp)(curval, nil) == 0)                                \
			has_nils = true;                                            \
		for (;j < i; j++) {                                             \
			if ((gdk_res = BUNappend(r, curval, false)) != GDK_SUCCEED) \
				goto finish;                                            \
		}                                                               \
	} while(0);

gdk_return
GDKanalyticalnthvalue(BAT *r, BAT *b, BAT *p, BAT *o, BUN nth, int tpe)
{
	int (*atomcmp)(const void *, const void *);
	const void* restrict nil;
	BUN i = 0, j = 0, ncnt, cnt = BATcount(b);
	bit *np, *pnp, *end;
	gdk_return gdk_res = GDK_SUCCEED;
	bool has_nils = false;

	(void) o;
	switch (tpe) {
		case TYPE_bte:
			ANALYTICAL_NTHVALUE_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_NTHVALUE_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_NTHVALUE_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_NTHVALUE_IMP(lng)
			break;
#ifdef HAVE_HGE
		case TYPE_hge:
			ANALYTICAL_NTHVALUE_IMP(hge)
			break;
#endif
		case TYPE_flt:
			ANALYTICAL_NTHVALUE_IMP(flt)
			break;
		case TYPE_dbl:
			ANALYTICAL_NTHVALUE_IMP(dbl)
			break;
		default: {
			BATiter bpi = bat_iterator(b);
			const void *restrict curval;
			nil = ATOMnilptr(tpe);
			atomcmp = ATOMcompare(tpe);
			if(nth == BUN_NONE) {
				has_nils = true;
				for(i=0; i<cnt; i++) {
					if ((gdk_res = BUNappend(r, nil, false)) != GDK_SUCCEED)
						goto finish;
				}
			} else if (p) {
				pnp = np = (bit*)Tloc(p, 0);
				end = np + cnt;
				for(; np<end; np++) {
					if (*np) {
						i += (np - pnp);
						ANALYTICAL_NTHVALUE_OTHERS
						pnp = np;
					}
				}
				i += (np - pnp);
				ANALYTICAL_NTHVALUE_OTHERS
			} else { /* single value, ie no ordering */
				i += cnt;
				ANALYTICAL_NTHVALUE_OTHERS
			}
		}
	}
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return gdk_res;
}

#undef ANALYTICAL_NTHVALUE_IMP
#undef NTHVALUE_CALC
#undef ANALYTICAL_NTHVALUE_OTHERS

#define ANALYTICAL_LAG_CALC(TPE)            \
	do {                                    \
		for(i=0; i<lag && rb<rp; i++, rb++) \
			*rb = def;                      \
		if(lag > 0 && is_##TPE##_nil(def))  \
			has_nils = true;                \
		for(;rb<rp; rb++, bp++) {           \
			next = *bp;                     \
			*rb = next;                     \
			if(is_##TPE##_nil(next))        \
				has_nils = true;            \
		}                                   \
	} while(0);

#define ANALYTICAL_LAG_IMP(TPE)                   \
	do {                                          \
		TPE *rp, *rb, *bp, *rend,                 \
			def = *((TPE *) default_value), next; \
		bp = (TPE*)Tloc(b, 0);                    \
		rb = rp = (TPE*)Tloc(r, 0);               \
		rend = rb + cnt;                          \
		if(lag == BUN_NONE) {                     \
			has_nils = true;                      \
			for(; rb<rend; rb++)                  \
				*rb = TPE##_nil;                  \
		} else if(p) {                            \
			pnp = np = (bit*)Tloc(p, 0);          \
			end = np + cnt;                       \
			for(; np<end; np++) {                 \
				if (*np) {                        \
					ncnt = (np - pnp);            \
					rp += ncnt;                   \
					ANALYTICAL_LAG_CALC(TPE)      \
					bp += (lag < ncnt) ? lag : 0; \
					pnp = np;                     \
				}                                 \
			}                                     \
			rp += (np - pnp);                     \
			ANALYTICAL_LAG_CALC(TPE)              \
		} else {                                  \
			rp += cnt;                            \
			ANALYTICAL_LAG_CALC(TPE)              \
		}                                         \
		goto finish;                              \
	} while(0);

#define ANALYTICAL_LAG_OTHERS                                                  \
	do {                                                                       \
		for(i=0; i<lag && k<j; i++, k++) {                                     \
			if ((gdk_res = BUNappend(r, default_value, false)) != GDK_SUCCEED) \
				goto finish;                                                   \
		}                                                                      \
		if(lag > 0 && (*atomcmp)(default_value, nil) == 0)                     \
			has_nils = true;                                                   \
		for(l=k-lag; k<j; k++, l++) {                                          \
			curval = BUNtail(bpi, l);                                          \
			if ((gdk_res = BUNappend(r, curval, false)) != GDK_SUCCEED)        \
				goto finish;                                                   \
			if((*atomcmp)(curval, nil) == 0)                                   \
				has_nils = true;                                               \
		}                                                                      \
	} while (0);

gdk_return
GDKanalyticallag(BAT *r, BAT *b, BAT *p, BAT *o, BUN lag, const void* restrict default_value, int tpe)
{
	int (*atomcmp)(const void *, const void *);
	const void *restrict nil;
	BUN i = 0, j = 0, k = 0, l = 0, ncnt, cnt = BATcount(b);
	bit *np, *pnp, *end;
	gdk_return gdk_res = GDK_SUCCEED;
	bool has_nils = false;

	assert(default_value);

	(void) o;
	switch (tpe) {
		case TYPE_bte:
			ANALYTICAL_LAG_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_LAG_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_LAG_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_LAG_IMP(lng)
			break;
#ifdef HAVE_HGE
		case TYPE_hge:
			ANALYTICAL_LAG_IMP(hge)
			break;
#endif
		case TYPE_flt:
			ANALYTICAL_LAG_IMP(flt)
			break;
		case TYPE_dbl:
			ANALYTICAL_LAG_IMP(dbl)
			break;
		default: {
			BATiter bpi = bat_iterator(b);
			const void *restrict curval;
			nil = ATOMnilptr(tpe);
			atomcmp = ATOMcompare(tpe);
			if(lag == BUN_NONE) {
				has_nils = true;
				for (j=0;j < cnt; j++) {
					if ((gdk_res = BUNappend(r, nil, false)) != GDK_SUCCEED)
						goto finish;
				}
			} else if(p) {
				pnp = np = (bit*)Tloc(p, 0);
				end = np + cnt;
				for(; np<end; np++) {
					if (*np) {
						j += (np - pnp);
						ANALYTICAL_LAG_OTHERS
						pnp = np;
					}
				}
				j += (np - pnp);
				ANALYTICAL_LAG_OTHERS
			} else {
				j += cnt;
				ANALYTICAL_LAG_OTHERS
			}
		}
	}
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return gdk_res;
}

#undef ANALYTICAL_LAG_IMP
#undef ANALYTICAL_LAG_CALC
#undef ANALYTICAL_LAG_OTHERS

#define LEAD_CALC(TPE)                       \
	do {                                     \
		if(lead < ncnt) {                    \
			bp += lead;                      \
			l = ncnt - lead;                 \
			for(i=0; i<l; i++, rb++, bp++) { \
				next = *bp;                  \
				*rb = next;                  \
				if(is_##TPE##_nil(next))     \
					has_nils = true;         \
			}                                \
		} else {                             \
			bp += ncnt;                      \
		}                                    \
		for(;rb<rp; rb++)                    \
			*rb = def;                       \
		if(lead > 0 && is_##TPE##_nil(def))  \
			has_nils = true;                 \
	} while(0);

#define ANALYTICAL_LEAD_IMP(TPE)                  \
	do {                                          \
		TPE *rp, *rb, *bp, *rend,                 \
			def = *((TPE *) default_value), next; \
		bp = (TPE*)Tloc(b, 0);                    \
		rb = rp = (TPE*)Tloc(r, 0);               \
		rend = rb + cnt;                          \
		if(lead == BUN_NONE) {                    \
			has_nils = true;                      \
			for(; rb<rend; rb++)                  \
				*rb = TPE##_nil;                  \
		} else if(p) {                            \
			pnp = np = (bit*)Tloc(p, 0);          \
			end = np + cnt;                       \
			for(; np<end; np++) {                 \
				if (*np) {                        \
					ncnt = (np - pnp);            \
					rp += ncnt;                   \
					LEAD_CALC(TPE)                \
					pnp = np;                     \
				}                                 \
			}                                     \
			ncnt = (np - pnp);                    \
			rp += ncnt;                           \
			LEAD_CALC(TPE)                        \
		} else {                                  \
			ncnt = cnt;                           \
			rp += ncnt;                           \
			LEAD_CALC(TPE)                        \
		}                                         \
		goto finish;                              \
	} while(0);

#define ANALYTICAL_LEAD_OTHERS                                                 \
	do {                                                                       \
		j += ncnt;                                                             \
		if(lead < ncnt) {                                                      \
			m = ncnt - lead;                                                   \
			for(i=0,n=k+lead; i<m; i++, n++) {                                 \
				curval = BUNtail(bpi, n);                                      \
				if ((gdk_res = BUNappend(r, curval, false)) != GDK_SUCCEED)    \
					goto finish;                                               \
				if((*atomcmp)(curval, nil) == 0)                               \
					has_nils = true;                                           \
			}                                                                  \
			k += i;                                                            \
		}                                                                      \
		for(; k<j; k++) {                                                      \
			if ((gdk_res = BUNappend(r, default_value, false)) != GDK_SUCCEED) \
				goto finish;                                                   \
		}                                                                      \
		if(lead > 0 && (*atomcmp)(default_value, nil) == 0)                    \
			has_nils = true;                                                   \
	} while(0);

gdk_return
GDKanalyticallead(BAT *r, BAT *b, BAT *p, BAT *o, BUN lead, const void* restrict default_value, int tpe)
{
	int (*atomcmp)(const void *, const void *);
	const void* restrict nil;
	BUN i = 0, j = 0, k = 0, l = 0, ncnt, cnt = BATcount(b);
	bit *np, *pnp, *end;
	gdk_return gdk_res = GDK_SUCCEED;
	bool has_nils = false;

	assert(default_value);

	(void) o;
	switch (tpe) {
		case TYPE_bte:
			ANALYTICAL_LEAD_IMP(bte)
			break;
		case TYPE_sht:
			ANALYTICAL_LEAD_IMP(sht)
			break;
		case TYPE_int:
			ANALYTICAL_LEAD_IMP(int)
			break;
		case TYPE_lng:
			ANALYTICAL_LEAD_IMP(lng)
			break;
#ifdef HAVE_HGE
		case TYPE_hge:
			ANALYTICAL_LEAD_IMP(hge)
			break;
#endif
		case TYPE_flt:
			ANALYTICAL_LEAD_IMP(flt)
			break;
		case TYPE_dbl:
			ANALYTICAL_LEAD_IMP(dbl)
			break;
		default: {
			BUN m = 0, n = 0;
			BATiter bpi = bat_iterator(b);
			const void *restrict curval;
			nil = ATOMnilptr(tpe);
			atomcmp = ATOMcompare(tpe);
			if(lead == BUN_NONE) {
				has_nils = true;
				for (j=0;j < cnt; j++) {
					if ((gdk_res = BUNappend(r, nil, false)) != GDK_SUCCEED)
						goto finish;
				}
			} else if(p) {
				pnp = np = (bit*)Tloc(p, 0);
				end = np + cnt;
				for(; np<end; np++) {
					if (*np) {
						ncnt = (np - pnp);
						ANALYTICAL_LEAD_OTHERS
						pnp = np;
					}
				}
				ncnt = (np - pnp);
				ANALYTICAL_LEAD_OTHERS
			} else {
				ncnt = cnt;
				ANALYTICAL_LEAD_OTHERS
			}
		}
	}
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return gdk_res;
}

#undef ANALYTICAL_LEAD_IMP
#undef LEAD_CALC
#undef ANALYTICAL_LEAD_OTHERS

#define ANALYTICAL_MIN_MAX_CALC(TPE, OP)        \
	do {                                        \
		TPE *pbp, *bp, *bs, *be, v, curval, *restrict rb; \
		pbp = bp = (TPE*)Tloc(b, 0);            \
		rb = (TPE*)Tloc(r, 0);                  \
		bp += cnt;                              \
		for(; pbp<bp; pbp++, i++, rb++) {       \
			bs = pbp - start[i];                \
			be = pbp + end[i];                  \
			curval = *bs;                       \
			bs++;                               \
			for(; bs<be; bs++) {                \
				v = *bs;                        \
				if(!is_##TPE##_nil(v)) {        \
					if(is_##TPE##_nil(curval))  \
						curval = v;             \
					else                        \
						curval = OP(v, curval); \
				}                               \
			}                                   \
			*rb = curval;                       \
			if(is_##TPE##_nil(curval))          \
				has_nils = true;                \
			curval = TPE##_nil;                 \
		}                                       \
	} while(0);

#ifdef HAVE_HUGE
#define ANALYTICAL_MIN_MAX_LIMIT(OP) \
	case TYPE_hge: \
		ANALYTICAL_MIN_MAX_CALC(hge, OP) \
	break;
#else
#define ANALYTICAL_MIN_MAX_LIMIT(OP)
#endif

#define ANALYTICAL_MIN_MAX(OP, IMP, SIGN_OP) \
gdk_return \
GDKanalytical##OP(BAT *r, BAT *b, BAT *s, BAT *e, int tpe) \
{ \
	bool has_nils = false; \
	BUN i = 0, j = 0, l = 0, cnt = BATcount(b); \
	int *restrict start, *restrict end; \
	gdk_return gdk_res = GDK_SUCCEED; \
 \
	assert(s && e); \
	start = (int*)Tloc(s, 0); \
	end = (int*)Tloc(e, 0); \
 \
	switch(tpe) { \
		case TYPE_bit: \
			ANALYTICAL_MIN_MAX_CALC(bit, IMP) \
			break; \
		case TYPE_bte: \
			ANALYTICAL_MIN_MAX_CALC(bte, IMP) \
			break; \
		case TYPE_sht: \
			ANALYTICAL_MIN_MAX_CALC(sht, IMP) \
			break; \
		case TYPE_int: \
			ANALYTICAL_MIN_MAX_CALC(int, IMP) \
			break; \
		case TYPE_lng: \
			ANALYTICAL_MIN_MAX_CALC(lng, IMP) \
			break; \
		ANALYTICAL_MIN_MAX_LIMIT(IMP) \
		case TYPE_flt: \
			ANALYTICAL_MIN_MAX_CALC(flt, IMP) \
			break; \
		case TYPE_dbl: \
			ANALYTICAL_MIN_MAX_CALC(dbl, IMP) \
			break; \
		default: { \
			BATiter bpi = bat_iterator(b); \
			const void *nil = ATOMnilptr(tpe); \
			int (*atomcmp)(const void *, const void *) = ATOMcompare(tpe); \
			void *curval; \
			for(; i<cnt; i++) { \
				j = i - start[i]; \
				l = i + end[i]; \
				curval = (void *)nil; \
				for (;j < l; j++) { \
					void *next = BUNtail(bpi, j); \
					if((*atomcmp)(next, nil) != 0) { \
						if((*atomcmp)(curval, nil) == 0) \
							curval = next; \
						else \
							curval = atomcmp(next, curval) SIGN_OP 0 ? curval : next; \
					} \
				} \
				if ((gdk_res = BUNappend(r, curval, false)) != GDK_SUCCEED) \
					goto finish; \
				if((*atomcmp)(curval, nil) == 0) \
					has_nils = true; \
			} \
		} \
	} \
finish: \
	BATsetcount(r, cnt); \
	r->tnonil = !has_nils; \
	r->tnil = has_nils; \
	return gdk_res; \
}

ANALYTICAL_MIN_MAX(min, MIN, >)
ANALYTICAL_MIN_MAX(max, MAX, <)

#undef ANALYTICAL_MIN_MAX_CALC
#undef ANALYTICAL_MIN_MAX_LIMIT
#undef ANALYTICAL_MIN_MAX

#define ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(TPE) \
	do {                                            \
		TPE *pbp, *bp;                              \
		pbp = bp = (TPE*)Tloc(b, 0);                \
		bp += cnt;                                  \
		TPE *bs, *be;                               \
		for(; pbp<bp; pbp++, i++, rb++) {           \
			bs = pbp - start[i];                    \
			be = pbp + end[i];                      \
			for(; bs<be; bs++)                      \
				curval += !is_##TPE##_nil(*bs);     \
			*rb = curval;                           \
			curval = 0;                             \
		}                                           \
	} while(0);

#define ANALYTICAL_COUNT_NO_NIL_STR_IMP(TPE_CAST, OFFSET)                 \
	do {                                                                  \
		for(; i<cnt; i++, rb++) {                                         \
			j = i - start[i];                                             \
			l = i + end[i];                                               \
			for(; j<l; j++)                                               \
				curval += base[(var_t) ((TPE_CAST) bp) OFFSET] != '\200'; \
			*rb = curval;                                                 \
			curval = 0;                                                   \
		}                                                                 \
	} while(0);

gdk_return
GDKanalyticalcount(BAT *r, BAT *b, BAT *s, BAT *e, const bit* restrict ignore_nils, int tpe)
{
	BUN i = 0, j = 0, l = 0, cnt = BATcount(b);
	int *restrict start, *restrict end;
	gdk_return gdk_res = GDK_SUCCEED;
	lng *restrict rb = (lng*)Tloc(r, 0), curval = 0;

	assert(s && e && ignore_nils);
	start = (int*)Tloc(s, 0);
	end = (int*)Tloc(e, 0);

	if(!*ignore_nils || b->T.nonil) {
		for(; i<cnt; i++, rb++)
			*rb = (start[i] + end[i]);
	} else {
		switch (tpe) {
			case TYPE_bit:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(bit)
				break;
			case TYPE_bte:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(bte)
				break;
			case TYPE_sht:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(sht)
				break;
			case TYPE_int:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(int)
				break;
			case TYPE_lng:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(lng)
				break;
#ifdef HAVE_HGE
			case TYPE_hge:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(hge)
				break;
#endif
			case TYPE_flt:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(flt)
				break;
			case TYPE_dbl:
				ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP(dbl)
				break;
			case TYPE_str: {
				const char *restrict base = b->tvheap->base;
				const void *restrict bp = Tloc(b, 0);
				switch (b->twidth) {
					case 1:
						ANALYTICAL_COUNT_NO_NIL_STR_IMP(const unsigned char *, [j] + GDK_VAROFFSET)
						break;
					case 2:
						ANALYTICAL_COUNT_NO_NIL_STR_IMP(const unsigned short *, [j] + GDK_VAROFFSET)
						break;
#if SIZEOF_VAR_T != SIZEOF_INT
					case 4:
						ANALYTICAL_COUNT_NO_NIL_STR_IMP(const unsigned int *, [j])
						break;
#endif
					default:
						ANALYTICAL_COUNT_NO_NIL_STR_IMP(const var_t *, [j])
						break;
				}
				break;
			}
			default: {
				const void *restrict nil = ATOMnilptr(tpe);
				int (*cmp)(const void *, const void *) = ATOMcompare(tpe);
				if (b->tvarsized) {
					const char *restrict base = b->tvheap->base;
					const void *restrict bp = Tloc(b, 0);
					for(; i<cnt; i++, rb++) {
						j = i - start[i];
						l = i + end[i];
						for(; j<l; j++)
							curval += (*cmp)(nil, base + ((const var_t *) bp)[j]) != 0;
						*rb = curval;
						curval = 0;
					}
				} else {
					for(; i<cnt; i++, rb++) {
						j = i - start[i];
						l = i + end[i];
						for(; j<l; j++)
							curval += (*cmp)(Tloc(b, j), nil) != 0;
						*rb = curval;
						curval = 0;
					}
				}
			}
		}
	}
	BATsetcount(r, cnt);
	r->tnonil = true;
	r->tnil = false;
	return gdk_res;
}

#undef ANALYTICAL_COUNT_NO_NIL_FIXED_SIZE_IMP
#undef ANALYTICAL_COUNT_NO_NIL_STR_IMP

#define ANALYTICAL_SUM_IMP_NUM(TPE1, TPE2)      \
	do {                                        \
		TPE1 *bs, *be, v;                       \
		for(; pbp<bp; pbp++, i++, rb++) {       \
			bs = pbp - start[i];                \
			be = pbp + end[i];                  \
			for(; bs<be; bs++) {                \
				v = *bs;                        \
				if (!is_##TPE1##_nil(v)) {      \
					if(is_##TPE2##_nil(curval)) \
						curval = (TPE2) v;      \
					else                        \
						ADD_WITH_CHECK(TPE1, v, TPE2, curval, TPE2, curval, GDK_##TPE2##_max, goto calc_overflow); \
				}                               \
			}                                   \
			*rb = curval;                       \
			if(is_##TPE2##_nil(curval))         \
				has_nils = true;                \
			else                                \
				curval = TPE2##_nil;            \
		}                                       \
	} while(0);

#define ANALYTICAL_SUM_IMP_FP(TPE1, TPE2)       \
	do {                                        \
		TPE1 *bs, *be;                          \
		BUN parcel;                             \
		for(; pbp<bp; pbp++, i++, rb++) {       \
			bs = pbp - start[i];                \
			be = pbp + end[i];                  \
			parcel = (be - bs);                 \
			if(dofsum(bs, 0, 0, parcel, &curval, 1, TYPE_##TPE1, TYPE_##TPE2, NULL, NULL, NULL, 0, 0, true, false, \
				  	  true, "GDKanalyticalsum") == BUN_NONE) { \
				goto bailout;                   \
			}                                   \
			*rb = curval;                       \
			if(is_##TPE2##_nil(curval))         \
				has_nils = true;                \
			else                                \
				curval = TPE2##_nil;            \
		}                                       \
	} while(0);

#define ANALYTICAL_SUM_CALC(TPE1, TPE2, IMP)    \
	do {                                        \
		TPE1 *pbp, *bp;                         \
		TPE2 *restrict rb, curval = TPE2##_nil; \
		pbp = bp = (TPE1*)Tloc(b, 0);           \
		rb = (TPE2*)Tloc(r, 0);                 \
		bp += cnt;                              \
		IMP(TPE1, TPE2)                         \
		goto finish;                            \
	} while(0);

gdk_return
GDKanalyticalsum(BAT *r, BAT *b, BAT *s, BAT *e, int tp1, int tp2)
{
	bool has_nils = false;
	BUN i = 0, cnt = BATcount(b), nils = 0;
	int abort_on_error = 1, *restrict start, *restrict end;

	assert(s && e);
	start = (int*)Tloc(s, 0);
	end = (int*)Tloc(e, 0);

	switch (tp2) {
		case TYPE_bte: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_SUM_CALC(bte, bte, ANALYTICAL_SUM_IMP_NUM);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_sht: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_SUM_CALC(bte, sht, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_sht:
					ANALYTICAL_SUM_CALC(sht, sht, ANALYTICAL_SUM_IMP_NUM);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_int: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_SUM_CALC(bte, int, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_sht:
					ANALYTICAL_SUM_CALC(sht, int, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_int:
					ANALYTICAL_SUM_CALC(int, int, ANALYTICAL_SUM_IMP_NUM);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_lng: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_SUM_CALC(bte, lng, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_sht:
					ANALYTICAL_SUM_CALC(sht, lng, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_int:
					ANALYTICAL_SUM_CALC(int, lng, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_lng:
					ANALYTICAL_SUM_CALC(lng, lng, ANALYTICAL_SUM_IMP_NUM);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_hge: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_SUM_CALC(bte, hge, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_sht:
					ANALYTICAL_SUM_CALC(sht, hge, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_int:
					ANALYTICAL_SUM_CALC(int, hge, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_lng:
					ANALYTICAL_SUM_CALC(lng, hge, ANALYTICAL_SUM_IMP_NUM);
					break;
				case TYPE_hge:
					ANALYTICAL_SUM_CALC(hge, hge, ANALYTICAL_SUM_IMP_NUM);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_flt: {
			switch (tp1) {
				case TYPE_flt:
					ANALYTICAL_SUM_CALC(flt, flt, ANALYTICAL_SUM_IMP_FP);
					break;
				default:
					goto nosupport;
					break;
			}
		}
		case TYPE_dbl: {
			switch (tp1) {
				case TYPE_flt:
					ANALYTICAL_SUM_CALC(flt, dbl, ANALYTICAL_SUM_IMP_FP);
					break;
				case TYPE_dbl:
					ANALYTICAL_SUM_CALC(dbl, dbl, ANALYTICAL_SUM_IMP_FP);
					break;
				default:
					goto nosupport;
					break;
			}
		}
		default:
			goto nosupport;
	}
bailout:
	GDKerror("error while calculating floating-point sum\n");
	return GDK_FAIL;
nosupport:
	GDKerror("sum: type combination (sum(%s)->%s) not supported.\n", ATOMname(tp1), ATOMname(tp2));
	return GDK_FAIL;
calc_overflow:
	GDKerror("22003!overflow in calculation.\n");
	return GDK_FAIL;
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return GDK_SUCCEED;
}

#undef ANALYTICAL_SUM_IMP_NUM
#undef ANALYTICAL_SUM_IMP_FP
#undef ANALYTICAL_SUM_CALC

#define ANALYTICAL_PROD_CALC_NUM(TPE1, TPE2, TPE3) \
	do {                                          \
		TPE1 *pbp, *bp, *bs, *be, v;              \
		TPE2 *restrict rb, curval = TPE2##_nil;   \
		pbp = bp = (TPE1*)Tloc(b, 0);             \
		rb = (TPE2*)Tloc(r, 0);                   \
		bp += cnt;                                \
				for(; pbp<bp; pbp++, i++, rb++) { \
			bs = pbp - start[i];                  \
			be = pbp + end[i];                    \
			for(; bs<be; bs++) {                  \
				v = *bs;                          \
				if (!is_##TPE1##_nil(v)) {        \
					if(is_##TPE2##_nil(curval))   \
						curval = (TPE2) v;        \
					else                          \
						MUL4_WITH_CHECK(TPE1, v, TPE2, curval, TPE2, curval, GDK_##TPE2##_max, TPE3, \
										goto calc_overflow); \
				}                                 \
			}                                     \
			*rb = curval;                         \
			if(is_##TPE2##_nil(curval))           \
				has_nils = true;                  \
			else                                  \
				curval = TPE2##_nil;              \
		}                                         \
		goto finish;                              \
	} while(0);

#define ANALYTICAL_PROD_CALC_NUM_LIMIT(TPE1, TPE2, REAL_IMP) \
	do {                                        \
		TPE1 *pbp, *bp, *bs, *be, v;            \
		TPE2 *restrict rb, curval = TPE2##_nil; \
		pbp = bp = (TPE1*)Tloc(b, 0);           \
		rb = (TPE2*)Tloc(r, 0);                 \
		bp += cnt;                              \
		for(; pbp<bp;pbp++, i++, rb++) {        \
			bs = pbp - start[i];                \
			be = pbp + end[i];                  \
			for(; bs<be; bs++) {                \
				v = *bs;                        \
				if (!is_##TPE1##_nil(v)) {      \
					if(is_##TPE2##_nil(curval)) \
						curval = (TPE2) v;      \
					else                        \
						REAL_IMP(TPE1, v, TPE2, curval, curval, GDK_##TPE2##_max, goto calc_overflow); \
				}                               \
			}                                   \
			*rb = curval;                       \
			if(is_##TPE2##_nil(curval))         \
				has_nils = true;                \
			else                                \
				curval = TPE2##_nil;            \
		}                                       \
		goto finish;                            \
	} while(0);

#define ANALYTICAL_PROD_CALC_FP(TPE1, TPE2)       \
	do {                                          \
		TPE1 *pbp, *bp, *bs, *be, v;              \
		TPE2 *restrict rb, curval = TPE2##_nil;   \
		pbp = bp = (TPE1*)Tloc(b, 0);             \
		rb = (TPE2*)Tloc(r, 0);                   \
		bp += cnt;                                \
				for(; pbp<bp;pbp++, i++, rb++) {  \
			bs = pbp - start[i];                  \
			be = pbp + end[i];                    \
			for(; bs<be; bs++) {                  \
				v = *bs;                          \
				if (!is_##TPE1##_nil(v)) {        \
					if(is_##TPE2##_nil(curval)) { \
						curval = (TPE2) v;        \
					} else if (ABSOLUTE(curval) > 1 && GDK_##TPE2##_max / ABSOLUTE(v) < ABSOLUTE(curval)) { \
						if (abort_on_error)       \
							goto calc_overflow;   \
						curval = TPE2##_nil;      \
						nils++;                   \
					} else {                      \
						curval *= v;              \
					}                             \
				}                                 \
			}                                     \
			*rb = curval;                         \
			if(is_##TPE2##_nil(curval))           \
				has_nils = true;                  \
			else                                  \
				curval = TPE2##_nil;              \
		}                                         \
		goto finish;                              \
	} while(0);

gdk_return
GDKanalyticalprod(BAT *r, BAT *b, BAT *s, BAT *e, int tp1, int tp2)
{
	bool has_nils = false;
	BUN i = 0, cnt = BATcount(b), nils = 0;
	int abort_on_error = 1, *restrict start, *restrict end;

	assert(s && e);
	start = (int*)Tloc(s, 0);
	end = (int*)Tloc(e, 0);

	switch (tp2) {
		case TYPE_bte: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_PROD_CALC_NUM(bte, bte, sht);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_sht: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_PROD_CALC_NUM(bte, sht, int);
					break;
				case TYPE_sht:
					ANALYTICAL_PROD_CALC_NUM(sht, sht, int);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_int: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_PROD_CALC_NUM(bte, int, lng);
					break;
				case TYPE_sht:
					ANALYTICAL_PROD_CALC_NUM(sht, int, lng);
					break;
				case TYPE_int:
					ANALYTICAL_PROD_CALC_NUM(int, int, lng);
					break;
				default:
					goto nosupport;
			}
			break;
		}
#ifdef HAVE_HGE
		case TYPE_lng: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_PROD_CALC_NUM(bte, lng, hge);
					break;
				case TYPE_sht:
					ANALYTICAL_PROD_CALC_NUM(sht, lng, hge);
					break;
				case TYPE_int:
					ANALYTICAL_PROD_CALC_NUM(int, lng, hge);
					break;
				case TYPE_lng:
					ANALYTICAL_PROD_CALC_NUM(lng, lng, hge);
					break;
				default:
					goto nosupport;
			}
			break;
		}
		case TYPE_hge: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(bte, hge, HGEMUL_CHECK);
					break;
				case TYPE_sht:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(sht, hge, HGEMUL_CHECK);
					break;
				case TYPE_int:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(int, hge, HGEMUL_CHECK);
					break;
				case TYPE_lng:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(lng, hge, HGEMUL_CHECK);
					break;
				case TYPE_hge:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(hge, hge, HGEMUL_CHECK);
					break;
				default:
					goto nosupport;
			}
			break;
		}
#else
		case TYPE_lng: {
			switch (tp1) {
				case TYPE_bte:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(bte, lng, LNGMUL_CHECK);
					break;
				case TYPE_sht:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(sht, lng, LNGMUL_CHECK);
					break;
				case TYPE_int:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(int, lng, LNGMUL_CHECK);
					break;
				case TYPE_lng:
					ANALYTICAL_PROD_CALC_NUM_LIMIT(lng, lng, LNGMUL_CHECK);
					break;
				default:
					goto nosupport;
			}
			break;
		}
#endif
		case TYPE_flt: {
			switch (tp1) {
				case TYPE_flt:
					ANALYTICAL_PROD_CALC_FP(flt, flt);
					break;
				default:
					goto nosupport;
					break;
			}
		}
		case TYPE_dbl: {
			switch (tp1) {
				case TYPE_flt:
					ANALYTICAL_PROD_CALC_FP(flt, dbl);
					break;
				case TYPE_dbl:
					ANALYTICAL_PROD_CALC_FP(dbl, dbl);
					break;
				default:
					goto nosupport;
					break;
			}
		}
		default:
			goto nosupport;
	}
nosupport:
	GDKerror("prod: type combination (prod(%s)->%s) not supported.\n", ATOMname(tp1), ATOMname(tp2));
	return GDK_FAIL;
calc_overflow:
	GDKerror("22003!overflow in calculation.\n");
	return GDK_FAIL;
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return GDK_SUCCEED;
}

#undef ANALYTICAL_PROD_CALC_NUM
#undef ANALYTICAL_PROD_CALC_NUM_LIMIT
#undef ANALYTICAL_PROD_CALC_FP

#define ANALYTICAL_AVERAGE_CALC_NUM(TPE,lng_hge)      \
	do {                                              \
		TPE *pbp, *bp, *bs, *be, v;                   \
		pbp = bp = (TPE*)Tloc(b, 0);                  \
		bp += cnt;                                    \
		for(; pbp<bp;pbp++, i++, rb++) {              \
			bs = pbp - start[i];                      \
			be = pbp + end[i];                        \
			for(; bs<be; bs++) {                      \
				v = *bs;                              \
				if (!is_##TPE##_nil(v)) {             \
					ADD_WITH_CHECK(TPE, v, lng_hge, sum, lng_hge, sum, GDK_##lng_hge##_max, goto avg_overflow##TPE); \
					/* count only when no overflow occurs */ \
					n++;                              \
				}                                     \
			}                                         \
			if(0) {                                   \
avg_overflow##TPE:                                    \
				assert(n > 0);                        \
				if (sum >= 0) {                       \
					a = (TPE) (sum / (lng_hge) n);    \
					rr = (BUN) (sum % (SBUN) n);      \
				} else {                              \
					sum = -sum;                       \
					a = - (TPE) (sum / (lng_hge) n);  \
					rr = (BUN) (sum % (SBUN) n);      \
					if (r) {                          \
						a--;                          \
						rr = n - rr;                  \
					}                                 \
				}                                     \
				for(; bs<be; bs++) {                  \
					v = *bs;                          \
					if (is_##TPE##_nil(v))            \
						continue;                     \
					AVERAGE_ITER(TPE, v, a, rr, n);   \
				}                                     \
				curval = a + (dbl) rr / n;            \
				goto calc_done##TPE;                  \
			}                                         \
			curval = n > 0 ? (dbl) sum / n : dbl_nil; \
calc_done##TPE:                                       \
			*rb = curval;                             \
			has_nils = has_nils || (n == 0);          \
			n = 0;                                    \
			sum = 0;                                  \
		}                                             \
		goto finish;                                  \
	} while(0);

#ifdef HAVE_HGE
#define ANALYTICAL_AVERAGE_LNG_HGE(TPE) ANALYTICAL_AVERAGE_CALC_NUM(TPE,hge)
#else
#define ANALYTICAL_AVERAGE_LNG_HGE(TPE) ANALYTICAL_AVERAGE_CALC_NUM(TPE,lng)
#endif

#define ANALYTICAL_AVERAGE_CALC_FP(TPE)      \
	do {                                     \
		TPE *pbp, *bp, *bs, *be, v;          \
		pbp = bp = (TPE*)Tloc(b, 0);         \
		bp += cnt;                           \
		for(; pbp<bp; pbp++, i++, rb++) {    \
			bs = pbp - start[i];             \
			be = pbp + end[i];               \
			for(; bs<be; bs++) {             \
				v = *bs;                     \
				if (!is_##TPE##_nil(v))      \
					AVERAGE_ITER_FLOAT(TPE, v, a, n); \
			}                                \
			curval = (n > 0) ? a : dbl_nil;  \
			*rb = curval;                    \
			has_nils = has_nils || (n == 0); \
			n = 0;                           \
			a = 0;                           \
		}                                    \
		goto finish;                         \
	} while(0);

gdk_return
GDKanalyticalavg(BAT *r, BAT *b, BAT *s, BAT *e, int tpe)
{
	bool has_nils = false;
	BUN i = 0, cnt = BATcount(b), nils = 0, n = 0, rr = 0;
	bool abort_on_error = true;
	int *restrict start, *restrict end;
	dbl *restrict rb = (dbl*)Tloc(r, 0), curval, a = 0;
#ifdef HAVE_HGE
	hge sum = 0;
#else
	lng sum = 0;
#endif

	assert(s && e);
	start = (int*)Tloc(s, 0);
	end = (int*)Tloc(e, 0);

	switch (tpe) {
		case TYPE_bte:
			ANALYTICAL_AVERAGE_LNG_HGE(bte);
			break;
		case TYPE_sht:
			ANALYTICAL_AVERAGE_LNG_HGE(sht);
			break;
		case TYPE_int:
			ANALYTICAL_AVERAGE_LNG_HGE(int);
			break;
		case TYPE_lng:
			ANALYTICAL_AVERAGE_LNG_HGE(lng);
			break;
#ifdef HAVE_HGE
		case TYPE_hge:
			ANALYTICAL_AVERAGE_LNG_HGE(hge);
			break;
#endif  
		case TYPE_flt:
			ANALYTICAL_AVERAGE_CALC_FP(flt);
			break;
		case TYPE_dbl:
			ANALYTICAL_AVERAGE_CALC_FP(dbl);
			break;
		default:
			GDKerror("GDKanalyticalavg: average of type %s unsupported.\n", ATOMname(tpe));
			return GDK_FAIL;
	}
finish:
	BATsetcount(r, cnt);
	r->tnonil = !has_nils;
	r->tnil = has_nils;
	return GDK_SUCCEED;
}

#undef ANALYTICAL_AVERAGE_LNG_HGE
#undef ANALYTICAL_AVERAGE_CALC_NUM
#undef ANALYTICAL_AVERAGE_CALC_FP
