/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "gdk.h"
#include "gdk_private.h"
#include "gdk_calc_private.h"

/*
 * All join variants produce some sort of join on two input BATs,
 * optionally subject to up to two candidate lists.  Only values in
 * the input BATs that are mentioned in the associated candidate list
 * (if provided) are eligible.  They all return two output BATs in the
 * first two arguments.  The join operations differ in the way in
 * which tuples from the two inputs are matched.
 *
 * The outputs consist of two aligned BATs (i.e. same length and same
 * hseqbase (0@0)) that contain the OIDs of the input BATs that match.
 * The candidate lists, if given, contain the OIDs of the associated
 * input BAT which must be considered for matching.  The input BATs
 * must have the same type.
 *
 * All functions also have a parameter nil_matches which indicates
 * whether NIL must be considered an ordinary value that can match, or
 * whether NIL must be considered to never match.
 *
 * The join functions that are provided here are:
 * BATjoin
 *	normal equi-join
 * BATleftjoin
 *	normal equi-join, but the left output is sorted
 * BATouterjoin
 *	equi-join, but the left output is sorted, and if there is no
 *	match for a value in the left input, there is still an output
 *	with NIL in the right output
 * BATsemijoin
 *	equi-join, but the left output is sorted, and if there are
 *	multiple matches, only one is returned (i.e., the left output
 *	is also key)
 * BATthetajoin
 *	theta-join: an extra operator must be provided encoded as an
 *	integer (macros JOIN_EQ, JOIN_NE, JOIN_LT, JOIN_LE, JOIN_GT,
 *	JOIN_GE); values match if the left input has the given
 *	relationship with the right input; order of the outputs is not
 *	guaranteed
 * BATbandjoin
 *	band-join: two extra input values (c1, c2) must be provided as
 *	well as Booleans (li, hi) that indicate whether the value
 *	ranges are inclusive or not; values in the left and right
 *	inputs match if right - c1 <[=] left <[=] right + c2; if c1 or
 *	c2 is NIL, there are no matches
 * BATrangejoin
 *	range-join: the right input consists of two aligned BATs,
 *	values match if the left value is between two corresponding
 *	right values; two extra Boolean parameters, li and hi,
 *	indicate whether equal values match
 *
 * In addition to these functions, there are two more functions that
 * are closely related:
 * BATintersect
 *	intersection: return a candidate list with OIDs of tuples in
 *	the left input whose value occurs in the right input
 * BATdiff
 *	difference: return a candidate list with OIDs of tuples in the
 *	left input whose value does not occur in the right input
 */

/* Perform a bunch of sanity checks on the inputs to a join. */
static gdk_return
joinparamcheck(BAT *l, BAT *r1, BAT *r2, BAT *sl, BAT *sr, const char *func)
{
	if (ATOMtype(l->ttype) != ATOMtype(r1->ttype) ||
	    (r2 && ATOMtype(l->ttype) != ATOMtype(r2->ttype))) {
		GDKerror("%s: inputs not compatible.\n", func);
		return GDK_FAIL;
	}
	if (r2 &&
	    (BATcount(r1) != BATcount(r2) || r1->hseqbase != r2->hseqbase)) {
		GDKerror("%s: right inputs not aligned.\n", func);
		return GDK_FAIL;
	}
	if ((sl && ATOMtype(sl->ttype) != TYPE_oid) ||
	    (sr && ATOMtype(sr->ttype) != TYPE_oid)) {
		GDKerror("%s: candidate lists must have type OID.\n", func);
		return GDK_FAIL;
	}
	if ((sl && !BATtordered(sl)) ||
	    (sr && !BATtordered(sr))) {
		GDKerror("%s: candidate lists must be sorted.\n", func);
		return GDK_FAIL;
	}
	if ((sl && !BATtkey(sl)) ||
	    (sr && !BATtkey(sr))) {
		GDKerror("%s: candidate lists must be unique.\n", func);
		return GDK_FAIL;
	}
	return GDK_SUCCEED;
}

/* Create the result bats for a join, returns the absolute maximum
 * number of outputs that could possibly be generated. */
static BUN
joininitresults(BAT **r1p, BAT **r2p, BUN lcnt, BUN rcnt, bool lkey, bool rkey,
		bool semi, bool nil_on_miss, bool only_misses, BUN estimate)
{
	BAT *r1, *r2;
	BUN maxsize, size;

	lkey |= lcnt <= 1;
	rkey |= rcnt <= 1;

	*r1p = NULL;
	if (r2p)
		*r2p = NULL;
	if (lcnt == 0) {
		/* there is nothing to match */
		maxsize = 0;
	} else if (!only_misses && !nil_on_miss && rcnt == 0) {
		/* if right is empty, we have no hits, so if we don't
		 * want misses, the result is empty */
		maxsize = 0;
	} else if (rkey | semi | only_misses) {
		/* each entry left matches at most one on right, in
		 * case nil_on_miss is also set, each entry matches
		 * exactly one (see below) */
		maxsize = lcnt;
	} else if (lkey) {
		/* each entry on right is matched at most once */
		if (nil_on_miss) {
			/* one entry left could match all right, and
			 * all other entries left match nil */
			maxsize = lcnt + rcnt - 1;
		} else {
			maxsize = rcnt;
		}
	} else {
		/* in the worst case we have a full cross product */
		if (lcnt == 0 || rcnt == 0)
			maxsize = nil_on_miss ? lcnt : 0;
		else if (BUN_MAX / lcnt >= rcnt)
			maxsize = lcnt * rcnt;
		else
			maxsize = BUN_MAX;
	}
	size = estimate == BUN_NONE ? lcnt : estimate;
	if (size > maxsize)
		size = maxsize;
	if ((rkey | semi | only_misses) & nil_on_miss) {
		/* see comment above: each entry left matches exactly
		 * once */
		size = maxsize;
	}

	if (maxsize == 0) {
		r1 = BATdense(0, 0, 0);
		if (r1 == NULL) {
			return BUN_NONE;
		}
		if (r2p) {
			r2 = BATdense(0, 0, 0);
			if (r2 == NULL) {
				BBPreclaim(r1);
				return BUN_NONE;
			}
			*r2p = r2;
		}
		*r1p = r1;
		return 0;
	}

	r1 = COLnew(0, TYPE_oid, size, TRANSIENT);
	if (r1 == NULL) {
		return BUN_NONE;
	}
	r1->tnil = false;
	r1->tnonil = true;
	r1->tkey = true;
	r1->tsorted = true;
	r1->trevsorted = true;
	r1->tseqbase = 0;
	*r1p = r1;
	if (r2p) {
		r2 = COLnew(0, TYPE_oid, size, TRANSIENT);
		if (r2 == NULL) {
			BBPreclaim(r1);
			return BUN_NONE;
		}
		r2->tnil = false;
		r2->tnonil = true;
		r2->tkey = true;
		r2->tsorted = true;
		r2->trevsorted = true;
		r2->tseqbase = 0;
		*r2p = r2;
	}
	return maxsize;
}

#define VALUE(s, x)	(s##vars ?					\
			 s##vars + VarHeapVal(s##vals, (x), s##width) : \
			 (const char *) s##vals + ((x) * s##width))
#define FVALUE(s, x)	((const char *) s##vals + ((x) * s##width))

#define APPEND(b, o)		(((oid *) b->theap.base)[b->batCount++] = (o))

/* Return BATs through r1p and r2p for the case that there is no
 * match between l and r, taking all flags into consideration.
 *
 * This means, if nil_on_miss is set or only_misses is set, *r1p is a
 * copy of the left candidate list or a dense list of all "head"
 * values of l, and *r2p (if r2p is not NULL) is all nil.  If neither
 * of those flags is set, the result is two empty BATs. */
static gdk_return
nomatch(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BUN lstart, BUN lend,
	const oid *lcand, const oid *lcandend,
	bool nil_on_miss, bool only_misses, const char *func, lng t0)
{
	BUN cnt;
	BAT *r1, *r2;

	if (lstart == lend || !(nil_on_miss | only_misses)) {
		/* return empty BATs */
		if ((*r1p = BATdense(0, 0, 0)) == NULL)
			return GDK_FAIL;
		if (r2p && (*r2p = BATdense(0, 0, 0)) == NULL) {
			BBPreclaim(*r1p);
			*r1p = NULL;
			return GDK_FAIL;
		}
		r1 = *r1p;
		r2 = r2p ? *r2p : NULL;
		ALGODEBUG fprintf(stderr,
				  "#%s(l=%s,r=%s)=(" ALGOBATFMT "," ALGOOPTBATFMT ") " LLFMT "us -- nomatch\n",
				  func, BATgetId(l), BATgetId(r),
				  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
				  GDKusec() - t0);
		return GDK_SUCCEED;
	}

	if (lcand) {
		BAT *r1;

		cnt = (BUN) (lcandend - lcand);
		r1 = COLnew(0, TYPE_oid, cnt, TRANSIENT);
		if (r1 == NULL)
			return GDK_FAIL;
		memcpy(Tloc(r1, 0), lcand, cnt * SIZEOF_OID);
		r1->tkey = true;
		r1->tnokey[0] = r1->tnokey[1] = 0;
		r1->tsorted = true;
		r1->tnosorted = 0;
		if (cnt == 1) {
			r1->trevsorted = true;
			r1->tnorevsorted = 0;
		} else {
			r1->trevsorted = false;
			r1->tnorevsorted = 1;
		}
		r1->tseqbase = oid_nil;
		r1->tnil = false;
		r1->tnonil = true;
		BATsetcount(r1, cnt);
		*r1p = r1;
	} else {
		cnt = lend - lstart;
		if ((*r1p = BATdense(0, l->hseqbase + lstart, cnt)) == NULL)
			return GDK_FAIL;
	}
	if (r2p &&
	    (*r2p = BATconstant(0, TYPE_void, &oid_nil, cnt, TRANSIENT)) == NULL) {
		BBPreclaim(*r1p);
		*r1p = NULL;
		return GDK_FAIL;
	}
	r1 = *r1p;
	r2 = r2p ? *r2p : NULL;
	ALGODEBUG fprintf(stderr,
			  "#%s(l=%s,r=%s)=(" ALGOBATFMT "," ALGOOPTBATFMT ") " LLFMT "us -- nomatch\n",
			  func, BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;
}

/* Implementation of join where there is a single value (possibly
 * repeated multiple times) on the left.  This means we can use a
 * point select to find matches in the right column. */
static gdk_return
selectjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	   BUN lstart, BUN lend, BUN lcnt,
	   const oid *restrict lcand, const oid *lcandend,
	   bool nil_matches, lng t0, bool swapped)
{
	BATiter li = bat_iterator(l);
	const void *v;
	BAT *bn = NULL;

	ALGODEBUG fprintf(stderr, "#selectjoin(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ",sl=" ALGOOPTBATFMT ","
			  "sr=" ALGOOPTBATFMT ",nil_matches=%d)%s\n",
			  ALGOBATPAR(l), ALGOBATPAR(r), ALGOOPTBATPAR(sl), ALGOOPTBATPAR(sr),
			  nil_matches,
			  swapped ? " swapped" : "");

	assert(lcnt > 0);
	assert(lcnt == 1 || (l->tsorted && l->trevsorted));

	if (lcand) {
		v = BUNtail(li, *lcand - l->hseqbase);
	} else {
		v = BUNtail(li, lstart);
	}

	if (!nil_matches &&
	    (*ATOMcompare(l->ttype))(v, ATOMnilptr(l->ttype)) == 0) {
		/* NIL doesn't match anything */
		return nomatch(r1p, r2p, l, r, lstart, lend,
			       lcand, lcandend, false, false,
			       "selectjoin", t0);
	}

	bn = BATselect(r, sr, v, NULL, true, true, false);
	if (bn == NULL) {
		return GDK_FAIL;
	}
	if (BATcount(bn) == 0) {
		BBPunfix(bn->batCacheid);
		return nomatch(r1p, r2p, l, r, lstart, lend,
			       lcand, lcandend, false, false,
			       "selectjoin", t0);
	}
	BAT *r1 = COLnew(0, TYPE_oid, lcnt * BATcount(bn), TRANSIENT);
	BAT *r2 = COLnew(0, TYPE_oid, lcnt * BATcount(bn), TRANSIENT);
	if (r1 == NULL || r2 == NULL) {
		BBPunfix(bn->batCacheid);
		BBPreclaim(r1);
		BBPreclaim(r2);
		return GDK_FAIL;
	}

	r1->tsorted = true;
	r1->trevsorted = lcnt == 1;
	r1->tseqbase = BATcount(bn) == 1 && lcand == NULL ? l->hseqbase + lstart : oid_nil;
	r1->tkey = BATcount(bn) == 1;
	r1->tnil = false;
	r1->tnonil = true;
	r2->tsorted = lcnt == 1 || BATcount(bn) == 1;
	r2->trevsorted = BATcount(bn) == 1;
	r2->tseqbase = lcnt == 1 && BATtdense(bn) ? bn->tseqbase : oid_nil;
	r2->tkey = lcnt == 1;
	r2->tnil = false;
	r2->tnonil = true;
	if (BATtdense(bn)) {
		oid *o1p = (oid *) Tloc(r1, 0);
		oid *o2p = (oid *) Tloc(r2, 0);
		oid bno = bn->tseqbase;
		BUN q = BATcount(bn);

		if (lcand) {
			while (lcand < lcandend) {
				for (BUN p = 0; p < q; p++) {
					*o1p++ = *lcand;
					*o2p++ = bno + p;
				}
				lcand++;
			}
		} else {
			while (lstart < lend) {
				for (BUN p = 0; p < q; p++) {
					*o1p++ = lstart + l->hseqbase;
					*o2p++ = bno + p;
				}
				lstart++;
			}
		}
	} else {
		oid *o1p = (oid *) Tloc(r1, 0);
		oid *o2p = (oid *) Tloc(r2, 0);
		const oid *bnp = (const oid *) Tloc(bn, 0);
		BUN q = BATcount(bn);

		if (lcand) {
			while (lcand < lcandend) {
				for (BUN p = 0; p < q; p++) {
					*o1p++ = *lcand;
					*o2p++ = bnp[p];
				}
				lcand++;
			}
		} else {
			while (lstart < lend) {
				for (BUN p = 0; p < q; p++) {
					*o1p++ = lstart + l->hseqbase;
					*o2p++ = bnp[p];
				}
				lstart++;
			}
		}
	}
	BATsetcount(r1, lcnt * BATcount(bn));
	BATsetcount(r2, lcnt * BATcount(bn));
	BBPunfix(bn->batCacheid);
	*r1p = r1;
	*r2p = r2;
	ALGODEBUG fprintf(stderr, "#selectjoin(l=%s,r=%s)=(" ALGOBATFMT "," ALGOOPTBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;
}

#if SIZEOF_OID == SIZEOF_INT
#define binsearch_oid(indir, offset, vals, lo, hi, v, ordering, last) binsearch_int(indir, offset, (const int *) vals, lo, hi, (int) (v), ordering, last)
#endif
#if SIZEOF_OID == SIZEOF_LNG
#define binsearch_oid(indir, offset, vals, lo, hi, v, ordering, last) binsearch_lng(indir, offset, (const lng *) vals, lo, hi, (lng) (v), ordering, last)
#endif

static gdk_return
mergejoin_void(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	       BUN lstart, BUN lend, BUN lcnt,
	       const oid *restrict lcand, const oid *lcandend,
	       bool nil_on_miss, bool only_misses, lng t0, bool swapped)
{
	oid lo, hi;
	BUN i;
	oid o, seq, *o1p = NULL, *o2p = NULL;
	BAT *r1 = NULL, *r2 = NULL;
	const oid *lvals = NULL;

	ALGODEBUG fprintf(stderr, "#mergejoin_void(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ",sl=" ALGOOPTBATFMT ","
			  "sr=" ALGOOPTBATFMT ","
			  "nil_on_miss=%d,only_misses=%d)%s\n",
			  ALGOBATPAR(l), ALGOBATPAR(r), ALGOOPTBATPAR(sl), ALGOOPTBATPAR(sr),
			  nil_on_miss, only_misses,
			  swapped ? " swapped" : "");

	/* r is dense, and if there is a candidate list, it too is
	 * dense.  This means we don't have to do any searches, we
	 * only need to compare ranges to know whether a value from l
	 * has a match in r */
	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(r->tsorted || r->trevsorted);
	assert(sl == NULL || sl->tsorted);
	assert(sr == NULL || sr->tsorted);
	assert(BATcount(l) > 0);
	assert(BATtdense(r));
	assert(BATcount(r) > 0);

	/* figure out range [lo..hi) of values in r that we need to match */
	lo = r->tseqbase;
	hi = lo + BATcount(r);
	if (sr) {
		assert(BATtdense(sr));
		assert(BATcount(sr) > 0);
		/* restrict [lo..hi) range further using candidate
		 * list */
		if (sr->tseqbase > r->hseqbase)
			lo += sr->tseqbase - r->hseqbase;
		if (sr->tseqbase + BATcount(sr) < r->hseqbase + BATcount(r))
			hi -= r->hseqbase + BATcount(r) - sr->tseqbase - BATcount(sr);
	}
	/* at this point, the matchable values in r are [lo..hi) */
	if (!nil_on_miss) {
		r1 = BATselect(l, sl, &lo, &hi, true, false, only_misses);
		if (r1 == NULL)
			return GDK_FAIL;
		if (only_misses && !l->tnonil) {
			/* also look for NILs */
			r2 = BATselect(l, sl, &oid_nil, NULL, true, false, false);
			if (r2 == NULL) {
				BBPreclaim(r1);
				return GDK_FAIL;
			}
			if (BATcount(r2) > 0) {
				BAT *mg = BATmergecand(r1, r2);
				BBPunfix(r1->batCacheid);
				BBPunfix(r2->batCacheid);
				r1 = mg;
				if (r1 == NULL)
					return GDK_FAIL;
			}
		}
		*r1p = r1;
		if (r2p == NULL)
			return GDK_SUCCEED;
		if (BATtdense(r1) && BATtdense(l)) {
			r2 = BATdense(0, l->tseqbase + r1->tseqbase - l->hseqbase + r->hseqbase - r->tseqbase, BATcount(r1));
			if (r2 == NULL) {
				BBPreclaim(r1);
				return GDK_FAIL;
			}
		} else {
			r2 = COLnew(0, TYPE_oid, BATcount(r1), TRANSIENT);
			if (r2 == NULL) {
				BBPreclaim(r1);
				return GDK_FAIL;
			}
			const oid *lp = (const oid *) Tloc(l, 0);
			const oid *o1p = (const oid *) Tloc(r1, 0);
			oid *o2p = (oid *) Tloc(r2, 0);
			hi = BATcount(r1);
			if (BATtdense(r1)) {
				lo = r1->tseqbase - l->hseqbase;
				if (r->tseqbase == r->hseqbase) {
					memcpy(o2p, lp + lo, hi * SIZEOF_OID);
				} else {
					hi += lo;
					for (o = 0; lo < hi; o++, lo++) {
						o2p[o] = lp[lo] - r->tseqbase + r->hseqbase;
					}
				}
			} else if (BATtdense(l)) {
				for (o = 0; o < hi; o++) {
					o2p[o] = o1p[o] - l->hseqbase + l->tseqbase - r->tseqbase + r->hseqbase;
				}
			} else {
				for (o = 0; o < hi; o++) {
					o2p[o] = lp[o1p[o] - l->hseqbase] - r->tseqbase + r->hseqbase;
				}
			}
			r2->tkey = l->tkey;
			r2->tsorted = l->tsorted;
			r2->trevsorted = l->trevsorted;
			r2->tnil = false;
			r2->tnonil = true;
			BATsetcount(r2, BATcount(r1));
		}
		*r2p = r2;
		return GDK_SUCCEED;
	}
	/* nil_on_miss is set, this means we must have a second output */
	assert(r2p);
	if (BATtdense(l)) {
		/* if l is dense, we can further restrict the [lo..hi)
		 * range to values in l that match with values in r */
		i = hi - lo;	/* remember these for nil_on_miss case below */
		o = lo;
		if (l->tseqbase + lstart> lo)
			lo = l->tseqbase + lstart;
		if (l->tseqbase + lend < hi)
			hi = l->tseqbase + lend;
		if (lcand == NULL) {
			/* l is dense, and so is the left candidate
			 * list (if it exists); this means we don't
			 * have to actually look at any values in l:
			 * we can just do some arithmetic; it also
			 * means that r1 will be dense, and if
			 * nil_on_miss is not set, or if all values in
			 * l match, r2 will too */
			seq = l->hseqbase + lstart;

			if (hi <= lo) {
				return nomatch(r1p, r2p, l, r,
					       seq - l->hseqbase,
					       seq + lcnt - l->hseqbase,
					       NULL, NULL, nil_on_miss,
					       only_misses,
					       "mergejoin_void", t0);
			}

			/* at this point, the matched values in l and
			 * r (taking candidate lists into account) are
			 * [lo..hi) which we can translate back to the
			 * respective OID values that we can store in
			 * r1 and r2; note that r1 will be dense since
			 * all values in l will match something (even
			 * if nil if nil_on_miss is set) */
			*r1p = r1 = BATdense(0, seq, lcnt);
			if (r1 == NULL)
				return GDK_FAIL;
			if (hi - lo < lcnt) {
				/* we need to fill in nils in r2 for
				 * missing values */
				*r2p = r2 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
				if (r2 == NULL) {
					BBPreclaim(*r1p);
					return GDK_FAIL;
				}
				o2p = (oid *) Tloc(r2, 0);
				for (o = seq - l->hseqbase + l->tseqbase; o < lo; o++)
					*o2p++ = oid_nil;
				for (o = lo; o < hi; o++)
					*o2p++ = o + r->hseqbase - r->tseqbase;
				for (o = hi; o < lcnt; o++)
					*o2p++ = oid_nil;
				r2->tnonil = false;
				r2->tnil = true;
				if (lcnt <= 1) {
					r2->tsorted = true;
					r2->trevsorted = true;
					if (lcnt == 0)
						BATtseqbase(r2, 0);
					else
						BATtseqbase(r2, *(oid*)Tloc(r2, 0));
				} else {
					r2->tsorted = false;
					r2->trevsorted = false;
					r2->tseqbase = oid_nil;
				}
				/* (hi - lo) different OIDs in r2,
				 * plus one for nil */
				r2->tkey = hi - lo + 1 == lcnt;
				BATsetcount(r2, lcnt);
			} else {
				/* no missing values */
				*r2p = r2 = BATdense(0, r->hseqbase + lo - r->tseqbase, lcnt);
				if (r2 == NULL) {
					BBPreclaim(*r1p);
					return GDK_FAIL;
				}
			}
			goto doreturn;
		}
		/* l is dense, but the candidate list exists and is
		 * not dense; we can, by manipulating the range
		 * [lo..hi), just look at the candidate list values */
		assert(!BATtdense(sl));

		/* translate lo and hi to l's OID values that now need
		 * to match */
		lo = lo - l->tseqbase + l->hseqbase;
		hi = hi - l->tseqbase + l->hseqbase;

		*r1p = r1 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
		if (r1 == NULL)
			return GDK_FAIL;
		o1p = (oid *) Tloc(r1, 0);
		*r2p = r2 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
		if (r2 == NULL) {
			BBPreclaim(r1);
			return GDK_FAIL;
		}
		o2p = (oid *) Tloc(r2, 0);
		r2->tnil = false;
		r2->tnonil = true;
		r2->tkey = true;
		r2->tsorted = true;
		for (i = 0; i < lcnt && lcand[i] < lo; i++) {
			*o1p++ = lcand[i];
			*o2p++ = oid_nil;
		}
		if (i > 0) {
			r2->tnil = true;
			r2->tnonil = false;
			r2->tkey = i == 1;
		}
		for (; i < lcnt && lcand[i] < hi; i++) {
			*o1p++ = lcand[i];
			*o2p++ = lcand[i] - l->hseqbase + l->tseqbase - r->tseqbase + r->hseqbase;
		}
		if (i < lcnt) {
			r2->tkey = !r2->tnil && lcnt - i == 1;
			r2->tnil = true;
			r2->tnonil = false;
			r2->tsorted = false;
			for (; i < lcnt; i++) {
				*o1p++ = lcand[i];
				*o2p++ = oid_nil;
			}
		}
		BATsetcount(r1, (BUN) (o1p - (oid *) Tloc(r1, 0)));
		r1->tseqbase = BATcount(r1) == 0 ? 0 : BATcount(r1) == 1 ? *(oid*)Tloc(r1, 0) : oid_nil;
		r1->tsorted = true;
		r1->trevsorted = BATcount(r1) <= 1;
		r1->tnil = false;
		r1->tnonil = true;
		r1->tkey = true;
		BATsetcount(r2, BATcount(r1));
		r2->tseqbase = r2->tnil || BATcount(r2) > 1 ? oid_nil : BATcount(r2) == 1 ? *(oid*)Tloc(r2, 0) : 0;
		r2->trevsorted = BATcount(r2) <= 1;
		goto doreturn;
	}
	/* l is not dense, so we need to look at the values and check
	 * whether they are in the range [lo..hi) */
	lvals = (const oid *) Tloc(l, 0);
	seq = l->hseqbase;
	lcnt = BATcount(l);
	if (lcand) {
		/* candidate list not dense, we need to do indirection
		 * through the candidate list to look at the value */

		lcnt = (BUN) (lcandend - lcand);
		*r1p = r1 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
		if (r1 == NULL)
			return GDK_FAIL;
		o1p = (oid *) Tloc(r1, 0);
		*r2p = r2 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
		if (r2 == NULL) {
			BBPreclaim(r1);
			return GDK_FAIL;
		}
		o2p = (oid *) Tloc(r2, 0);
		r2->tnil = false;
		r2->tnonil = true;
		for (i = 0; i < lcnt; i++) {
			oid c = lcand[i];

			if (c >= l->hseqbase && c < l->hseqbase + BATcount(l)) {
				o = lvals[c - l->hseqbase];
				if (o >= lo && o < hi) {
					*o1p++ = c;
					*o2p++ = o - r->tseqbase + r->hseqbase;
				} else {
					*o1p++ = c;
					*o2p++ = oid_nil;
					r2->tnil = true;
					r2->tnonil = false;
				}
			}
		}
		r1->tsorted = true;
		r1->trevsorted = BATcount(r1) <= 1;
		r1->tkey = true;
		r1->tseqbase = oid_nil;
		r1->tnil = false;
		r1->tnonil = true;
		BATsetcount(r1, (BUN) (o1p - (oid *) Tloc(r1, 0)));
		BATsetcount(r2, BATcount(r1));
		r2->tsorted = l->tsorted || BATcount(r2) <= 1;
		r2->trevsorted = l->trevsorted || BATcount(r2) <= 1;
		r2->tkey = l->tkey || BATcount(r2) <= 1;
		r2->tseqbase = oid_nil;
		goto doreturn;
	}

	lcnt = lend - lstart;
	lvals += lstart;

	*r1p = r1 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
	if (r1 == NULL)
		return GDK_FAIL;
	o1p = (oid *) Tloc(r1, 0);
	r1->tseqbase = seq;
	r1->tkey = true;
	r1->tsorted = true;
	r1->tnil = false;
	r1->tnonil = true;
	*r2p = r2 = COLnew(0, TYPE_oid, lcnt, TRANSIENT);
	if (r2 == NULL) {
		BBPreclaim(r1);
		return GDK_FAIL;
	}
	o2p = (oid *) Tloc(r2, 0);
	r2->tnil = false;
	r2->tnonil = true;
	for (i = 0; i < lcnt; i++) {
		o = lvals[i];
		if (o >= lo && o < hi) {
			*o1p++ = i + seq;
			*o2p++ = o + r->hseqbase - r->tseqbase;
		} else {
			*o1p++ = i + seq;
			*o2p++ = oid_nil;
			r2->tnil = true;
			r2->tnonil = false;
		}
	}
	BATsetcount(r1, (BUN) (o1p - (oid *) Tloc(r1, 0)));
	r1->trevsorted = BATcount(r1) <= 1;
	if (BATcount(r1) <= 1) {
		r1->trevsorted = true;
		if (BATcount(r1) == 0) {
			r1->tseqbase = 0;
		} else {
			r1->tseqbase = *(oid *) Tloc(r1, 0);
		}
	} else {
		r1->trevsorted = false;
		r1->tnorevsorted = 1;
	}
	BATsetcount(r2, BATcount(r1));
	if (BATcount(r2) <= 1) {
		if (BATcount(r2) == 0) {
			r2->tseqbase = 0;
		} else {
			/* can be oid_nil */
			r2->tseqbase = *(oid *) Tloc(r2, 0);
		}
		r2->tkey = true;
		r2->tsorted = true;
		r2->trevsorted = true;
	} else {
		if (r2->tnil) {
			r2->tkey = false;
			r2->tsorted = false;
			r2->trevsorted = false;
		} else {
			r2->tkey = l->tkey;
			r2->tsorted = l->tsorted;
			r2->trevsorted = l->trevsorted;
		}
		r2->tseqbase = oid_nil;
	}
  doreturn:
	if (r1->tkey)
		virtualize(r1);
	if (r2->tkey && r2->tsorted)
		virtualize(r2);
	ALGODEBUG fprintf(stderr, "#mergejoin_void(l=%s,r=%s)=(" ALGOBATFMT "," ALGOOPTBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;
}

static gdk_return
mergejoin_int(BAT **r1p, BAT **r2p, BAT *l, BAT *r,
	      bool nil_matches, BUN estimate, lng t0, bool swapped)
{
	BAT *r1, *r2;
	BUN lstart, lend;
	BUN rstart, rend;
	BUN lscan, rscan;	/* opportunistic scan window */
	BUN maxsize;
	const int *lvals, *rvals;
	int v;
	BUN nl, nr;
	oid lv;
	BUN i;

	ALGODEBUG fprintf(stderr, "#mergejoin_int(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ")%s\n",
			  ALGOBATPAR(l), ALGOBATPAR(r),
			  swapped ? " swapped" : "");

	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(r->tsorted || r->trevsorted);

	lstart = rstart = 0;
	lend = BATcount(l);
	rend = BATcount(r);
	lvals = (const int *) Tloc(l, 0);
	rvals = (const int *) Tloc(r, 0);
	assert(!r->tvarsized || !r->ttype);

	/* basic properties will be adjusted if necessary later on,
	 * they were initially set by joininitresults() */

	if (lend == 0 || rend == 0) {
		/* there are no matches */
		return nomatch(r1p, r2p, l, r, lstart, lend, NULL, NULL,
			       false, false, "mergejoin_int", t0);
	}

	if ((maxsize = joininitresults(r1p, r2p, BATcount(l), BATcount(r),
				       l->tkey, r->tkey, false, false,
				       false, estimate)) == BUN_NONE)
		return GDK_FAIL;
	r1 = *r1p;
	r2 = r2p ? *r2p : NULL;

	/* determine opportunistic scan window for l and r */
	for (nl = lend - lstart, lscan = 4; nl > 0; lscan++)
		nl >>= 1;
	for (nr = rend - rstart, rscan = 4; nr > 0; rscan++)
		nr >>= 1;

	if (!nil_matches) {
		/* skip over nils at the start of the columns */
		if (lscan < lend - lstart && is_int_nil(lvals[lstart + lscan])) {
			lstart = binsearch_int(NULL, 0, lvals, lstart + lscan,
					       lend - 1, int_nil, 1, 1);
		} else {
			while (is_int_nil(lvals[lstart]))
				lstart++;
		}
		if (rscan < rend - rstart && is_int_nil(rvals[rstart + rscan])) {
			rstart = binsearch_int(NULL, 0, rvals, rstart + rscan,
					       rend - 1, int_nil, 1, 1);
		} else {
			while (is_int_nil(rvals[rstart]))
				rstart++;
		}
	}
	/* from here on we don't have to worry about nil values */

	while (lstart < lend && rstart < rend) {
		v = rvals[rstart];

		if (lscan < lend - lstart && lvals[lstart + lscan] < v) {
			lstart = binsearch_int(NULL, 0, lvals, lstart + lscan,
					       lend - 1, v, 1, 0);
		} else {
			/* scan l for v */
			while (lstart < lend && lvals[lstart] < v)
				lstart++;
		}
		if (lstart >= lend) {
			/* nothing found */
			break;
		}

		/* Here we determine the next value in l that we are
		 * going to try to match in r.  We will also count the
		 * number of occurrences in l of that value.
		 * Afterwards, v points to the value and nl is the
		 * number of times it occurs.  Also, lstart will
		 * point to the next value to be considered (ready for
		 * the next iteration).
		 * If there are many equal values in l (more than
		 * lscan), we will use binary search to find the end
		 * of the sequence.  Obviously, we can do this only if
		 * l is actually sorted (lscan > 0). */
		nl = 1;		/* we'll match (at least) one in l */
		nr = 0;		/* maybe we won't match anything in r */
		v = lvals[lstart];
		if (l->tkey) {
			/* if l is key, there is a single value */
			lstart++;
		} else if (lscan < lend - lstart &&
			   v == lvals[lstart + lscan]) {
			/* lots of equal values: use binary search to
			 * find end */
			nl = binsearch_int(NULL, 0, lvals, lstart + lscan,
					   lend - 1, v, 1, 1);
			nl -= lstart;
			lstart += nl;
		} else {
			/* just scan */
			while (++lstart < lend && v == lvals[lstart])
				nl++;
		}
		/* lstart points one beyond the value we're
		 * going to match: ready for the next iteration. */

		/* First we find the first value in r that is at
		 * least as large as v, then we find the first
		 * value in r that is larger than v.  The difference
		 * is the number of values equal to v and is stored in
		 * nr.
		 * We will use binary search on r to find both ends of
		 * the sequence of values that are equal to v in case
		 * the position is "too far" (more than rscan
		 * away). */

		/* first find the location of the first value in r
		 * that is >= v, then find the location of the first
		 * value in r that is > v; the difference is the
		 * number of values equal to v */

		/* look ahead a little (rscan) in r to see whether
		 * we're better off doing a binary search */
		if (rscan < rend - rstart && rvals[rstart + rscan] < v) {
			/* value too far away in r: use binary
			 * search */
			rstart = binsearch_int(NULL, 0, rvals, rstart + rscan,
					       rend - 1, v, 1, 0);
		} else {
			/* scan r for v */
			while (rstart < rend && rvals[rstart] < v)
				rstart++;
		}
		if (rstart == rend) {
			/* nothing found */
			break;
		}

		/* now find the end of the sequence of equal values v */

		/* if r is key, there is zero or one match, otherwise
		 * look ahead a little (rscan) in r to see whether
		 * we're better off doing a binary search */
		if (r->tkey) {
			if (rstart < rend && v == rvals[rstart]) {
				nr = 1;
				rstart++;
			}
		} else if (rscan < rend - rstart &&
			   v == rvals[rstart + rscan]) {
			/* range too large: use binary search */
			nr = binsearch_int(NULL, 0, rvals, rstart + rscan,
					   rend - 1, v, 1, 1);
			nr -= rstart;
			rstart += nr;
		} else {
			/* scan r for end of range */
			while (rstart < rend && v == rvals[rstart]) {
				nr++;
				rstart++;
			}
		}
		/* rstart points to first value > v or end of
		 * r, and nr is the number of values in r that
		 * are equal to v */
		if (nr == 0) {
			/* no entries in r found */
			continue;
		}
		/* make space: nl values in l match nr values in r, so
		 * we need to add nl * nr values in the results */
		if (BATcount(r1) + nl * nr > BATcapacity(r1)) {
			/* make some extra space by extrapolating how
			 * much more we need (fraction of l we've seen
			 * so far is used as the fraction of the
			 * expected result size we've produced so
			 * far) */
			BUN newcap = (BUN) ((double) BATcount(l) / (BATcount(l) - (lend - lstart)) * (BATcount(r1) + nl * nr) * 1.1);
			if (newcap < nl * nr + BATcount(r1))
				newcap = nl * nr + BATcount(r1) + 1024;
			if (newcap > maxsize)
				newcap = maxsize;
			/* make sure heap.free is set properly before
			 * extending */
			BATsetcount(r1, BATcount(r1));
			if (BATextend(r1, newcap) != GDK_SUCCEED)
				goto bailout;
			BATsetcount(r2, BATcount(r2));
			if (BATextend(r2, newcap) != GDK_SUCCEED)
				goto bailout;
			assert(BATcapacity(r1) == BATcapacity(r2));
		}

		/* maintain properties */
		if (nl > 1) {
			/* value occurs multiple times in l, so entry
			 * in r will be repeated multiple times: hence
			 * r2 is not key and not dense */
			r2->tkey = false;
			r2->tseqbase = oid_nil;
			/* multiple different values will be inserted
			 * in r1 (always in order), so not reverse
			 * ordered anymore */
			r1->trevsorted = false;
		}
		if (nr > 1) {
			/* value occurs multiple times in r, so entry
			 * in l will be repeated multiple times: hence
			 * r1 is not key and not dense */
			r1->tkey = false;
			r1->tseqbase = oid_nil;
			/* multiple different values will be inserted
			 * in r2 (in order), so not reverse ordered
			 * anymore */
			r2->trevsorted = false;
			if (nl > 1) {
				/* multiple values in l match multiple
				 * values in r, so an ordered sequence
				 * will be inserted multiple times in
				 * r2, so r2 is not ordered anymore */
				r2->tsorted = false;
			}
		}
		if (BATcount(r1) > 0) {
			/* a new, higher value will be inserted into
			 * r1, so r1 is not reverse ordered anymore */
			r1->trevsorted = false;
			/* a new higher value will be added to r2 */
			r2->trevsorted = false;
			if (BATtdense(r1) &&
			    ((oid *) r1->theap.base)[r1->batCount - 1] + 1 != l->hseqbase + lstart - nl) {
				r1->tseqbase = oid_nil;
			}
		}

		if (BATcount(r2) > 0 &&
		    BATtdense(r2) &&
		    ((oid *) r2->theap.base)[r2->batCount - 1] + 1 != r->hseqbase + rstart - nr) {
			r2->tseqbase = oid_nil;
		}

		/* insert values */
		lv = l->hseqbase + lstart - nl;
		for (i = 0; i < nl; i++) {
			BUN j;
			oid rv;

			rv = r->hseqbase + rstart - nr;
			for (j = 0; j < nr; j++) {
				APPEND(r1, lv);
				APPEND(r2, rv);
				rv++;
			}
			lv++;
		}
	}
	/* also set other bits of heap to correct value to indicate size */
	BATsetcount(r1, BATcount(r1));
	BATsetcount(r2, BATcount(r2));
	assert(BATcount(r1) == BATcount(r2));
	if (BATcount(r1) > 0) {
		if (BATtdense(r1))
			r1->tseqbase = ((oid *) r1->theap.base)[0];
		if (BATtdense(r2))
			r2->tseqbase = ((oid *) r2->theap.base)[0];
	} else {
		r1->tseqbase = r2->tseqbase = 0;
	}
	ALGODEBUG fprintf(stderr, "#mergejoin_int(l=%s,r=%s)=(" ALGOBATFMT "," ALGOBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;

  bailout:
	BBPreclaim(r1);
	BBPreclaim(r2);
	return GDK_FAIL;
}

static gdk_return
mergejoin_lng(BAT **r1p, BAT **r2p, BAT *l, BAT *r,
	      bool nil_matches, BUN estimate, lng t0, bool swapped)
{
	BAT *r1, *r2;
	BUN lstart, lend;
	BUN rstart, rend;
	BUN lscan, rscan;	/* opportunistic scan window */
	BUN maxsize;
	const lng *lvals, *rvals;
	lng v;
	BUN nl, nr;
	oid lv;
	BUN i;

	ALGODEBUG fprintf(stderr, "#mergejoin_lng(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ")%s\n",
			  ALGOBATPAR(l), ALGOBATPAR(r),
			  swapped ? " swapped" : "");

	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(r->tsorted || r->trevsorted);

	lstart = rstart = 0;
	lend = BATcount(l);
	rend = BATcount(r);
	lvals = (const lng *) Tloc(l, 0);
	rvals = (const lng *) Tloc(r, 0);
	assert(!r->tvarsized || !r->ttype);

	/* basic properties will be adjusted if necessary later on,
	 * they were initially set by joininitresults() */

	if (lend == 0 || rend == 0) {
		/* there are no matches */
		return nomatch(r1p, r2p, l, r, lstart, lend, NULL, NULL,
			       false, false, "mergejoin_lng", t0);
	}

	if ((maxsize = joininitresults(r1p, r2p, BATcount(l), BATcount(r),
				       l->tkey, r->tkey, false, false,
				       false, estimate)) == BUN_NONE)
		return GDK_FAIL;
	r1 = *r1p;
	r2 = r2p ? *r2p : NULL;

	/* determine opportunistic scan window for l and r */
	for (nl = lend - lstart, lscan = 4; nl > 0; lscan++)
		nl >>= 1;
	for (nr = rend - rstart, rscan = 4; nr > 0; rscan++)
		nr >>= 1;

	if (!nil_matches) {
		/* skip over nils at the start of the columns */
		if (lscan < lend - lstart && is_lng_nil(lvals[lstart + lscan])) {
			lstart = binsearch_lng(NULL, 0, lvals, lstart + lscan,
					       lend - 1, lng_nil, 1, 1);
		} else {
			while (is_lng_nil(lvals[lstart]))
				lstart++;
		}
		if (rscan < rend - rstart && is_lng_nil(rvals[rstart + rscan])) {
			rstart = binsearch_lng(NULL, 0, rvals, rstart + rscan,
					       rend - 1, lng_nil, 1, 1);
		} else {
			while (is_lng_nil(rvals[rstart]))
				rstart++;
		}
	}
	/* from here on we don't have to worry about nil values */

	while (lstart < lend && rstart < rend) {
		v = rvals[rstart];

		if (lscan < lend - lstart && lvals[lstart + lscan] < v) {
			lstart = binsearch_lng(NULL, 0, lvals, lstart + lscan,
					       lend - 1, v, 1, 0);
		} else {
			/* scan l for v */
			while (lstart < lend && lvals[lstart] < v)
				lstart++;
		}
		if (lstart >= lend) {
			/* nothing found */
			break;
		}

		/* Here we determine the next value in l that we are
		 * going to try to match in r.  We will also count the
		 * number of occurrences in l of that value.
		 * Afterwards, v points to the value and nl is the
		 * number of times it occurs.  Also, lstart will
		 * point to the next value to be considered (ready for
		 * the next iteration).
		 * If there are many equal values in l (more than
		 * lscan), we will use binary search to find the end
		 * of the sequence.  Obviously, we can do this only if
		 * l is actually sorted (lscan > 0). */
		nl = 1;		/* we'll match (at least) one in l */
		nr = 0;		/* maybe we won't match anything in r */
		v = lvals[lstart];
		if (l->tkey) {
			/* if l is key, there is a single value */
			lstart++;
		} else if (lscan < lend - lstart &&
			   v == lvals[lstart + lscan]) {
			/* lots of equal values: use binary search to
			 * find end */
			nl = binsearch_lng(NULL, 0, lvals, lstart + lscan,
					   lend - 1, v, 1, 1);
			nl -= lstart;
			lstart += nl;
		} else {
			/* just scan */
			while (++lstart < lend && v == lvals[lstart])
				nl++;
		}
		/* lstart points one beyond the value we're
		 * going to match: ready for the next iteration. */

		/* First we find the first value in r that is at
		 * least as large as v, then we find the first
		 * value in r that is larger than v.  The difference
		 * is the number of values equal to v and is stored in
		 * nr.
		 * We will use binary search on r to find both ends of
		 * the sequence of values that are equal to v in case
		 * the position is "too far" (more than rscan
		 * away). */

		/* first find the location of the first value in r
		 * that is >= v, then find the location of the first
		 * value in r that is > v; the difference is the
		 * number of values equal to v */

		/* look ahead a little (rscan) in r to see whether
		 * we're better off doing a binary search */
		if (rscan < rend - rstart && rvals[rstart + rscan] < v) {
			/* value too far away in r: use binary
			 * search */
			rstart = binsearch_lng(NULL, 0, rvals, rstart + rscan,
					       rend - 1, v, 1, 0);
		} else {
			/* scan r for v */
			while (rstart < rend && rvals[rstart] < v)
				rstart++;
		}
		if (rstart == rend) {
			/* nothing found */
			break;
		}

		/* now find the end of the sequence of equal values v */

		/* if r is key, there is zero or one match, otherwise
		 * look ahead a little (rscan) in r to see whether
		 * we're better off doing a binary search */
		if (r->tkey) {
			if (rstart < rend && v == rvals[rstart]) {
				nr = 1;
				rstart++;
			}
		} else if (rscan < rend - rstart &&
			   v == rvals[rstart + rscan]) {
			/* range too large: use binary search */
			nr = binsearch_lng(NULL, 0, rvals, rstart + rscan,
					   rend - 1, v, 1, 1);
			nr -= rstart;
			rstart += nr;
		} else {
			/* scan r for end of range */
			while (rstart < rend && v == rvals[rstart]) {
				nr++;
				rstart++;
			}
		}
		/* rstart points to first value > v or end of
		 * r, and nr is the number of values in r that
		 * are equal to v */
		if (nr == 0) {
			/* no entries in r found */
			continue;
		}
		/* make space: nl values in l match nr values in r, so
		 * we need to add nl * nr values in the results */
		if (BATcount(r1) + nl * nr > BATcapacity(r1)) {
			/* make some extra space by extrapolating how
			 * much more we need (fraction of l we've seen
			 * so far is used as the fraction of the
			 * expected result size we've produced so
			 * far) */
			BUN newcap = (BUN) ((double) BATcount(l) / (BATcount(l) - (lend - lstart)) * (BATcount(r1) + nl * nr) * 1.1);
			if (newcap < nl * nr + BATcount(r1))
				newcap = nl * nr + BATcount(r1) + 1024;
			if (newcap > maxsize)
				newcap = maxsize;
			/* make sure heap.free is set properly before
			 * extending */
			BATsetcount(r1, BATcount(r1));
			if (BATextend(r1, newcap) != GDK_SUCCEED)
				goto bailout;
			BATsetcount(r2, BATcount(r2));
			if (BATextend(r2, newcap) != GDK_SUCCEED)
				goto bailout;
			assert(BATcapacity(r1) == BATcapacity(r2));
		}

		/* maintain properties */
		if (nl > 1) {
			/* value occurs multiple times in l, so entry
			 * in r will be repeated multiple times: hence
			 * r2 is not key and not dense */
			r2->tkey = false;
			r2->tseqbase = oid_nil;
			/* multiple different values will be inserted
			 * in r1 (always in order), so not reverse
			 * ordered anymore */
			r1->trevsorted = false;
		}
		if (nr > 1) {
			/* value occurs multiple times in r, so entry
			 * in l will be repeated multiple times: hence
			 * r1 is not key and not dense */
			r1->tkey = false;
			r1->tseqbase = oid_nil;
			/* multiple different values will be inserted
			 * in r2 (in order), so not reverse ordered
			 * anymore */
			r2->trevsorted = false;
			if (nl > 1) {
				/* multiple values in l match multiple
				 * values in r, so an ordered sequence
				 * will be inserted multiple times in
				 * r2, so r2 is not ordered anymore */
				r2->tsorted = false;
			}
		}
		if (BATcount(r1) > 0) {
			/* a new, higher value will be inserted into
			 * r1, so r1 is not reverse ordered anymore */
			r1->trevsorted = false;
			/* a new higher value will be added to r2 */
			r2->trevsorted = false;
			if (BATtdense(r1) &&
			    ((oid *) r1->theap.base)[r1->batCount - 1] + 1 != l->hseqbase + lstart - nl) {
				r1->tseqbase = oid_nil;
			}
		}

		if (BATcount(r2) > 0 &&
		    BATtdense(r2) &&
		    ((oid *) r2->theap.base)[r2->batCount - 1] + 1 != r->hseqbase + rstart - nr) {
			r2->tseqbase = oid_nil;
		}

		/* insert values */
		lv = l->hseqbase + lstart - nl;
		for (i = 0; i < nl; i++) {
			BUN j;
			oid rv;

			rv = r->hseqbase + rstart - nr;
			for (j = 0; j < nr; j++) {
				APPEND(r1, lv);
				APPEND(r2, rv);
				rv++;
			}
			lv++;
		}
	}
	/* also set other bits of heap to correct value to indicate size */
	BATsetcount(r1, BATcount(r1));
	BATsetcount(r2, BATcount(r2));
	assert(BATcount(r1) == BATcount(r2));
	if (BATcount(r1) > 0) {
		if (BATtdense(r1))
			r1->tseqbase = ((oid *) r1->theap.base)[0];
		if (BATtdense(r2))
			r2->tseqbase = ((oid *) r2->theap.base)[0];
	} else {
		r1->tseqbase = r2->tseqbase = 0;
	}
	ALGODEBUG fprintf(stderr, "#mergejoin_lng(l=%s,r=%s)=(" ALGOBATFMT "," ALGOBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;

  bailout:
	BBPreclaim(r1);
	BBPreclaim(r2);
	return GDK_FAIL;
}

/* Perform a "merge" join on l and r (if both are sorted) with
 * optional candidate lists, or join using binary search on r if l is
 * not sorted.  The return BATs have already been created by the
 * caller.
 *
 * If nil_matches is set, nil values are treated as ordinary values
 * that can match; otherwise nil values never match.
 *
 * If nil_on_miss is set, a nil value is returned in r2 if there is no
 * match in r for a particular value in l (left outer join).
 *
 * If semi is set, only a single set of values in r1/r2 is returned if
 * there is a match of l in r, no matter how many matches there are in
 * r; otherwise all matches are returned.
 *
 * maxsize is the absolute maximum size the output BATs can become (if
 * they were to become larger, we have a bug).
 *
 * t0 and swapped are only for debugging (ALGOMASK set in GDKdebug).
 */
static gdk_return
mergejoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	  BUN lstart, BUN lend, BUN lcnt,
	  const oid *restrict lcand, const oid *lcandend,
	  BUN rstart, BUN rend, BUN rcnt,
	  const oid *restrict rcand, const oid *rcandend,
	  bool nil_matches, bool nil_on_miss, bool semi, bool only_misses,
	  BUN estimate, lng t0, bool swapped)
{
	BUN rstartorig;
	const oid *rcandorig;
	/* [lr]scan determine how far we look ahead in l/r in order to
	 * decide whether we want to do a binary search or a scan */
	BUN lscan, rscan;
	const void *lvals, *rvals; /* the values of l/r (NULL if dense) */
	const char *lvars, *rvars; /* the indirect values (NULL if fixed size) */
	int lwidth, rwidth;	   /* width of the values */
	const void *nil = ATOMnilptr(l->ttype);
	int (*cmp)(const void *, const void *) = ATOMcompare(l->ttype);
	const void *v;		/* points to value under consideration */
	const void *prev = NULL;
	BUN nl, nr;
	bool insert_nil;
	/* equal_order is set if we can scan both BATs in the same
	 * order, so when both are sorted or both are reverse sorted
	 * -- important to know in order to skip over values; if l is
	 * not sorted, this must be set to true and we will always do a
	 * binary search on all of r */
	bool equal_order;
	/* [lr]ordering is either 1 or -1 depending on the order of
	 * l/r: it determines the comparison function used */
	int lordering, rordering;
	oid lv;
	BUN i, j;		/* counters */
	bool lskipped = false;	/* whether we skipped values in l */
	lng loff = 0, roff = 0;	/* set if l/r is dense */
	oid lval = oid_nil, rval = oid_nil; /* temporary space to point v to */

	if (sl == NULL && sr == NULL && !nil_on_miss &&
	    !semi && !only_misses && l->tsorted && r->tsorted && r2p != NULL) {
		/* special cases with far fewer options */
		switch (ATOMbasetype(l->ttype)) {
		case TYPE_int:
			return mergejoin_int(r1p, r2p, l, r, nil_matches,
					     estimate, t0, swapped);
		case TYPE_lng:
			return mergejoin_lng(r1p, r2p, l, r, nil_matches,
					     estimate, t0, swapped);
		}
	}

	ALGODEBUG fprintf(stderr, "#mergejoin(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ",sl=" ALGOOPTBATFMT ","
			  "sr=" ALGOOPTBATFMT ",nil_matches=%d,"
			  "nil_on_miss=%d,semi=%d)%s\n",
			  ALGOBATPAR(l), ALGOBATPAR(r), ALGOOPTBATPAR(sl), ALGOOPTBATPAR(sr),
			  nil_matches, nil_on_miss, semi,
			  swapped ? " swapped" : "");

	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(r->tsorted || r->trevsorted);
	assert(sl == NULL || sl->tsorted);
	assert(sr == NULL || sr->tsorted);

	lvals = BATtvoid(l) ? NULL : Tloc(l, 0);
	rvals = BATtvoid(r) ? NULL : Tloc(r, 0);
	if (l->tvarsized && l->ttype) {
		assert(r->tvarsized && r->ttype);
		lvars = l->tvheap->base;
		rvars = r->tvheap->base;
	} else {
		assert(!r->tvarsized || !r->ttype);
		lvars = rvars = NULL;
	}
	lwidth = l->twidth;
	rwidth = r->twidth;

	/* basic properties will be adjusted if necessary later on,
	 * they were initially set by joininitresults() */

	if (lstart == lend ||
	    rstart == rend ||
	    (!nil_matches &&
	     ((BATtvoid(l) && !BATtdense(l)) ||
	      (BATtvoid(r) && !BATtdense(r)))) ||
	    (BATtvoid(l) && !BATtdense(l) &&
	     (r->tnonil ||
	      (BATtvoid(r) && BATtdense(r)))) ||
	    (BATtvoid(r) && !BATtdense(r) &&
	     (l->tnonil ||
	      (BATtvoid(l) && BATtdense(l))))) {
		/* there are no matches */
		return nomatch(r1p, r2p, l, r, lstart, lend, lcand, lcandend,
			       nil_on_miss, only_misses, "mergejoin", t0);
	}

	BUN maxsize = joininitresults(r1p, r2p, lcnt, rcnt,
				      l->tkey, r->tkey, semi, nil_on_miss,
				      only_misses, estimate);
	if (maxsize == BUN_NONE)
		return GDK_FAIL;
	BAT *r1 = *r1p;
	BAT *r2 = r2p ? *r2p : NULL;

	if (l->tsorted || l->trevsorted) {
		/* determine opportunistic scan window for l */
		for (nl = lcand ? (BUN) (lcandend - lcand) : lend - lstart,
			     lscan = 4;
		     nl > 0;
		     lscan++)
			nl >>= 1;
		equal_order = (l->tsorted && r->tsorted) ||
			(l->trevsorted && r->trevsorted &&
			 !BATtvoid(l) && !BATtvoid(r));
		lordering = l->tsorted && (r->tsorted || !equal_order) ? 1 : -1;
		rordering = equal_order ? lordering : -lordering;
	} else {
		/* if l not sorted, we will always use binary search
		 * on r */
		assert(!BATtvoid(l)); /* void is always sorted */
		lscan = 0;
		equal_order = true;
		lordering = 1;
		rordering = r->tsorted ? 1 : -1;
	}
	/* determine opportunistic scan window for r; if l is not
	 * sorted this is only used to find range of equal values */
	for (nl = rcnt, rscan = 4; nl > 0; rscan++)
		nl >>= 1;

	if (BATtvoid(l)) {
		assert(lvals == NULL);
		if (lcand) {
			lstart = 0;
			lend = (BUN) (lcandend - lcand);
		}
		if (!BATtdense(l))
			loff = lng_nil;
		else
			loff = (lng) l->tseqbase - (lng) l->hseqbase;
	}
	if (BATtvoid(r)) {
		assert(rvals == NULL);
		if (rcand) {
			rstart = 0;
			rend = rcnt;
		}
		if (!BATtdense(r))
			roff = lng_nil;
		else
			roff = (lng) r->tseqbase - (lng) r->hseqbase;
	}
	assert(loff == 0 || lvals == NULL);
	assert(roff == 0 || rvals == NULL);
	/* At this point the various variables that help us through
	 * the algorithm have been set.  The table explains them.  The
	 * first two columns are the inputs, the next three columns
	 * are the variables, the final two columns indicate how the
	 * variables can be used.
	 *
	 * l/r    sl/sr | vals  cand  off | result   value being matched
	 * -------------+-----------------+----------------------------------
	 * dense  NULL  | NULL  NULL  set | i        off==nil?nil:i+off
	 * dense  dense | NULL  NULL  set | i        off==nil?nil:i+off
	 * dense  set   | NULL  set   set | cand[i]  off==nil?nil:cand[i]+off
	 * set    NULL  | set   NULL  0   | i        vals[i]
	 * set    dense | set   NULL  0   | i        vals[i]
	 * set    set   | set   set   0   | cand[i]  vals[cand[i]]
	 *
	 * If {l,r}off is lng_nil, all values in the corresponding bat
	 * are oid_nil because the bat has type VOID and the tseqbase
	 * is nil.
	 */

	rcandorig = rcand;
	rstartorig = rstart;

	/* Before we start adding values to r1 and r2, the properties
	 * are as follows:
	 * tseqbase - 0
	 * tkey - true
	 * tsorted - true
	 * trevsorted - true
	 * tnil - false
	 * tnonil - true
	 * We will modify these as we go along.
	 */
	while (lcand ? lcand < lcandend : lstart < lend) {
		if (lscan == 0) {
			/* always search r completely */
			rcand = rcandorig;
			rstart = rstartorig;
		} else {
			/* If l is sorted (lscan > 0), we look at the
			 * next value in r to see whether we can jump
			 * over a large section of l using binary
			 * search.  We do this by looking ahead in l
			 * (lscan far, to be precise) and seeing if
			 * the value there is still too "small"
			 * (definition depends on sort order of l).
			 * If it is, we use binary search on l,
			 * otherwise we scan l for the next position
			 * with a value greater than or equal to the
			 * value in r.
			 * The next value to match in r is the first
			 * if equal_order is set, the last
			 * otherwise.
			 * When skipping over values in l, we count
			 * how many we skip in nlx.  We need this in
			 * case only_misses or nil_on_miss is set, and
			 * to properly set the dense property in the
			 * first output BAT. */
			BUN nlx = 0; /* number of non-matching values in l */

			if (rcand) {
				if (rcand == rcandend) {
					v = NULL; /* no more values */
				} else if (rvals) {
					v = VALUE(r, (equal_order ? rcand[0] : rcandend[-1]) - r->hseqbase);
				} else {
					rval = is_lng_nil(roff) ? oid_nil : (oid) ((lng) (equal_order ? rcand[0] : rcandend[-1]) + roff);
					v = &rval;
				}
			} else {
				if (rstart == rend) {
					v = NULL;
				} else if (rvals) {
					v = VALUE(r, equal_order ? rstart : rend - 1);
				} else {
					if (is_lng_nil(roff))
						rval = oid_nil;
					else if (equal_order)
						rval = (oid) ((lng) rstart + r->tseqbase);
					else
						rval = (oid) ((lng) rend - 1 + r->tseqbase);
					v = &rval;
				}
			}
			/* here, v points to next value in r, or if
			 * we're at the end of r, v is NULL */
			if (v == NULL) {
				if (lcand) {
					nlx = (BUN) (lcandend - lcand);
					lcand = lcandend;
				} else {
					nlx = lend - lstart;
					lstart = lend;
				}
			} else if (is_lng_nil(loff)) {
				/* all l values are NIL, and the type is OID */
				if (!is_oid_nil(* (oid *) v)) {
					/* value we're looking at in r
					 * is not NIL, so we match
					 * nothing */
					if (lcand) {
						nlx = (BUN) (lcandend - lcand);
						lcand = lcandend;
					} else {
						nlx = lend - lstart;
						lstart = lend;
					}
					v = NULL;
				}
			} else if (lcand) {
				if (lscan < (BUN) (lcandend - lcand)) {
					if (lvals) {
						if (lordering * cmp(VALUE(l, lcand[lscan] - l->hseqbase), v) < 0) {
							nlx = binsearch(lcand, l->hseqbase, l->ttype, lvals, lvars, lwidth, lscan, (BUN) (lcandend - lcand), v, lordering, 0);
						}
					} else {
						lval = (oid) ((lng) *(const oid*)v - loff);
						if (lordering > 0 ? lcand[lscan] < lval : lcand[lscan] > lval) {
							nlx = binsearch(NULL, 0, TYPE_oid, (const char *) lcand, NULL, SIZEOF_OID, 0, lcandend - lcand, &lval, 1, 0);
						}
					}
					lcand += nlx;
					if (lcand == lcandend)
						v = NULL;
				}
			} else if (lvals) {
				if (lscan + lstart < lend) {
					if (lordering * cmp(VALUE(l, lstart + lscan),
							    v) < 0) {
						nlx = lstart;
						lstart = binsearch(NULL, 0, l->ttype, lvals, lvars, lwidth, lstart + lscan, lend, v, lordering, 0);
						nlx = lstart - nlx;
						if (lstart == lend)
							v = NULL;
					}
				}
			} else if (!is_oid_nil(*(const oid *)v)) {
				if (*(const oid *)v > l->tseqbase) {
					nlx = lstart;
					lstart = *(const oid *)v - l->tseqbase;
					if (lstart >= lend) {
						lstart = lend;
						v = NULL;
					}
					nlx = lstart - nlx;
				}
			}
			if (nlx > 0) {
				if (only_misses) {
					if (lcand) {
						lskipped |= nlx > 1;
						while (nlx > 0) {
							APPEND(r1, lcand[-(ssize_t)nlx]);
							nlx--;
						}
					} else {
						while (nlx > 0) {
							APPEND(r1, l->hseqbase + lstart - nlx);
							nlx--;
						}
					}
					if (lskipped)
						r1->tseqbase = oid_nil;
					if (r1->trevsorted && BATcount(r1) > 1)
						r1->trevsorted = false;
				} else if (nil_on_miss) {
					if (r2->tnonil) {
						r2->tnil = true;
						r2->tnonil = false;
						r2->tseqbase = oid_nil;
						r2->tsorted = false;
						r2->trevsorted = false;
						r2->tkey = false;
					}
					if (lcand) {
						lskipped |= nlx > 1;
						while (nlx > 0) {
							APPEND(r1, lcand[-(ssize_t)nlx]);
							APPEND(r2, oid_nil);
							nlx--;
						}
					} else {
						while (nlx > 0) {
							APPEND(r1, l->hseqbase + lstart - nlx);
							APPEND(r2, oid_nil);
							nlx--;
						}
					}
					if (lskipped)
						r1->tseqbase = oid_nil;
					if (r1->trevsorted && BATcount(r1) > 1)
						r1->trevsorted = false;
				} else {
					lskipped = BATcount(r1) > 0;
				}
			}
			if (v == NULL) {
				/* we have exhausted the inputs */
				break;
			}
		}

		/* Here we determine the next value in l that we are
		 * going to try to match in r.  We will also count the
		 * number of occurrences in l of that value.
		 * Afterwards, v points to the value and nl is the
		 * number of times it occurs.  Also, lstart/lcand will
		 * point to the next value to be considered (ready for
		 * the next iteration).
		 * If there are many equal values in l (more than
		 * lscan), we will use binary search to find the end
		 * of the sequence.  Obviously, we can do this only if
		 * l is actually sorted (lscan > 0). */
		nl = 1;		/* we'll match (at least) one in l */
		nr = 0;		/* maybe we won't match anything in r */
		if (lcand) {
			if (is_lng_nil(loff)) {
				/* all values are nil */
				lval = oid_nil;
				v = &lval;
				nl = (BUN) (lcandend - lcand);
				lcand = lcandend;
			} else if (lvals == NULL) {
				/* l is dense, i.e. key, i.e. a single value */
				lval = (oid) ((lng) *lcand++ + loff);
				v = &lval;
			} else {
				v = VALUE(l, lcand[0] - l->hseqbase);
				if (l->tkey) {
					/* if l is key, there is a
					 * single value */
					lcand++;
				} else if (lscan > 0 &&
					   lscan < (BUN) (lcandend - lcand) &&
					   cmp(v, VALUE(l, lcand[lscan] - l->hseqbase)) == 0) {
					/* lots of equal values: use
					 * binary search to find
					 * end */
					nl = binsearch(lcand, l->hseqbase,
						       l->ttype, lvals, lvars,
						       lwidth, lscan,
						       (BUN) (lcandend - lcand),
						       v, lordering, 1);
					lcand += nl;
				} else {
					while (++lcand < lcandend &&
					       cmp(v, VALUE(l, lcand[0] - l->hseqbase)) == 0)
						nl++;
				}
			}
		} else if (lvals) {
			v = VALUE(l, lstart);
			if (l->tkey) {
				/* if l is key, there is a single value */
				lstart++;
			} else if (lscan > 0 &&
				   lscan + lstart < lend &&
				   cmp(v, VALUE(l, lstart + lscan)) == 0) {
				/* lots of equal values: use binary
				 * search to find end */
				nl = binsearch(NULL, 0, l->ttype, lvals, lvars,
					       lwidth, lstart + lscan,
					       lend, v, lordering, 1);
				nl -= lstart;
				lstart += nl;
			} else {
				while (++lstart < lend &&
				       cmp(v, VALUE(l, lstart)) == 0)
					nl++;
			}
		} else if (is_lng_nil(loff)) {
			lval = oid_nil;
			v = &lval;
			nl = lend - lstart;
			lstart = lend;
		} else {
			lval = lstart + l->tseqbase;
			v = &lval;
			lstart++;
		}
		/* lcand/lstart points one beyond the value we're
		 * going to match: ready for the next iteration. */
		if (!nil_matches && !l->tnonil && cmp(v, nil) == 0) {
			/* v is nil and nils don't match anything, set
			 * to NULL to indicate nil */
			v = NULL;
		}
		/* First we find the "first" value in r that is "at
		 * least as large" as v, then we find the "first"
		 * value in r that is "larger" than v.  The difference
		 * is the number of values equal to v and is stored in
		 * nr.  The definitions of "larger" and "first" depend
		 * on the orderings of l and r.  If equal_order is
		 * set, we go through r from low to high, changing
		 * rstart/rcand (this includes the case that l is not
		 * sorted); otherwise we go through r from high to
		 * low, changing rend/rcandend.
		 * In either case, we will use binary search on r to
		 * find both ends of the sequence of values that are
		 * equal to v in case the position is "too far" (more
		 * than rscan away). */
		if (v == NULL) {
			nr = 0;	/* nils don't match anything */
		} else if (is_lng_nil(roff)) {
			if (is_oid_nil(*(const oid *) v)) {
				/* all values in r match */
				nr = rcand ? (BUN) (rcandend - rcand) : rend - rstart;
			} else {
				/* no value in r matches */
				nr = 0;
			}
			/* in either case, we're done after this */
			rstart = rend;
			rcand = rcandend;
		} else if (equal_order) {
			/* first find the location of the first value
			 * in r that is >= v, then find the location
			 * of the first value in r that is > v; the
			 * difference is the number of values equal
			 * v; we change rcand/rstart */
			if (rcand) {
				/* look ahead a little (rscan) in r to
				 * see whether we're better off doing
				 * a binary search */
				if (rvals) {
					if (rscan < (BUN) (rcandend - rcand) &&
					    rordering * cmp(v, VALUE(r, rcand[rscan] - r->hseqbase)) > 0) {
						/* value too far away
						 * in r: use binary
						 * search */
						rcand += binsearch(rcand, r->hseqbase, r->ttype, rvals, rvars, rwidth, rscan, (BUN) (rcandend - rcand), v, rordering, 0);
					} else {
						/* scan r for v */
						while (rcand < rcandend &&
						       rordering * cmp(v, VALUE(r, rcand[0] - r->hseqbase)) > 0)
							rcand++;
					}
					if (rcand < rcandend &&
					    cmp(v, VALUE(r, rcand[0] - r->hseqbase)) == 0) {
						/* if we found an equal value,
						 * look for the last equal
						 * value */
						if (r->tkey) {
							/* r is key, there can
							 * only be a single
							 * equal value */
							nr = 1;
							rcand++;
						} else if (rscan < (BUN) (rcandend - rcand) &&
							   cmp(v, VALUE(r, rcand[rscan] - r->hseqbase)) == 0) {
							/* many equal values:
							 * use binary search to
							 * find the end */
							nr = binsearch(rcand, r->hseqbase, r->ttype, rvals, rvars, rwidth, rscan, (BUN) (rcandend - rcand), v, rordering, 1);
							rcand += nr;
						} else {
							/* scan r for end of
							 * range */
							do {
								nr++;
								rcand++;
							} while (rcand < rcandend &&
								 cmp(v, VALUE(r, rcand[0] - r->hseqbase)) == 0);
						}
					}
				} else {
					rval = (oid) ((lng) *(const oid*)v - roff);
					if (rscan < (BUN) (rcandend - rcand) &&
					    (rordering > 0 ? rcand[rscan] < rval : rcand[rscan] > rval)) {
						rcand += binsearch(NULL, 0, TYPE_oid, (const char *) rcand, NULL, SIZEOF_OID, rscan, rcandend - rcand, &rval, 1, 0);
					} else {
						while (rcand < rcandend &&
						       (rordering > 0 ? *rcand < rval : *rcand > rval))
							rcand++;
					}
					if (rcand < rcandend && *rcand == rval) {
						nr = 1;
						rcand++;
					}
				}
			} else if (rvals) {
				if (rstart + rscan < rend &&
				    rordering * cmp(v, VALUE(r, rstart + rscan)) > 0) {
					/* value too far away
					 * in r: use binary
					 * search */
					rstart = binsearch(NULL, 0, r->ttype, rvals, rvars, rwidth, rstart + rscan, rend, v, rordering, 0);
				} else {
					/* scan r for v */
					while (rstart < rend &&
					       rordering * cmp(v, VALUE(r, rstart)) > 0)
						rstart++;
				}
				if (rstart < rend &&
				    cmp(v, VALUE(r, rstart)) == 0) {
					/* if we found an equal value,
					 * look for the last equal
					 * value */
					if (r->tkey) {
						/* r is key, there can only be a single equal value */
						nr = 1;
						rstart++;
					} else if (rstart + rscan < rend &&
						   cmp(v, VALUE(r, rstart + rscan)) == 0) {
						/* use binary search to find the end */
						nr = binsearch(NULL, 0, r->ttype, rvals, rvars, rwidth, rstart + rscan, rend, v, rordering, 1);
						nr -= rstart;
						rstart += nr;
					} else {
						/* scan r for end of range */
						do {
							nr++;
							rstart++;
						} while (rstart < rend &&
							 cmp(v, VALUE(r, rstart)) == 0);
					}
				}
			} else if (!is_oid_nil((rval = *(const oid *)v))) {
				/* r is dense or void-nil, so we don't
				 * need to search, we know there is
				 * either zero or one match (note that
				 * all nils have already been dealt
				 * with) */
				if (rval >= rstart + r->tseqbase) {
					if (rval >= rend + r->tseqbase) {
						/* beyond the end: no match */
						rstart = rend;
					} else {
						/* within range: a
						 * single match */
						rstart = rval - r->tseqbase + 1;
						nr = 1;
					}
				}
			}
			/* rstart or rcand points to first value > v
			 * or end of r, and nr is the number of values
			 * in r that are equal to v */
		} else {
			/* first find the location of the first value
			 * in r that is > v, then find the location
			 * of the first value in r that is >= v; the
			 * difference is the number of values equal
			 * v; we change rcandend/rend */
			if (rcand) {
				/* look back from the end a little
				 * (rscan) in r to see whether we're
				 * better off doing a binary search */
				if (rvals) {
					if (rscan < (BUN) (rcandend - rcand) &&
					    rordering * cmp(v, VALUE(r, rcandend[-(ssize_t)rscan - 1] - r->hseqbase)) < 0) {
						/* value too far away
						 * in r: use binary
						 * search */
						rcandend = rcand + binsearch(rcand, r->hseqbase, r->ttype, rvals, rvars, rwidth, 0, (BUN) (rcandend - rcand) - rscan, v, rordering, 1);
					} else {
						/* scan r for v */
						while (rcand < rcandend &&
						       rordering * cmp(v, VALUE(r, rcandend[-1] - r->hseqbase)) < 0)
							rcandend--;
					}
					if (rcand < rcandend &&
					    cmp(v, VALUE(r, rcandend[-1] - r->hseqbase)) == 0) {
						/* if we found an equal value,
						 * look for the last equal
						 * value */
						if (r->tkey) {
							/* r is key, there can only be a single equal value */
							nr = 1;
							rcandend--;
						} else if (rscan < (BUN) (rcandend - rcand) &&
							   cmp(v, VALUE(r, rcandend[-(ssize_t)rscan - 1] - r->hseqbase)) == 0) {
							/* use binary search to find the start */
							nr = binsearch(rcand, r->hseqbase, r->ttype, rvals, rvars, rwidth, 0, (BUN) (rcandend - rcand) - rscan, v, rordering, 0);
							nr = (BUN) (rcandend - rcand) - nr;
							rcandend -= nr;
						} else {
							/* scan r for start of range */
							do {
								nr++;
								rcandend--;
							} while (rcand < rcandend &&
								 cmp(v, VALUE(r, rcandend[-1] - r->hseqbase)) == 0);
						}
					}
				} else {
					rval = (oid) ((lng) *(const oid*)v - roff);
					if (rscan < (BUN) (rcandend - rcand) &&
					    (rordering > 0 ? rcandend[-(ssize_t)rscan - 1] > rval : rcandend[-(ssize_t)rscan - 1] < rval)) {
						rcandend = rcand + binsearch(NULL, 0, TYPE_oid, (const char *) rcand, NULL, SIZEOF_OID, 0, rcandend - rcand - rscan, &rval, 1, 1);
					} else {
						while (rcand < rcandend &&
						       (rordering > 0 ? rcandend[-1] > rval : rcandend[-1] < rval))
							rcand++;
					}
					if (rcand < rcandend && rcandend[-1] == rval) {
						nr = 1;
						rcandend--;
					}
				}
			} else if (rvals) {
				if (rstart + rscan < rend &&
				    rordering * cmp(v, VALUE(r, rend - rscan - 1)) < 0) {
					/* value too far away
					 * in r: use binary
					 * search */
					rend = binsearch(NULL, 0, r->ttype, rvals, rvars, rwidth, rstart, rend - rscan, v, rordering, 1);
				} else {
					/* scan r for v */
					while (rstart < rend &&
					       rordering * cmp(v, VALUE(r, rend - 1)) < 0)
						rend--;
				}
				if (rstart < rend &&
				    cmp(v, VALUE(r, rend - 1)) == 0) {
					/* if we found an equal value,
					 * look for the last equal
					 * value */
					if (r->tkey) {
						/* r is key, there can only be a single equal value */
						nr = 1;
						rend--;
					} else if (rstart + rscan < rend &&
						   cmp(v, VALUE(r, rend - rscan - 1)) == 0) {
						/* use binary search to find the end */
						nr = binsearch(NULL, 0, r->ttype, rvals, rvars, rwidth, rstart, rend - rscan, v, rordering, 0);
						nr = rend - nr;
						rend -= nr;
					} else {
						/* scan r for end of range */
						do {
							nr++;
							rend--;
						} while (rstart < rend &&
							 cmp(v, VALUE(r, rend - 1)) == 0);
					}
				}
			} else if (!is_oid_nil((rval = *(const oid *)v))) {
				/* r is dense or void-nil, so we don't
				 * need to search, we know there is
				 * either zero or one match (note that
				 * all nils have already been dealt
				 * with) */
				if (rval < rend + r->tseqbase) {
					if (rval < rstart + r->tseqbase) {
						/* beyond the end: no match */
						rend = rstart;
					} else {
						/* within range: a
						 * single match */
						rend = rval - r->tseqbase;
						nr = 1;
					}
				}
			}
			/* rstart or rcand points to first value > v
			 * or end of r, and nr is the number of values
			 * in r that are equal to v */
		}

		if (nr == 0) {
			/* no entries in r found */
			if (!(nil_on_miss | only_misses)) {
				if (lscan > 0 &&
				    (rcand ? rcand == rcandend : rstart == rend)) {
					/* nothing more left to match
					 * in r */
					break;
				}
				lskipped = BATcount(r1) > 0;
				continue;
			}
			/* insert a nil to indicate a non-match */
			insert_nil = true;
			nr = 1;
			if (r2) {
				r2->tnil = true;
				r2->tnonil = false;
				r2->tsorted = false;
				r2->trevsorted = false;
				r2->tseqbase = oid_nil;
				r2->tkey = false;
			}
		} else if (only_misses) {
			/* we had a match, so we're not interested */
			lskipped = BATcount(r1) > 0;
			continue;
		} else {
			insert_nil = false;
			if (semi) {
				/* for semi-join, only insert single
				 * value */
				nr = 1;
			}
		}
		if (lcand &&
		    nl > 1 &&
		    lcand[-1] != lcand[-1 - (ssize_t) nl] + nl) {
			/* not all values in the range are
			 * candidates */
			lskipped = true;
		}
		/* make space: nl values in l match nr values in r, so
		 * we need to add nl * nr values in the results */
		if (BATcount(r1) + nl * nr > BATcapacity(r1)) {
			/* make some extra space by extrapolating how
			 * much more we need (fraction of l we've seen
			 * so far is used as the fraction of the
			 * expected result size we've produced so
			 * far) */
			BUN newcap = (BUN) ((double) lcnt / (lcnt - (lcand ? (BUN) (lcandend - lcand) : (lend - lstart))) * (BATcount(r1) + nl * nr) * 1.1);
			if (newcap < nl * nr + BATcount(r1))
				newcap = nl * nr + BATcount(r1) + 1024;
			if (newcap > maxsize)
				newcap = maxsize;
			/* make sure heap.free is set properly before
			 * extending */
			BATsetcount(r1, BATcount(r1));
			if (BATextend(r1, newcap) != GDK_SUCCEED)
				goto bailout;
			if (r2) {
				BATsetcount(r2, BATcount(r2));
				if (BATextend(r2, newcap) != GDK_SUCCEED)
					goto bailout;
				assert(BATcapacity(r1) == BATcapacity(r2));
			}
		}

		/* maintain properties */
		if (nl > 1) {
			if (r2) {
				/* value occurs multiple times in l,
				 * so entry in r will be repeated
				 * multiple times: hence r2 is not key
				 * and not dense */
				r2->tkey = false;
				r2->tseqbase = oid_nil;
			}
			/* multiple different values will be inserted
			 * in r1 (always in order), so not reverse
			 * ordered anymore */
			r1->trevsorted = false;
		}
		if (nr > 1) {
			/* value occurs multiple times in r, so entry
			 * in l will be repeated multiple times: hence
			 * r1 is not key and not dense */
			r1->tkey = false;
			r1->tseqbase = oid_nil;
			if (r2) {
				/* multiple different values will be
				 * inserted in r2 (in order), so not
				 * reverse ordered anymore */
				r2->trevsorted = false;
				if (nl > 1) {
					/* multiple values in l match
					 * multiple values in r, so an
					 * ordered sequence will be
					 * inserted multiple times in
					 * r2, so r2 is not ordered
					 * anymore */
					r2->tsorted = false;
				}
			}
		}
		if (lscan == 0) {
			/* deduce relative positions of r matches for
			 * this and previous value in v */
			if (prev && r2) {
				/* keyness or r2 can only be assured
				 * as long as matched values are
				 * ordered */
				int ord = rordering * cmp(prev, v);
				if (ord < 0) {
					/* previous value in l was
					 * less than current */
					r2->trevsorted = false;
					r2->tkey &= r2->tsorted;
				} else if (ord > 0) {
					/* previous value was
					 * greater */
					r2->tsorted = false;
					r2->tkey &= r2->trevsorted;
				} else {
					/* value can be equal if
					 * intervening values in l
					 * didn't match anything; if
					 * multiple values match in r,
					 * r2 won't be sorted */
					r2->tkey = false;
					if (nr > 1) {
						r2->tsorted = false;
						r2->trevsorted = false;
					}
				}
			}
			prev = v;
		}
		if (BATcount(r1) > 0) {
			/* a new, higher value will be inserted into
			 * r1, so r1 is not reverse ordered anymore */
			r1->trevsorted = false;
			if (r2) {
				/* depending on whether l and r are
				 * ordered the same or not, a new
				 * higher or lower value will be added
				 * to r2 */
				if (equal_order)
					r2->trevsorted = false;
				else {
					r2->tsorted = false;
					r2->tseqbase = oid_nil;
				}
			}
			/* if there is a left candidate list, it may
			 * be that the next value added isn't
			 * consecutive with the last one */
			if (lskipped ||
			    (lcand && ((oid *) r1->theap.base)[r1->batCount - 1] + 1 != lcand[-(ssize_t)nl]))
				r1->tseqbase = oid_nil;
		}

		/* insert values: first the left output */
		if (lcand) {
			for (i = nl; i > 0; i--) {
				lv = lcand[-(ssize_t)i];
				for (j = 0; j < nr; j++)
					APPEND(r1, lv);
			}
		} else {
			for (i = nl; i > 0; i--) {
				lv = l->hseqbase + lstart - i;
				for (j = 0; j < nr; j++)
					APPEND(r1, lv);
			}
		}
		/* then the right output, various different ways of
		 * doing it */
		if (r2 == NULL) {
			/* nothing to do */
		} else if (insert_nil) {
			do {
				for (i = 0; i < nr; i++) {
					APPEND(r2, oid_nil);
				}
			} while (--nl > 0);
		} else if (rcand && equal_order) {
			if (r2->batCount > 0 &&
			    BATtdense(r2) &&
			    ((oid *) r2->theap.base)[r2->batCount - 1] + 1 != rcand[-(ssize_t)nr])
				r2->tseqbase = oid_nil;
			do {
				for (i = nr; i > 0; i--) {
					APPEND(r2, rcand[-(ssize_t)i]);
				}
			} while (--nl > 0);
		} else if (rcand) {
			if (r2->batCount > 0 &&
			    BATtdense(r2) &&
			    ((oid *) r2->theap.base)[r2->batCount - 1] + 1 != rcandend[0])
				r2->tseqbase = oid_nil;
			do {
				for (i = 0; i < nr; i++) {
					APPEND(r2, rcandend[i]);
				}
			} while (--nl > 0);
		} else if (equal_order) {
			if (r2->batCount > 0 &&
			    BATtdense(r2) &&
			    ((oid *) r2->theap.base)[r2->batCount - 1] + 1 != r->hseqbase + rstart - nr)
				r2->tseqbase = oid_nil;
			do {
				for (i = nr; i > 0; i--) {
					APPEND(r2, r->hseqbase + rstart - i);
				}
			} while (--nl > 0);
		} else {
			if (r2->batCount > 0 &&
			    BATtdense(r2) &&
			    ((oid *) r2->theap.base)[r2->batCount - 1] + 1 != rend + r->hseqbase)
				r2->tseqbase = oid_nil;
			do {
				for (i = 0; i < nr; i++) {
					APPEND(r2, rend + r->hseqbase + i);
				}
			} while (--nl > 0);
		}
	}
	/* also set other bits of heap to correct value to indicate size */
	BATsetcount(r1, BATcount(r1));
	r1->tseqbase = oid_nil;
	if (r2) {
		BATsetcount(r2, BATcount(r2));
		assert(BATcount(r1) == BATcount(r2));
		r2->tseqbase = oid_nil;
	}
	if (BATcount(r1) > 0) {
		if (BATtdense(r1))
			r1->tseqbase = ((oid *) r1->theap.base)[0];
		if (r2 && BATtdense(r2))
			r2->tseqbase = ((oid *) r2->theap.base)[0];
	} else {
		r1->tseqbase = 0;
		if (r2) {
			r2->tseqbase = 0;
		}
	}
	ALGODEBUG fprintf(stderr, "#mergejoin(l=%s,r=%s)=(" ALGOBATFMT "," ALGOOPTBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;

  bailout:
	BBPreclaim(r1);
	BBPreclaim(r2);
	return GDK_FAIL;
}

/* binary search in a candidate list, return true if found, false if not */
inline bool
binsearchcand(const oid *cand, BUN lo, BUN hi, oid v)
{
	BUN mid;

	--hi;			/* now hi is inclusive */
	if (v < cand[lo] || v > cand[hi])
		return false;
	while (hi > lo) {
		mid = (lo + hi) / 2;
		if (cand[mid] == v)
			return true;
		if (cand[mid] < v)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return cand[lo] == v;
}

#define HASHLOOPBODY()							\
	do {								\
		if (BUNlast(r1) == BATcapacity(r1)) {			\
			newcap = BATgrows(r1);				\
			if (newcap > maxsize)				\
				newcap = maxsize;			\
			BATsetcount(r1, BATcount(r1));			\
			if (BATextend(r1, newcap) != GDK_SUCCEED)	\
				goto bailout;				\
			if (r2) {					\
				BATsetcount(r2, BATcount(r2));		\
				if (BATextend(r2, newcap) != GDK_SUCCEED) \
					goto bailout;			\
				assert(BATcapacity(r1) == BATcapacity(r2)); \
			}						\
		}							\
		APPEND(r1, lo);						\
		if (r2)							\
			APPEND(r2, ro);					\
		nr++;							\
	} while (false)

#define HASHloop_bound(bi, h, hb, v, lo, hi)		\
	for (hb = HASHget(h, HASHprobe((h), v));	\
	     hb != HASHnil(h);				\
	     hb = HASHgetlink(h,hb))			\
		if (hb >= (lo) && hb < (hi) &&		\
		    (cmp == NULL ||			\
		     (*cmp)(v, BUNtail(bi, hb)) == 0))

#define HASHloop_bound_TYPE(bi, h, hb, v, lo, hi, TYPE)			\
	for (hb = HASHget(h, hash_##TYPE(h, v));			\
	     hb != HASHnil(h);						\
	     hb = HASHgetlink(h,hb))					\
		if (hb >= (lo) && hb < (hi) &&				\
		    * (const TYPE *) v == * (const TYPE *) BUNtloc(bi, hb))

#define HASHJOIN(TYPE, WIDTH)						\
	do {								\
		BUN hashnil = HASHnil(hsh);				\
		for (lo = lstart + l->hseqbase;				\
		     lstart < lend;					\
		     lo++) {						\
			v = FVALUE(l, lstart);				\
			lstart++;					\
			nr = 0;						\
			if (!is_##TYPE##_nil(*(const TYPE*)v)) {	\
				for (rb = HASHget##WIDTH(hsh, hash_##TYPE(hsh, v)); \
				     rb != hashnil;			\
				     rb = HASHgetlink##WIDTH(hsh, rb))	\
					if (rb >= rl && rb < rh &&	\
					    * (const TYPE *) v == ((const TYPE *) base)[rb]) { \
						ro = (oid) (rb - rl + rseq); \
						HASHLOOPBODY();		\
					}				\
			}						\
			if (nr == 0) {					\
				lskipped = BATcount(r1) > 0;		\
			} else {					\
				if (lskipped) {				\
					r1->tseqbase = oid_nil;		\
				}					\
				if (nr > 1) {				\
					r1->tkey = false;		\
					r1->tseqbase = oid_nil;		\
				}					\
				if (BATcount(r1) > nr)			\
					r1->trevsorted = false;		\
			}						\
		}							\
	} while (false)

static gdk_return
hashjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	 BUN lstart, BUN lend, BUN lcnt,
	 const oid *restrict lcand, const oid *lcandend,
	 BUN rstart, BUN rend, BUN rcnt,
	 const oid *restrict rcand, const oid *rcandend,
	 bool nil_matches, bool nil_on_miss, bool semi, bool only_misses,
	 BUN estimate, lng t0, bool swapped, bool phash, const char *reason)
{
	oid lo, ro;
	BATiter ri;
	BUN rb;
	BUN rl, rh;
	oid rseq;
	BUN nr, newcap;
	const char *lvals;
	const char *lvars;
	int lwidth;
	const void *nil = ATOMnilptr(l->ttype);
	int (*cmp)(const void *, const void *) = ATOMcompare(l->ttype);
	oid lval = oid_nil;	/* hold value if l is dense */
	const char *v = (const char *) &lval;
	bool lskipped = false;	/* whether we skipped values in l */
	Hash *restrict hsh;
	int t;

	(void) rcandend;
	ALGODEBUG fprintf(stderr, "#hashjoin(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ",sl=" ALGOOPTBATFMT ","
			  "sr=" ALGOOPTBATFMT ",nil_matches=%d,"
			  "nil_on_miss=%d,semi=%d,only_misses=%d)%s%s%s\n",
			  ALGOBATPAR(l), ALGOBATPAR(r),
			  ALGOOPTBATPAR(sl), ALGOOPTBATPAR(sr),
			  nil_matches, nil_on_miss, semi, only_misses,
			  swapped ? " swapped" : "",
			  *reason ? " " : "", reason);

	assert(!BATtvoid(r));
	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(sl == NULL || sl->tsorted);
	assert(sr == NULL || sr->tsorted);

	lwidth = l->twidth;
	lvals = (const char *) Tloc(l, 0);
	if (l->tvarsized && l->ttype) {
		assert(r->tvarsized && r->ttype);
		lvars = l->tvheap->base;
	} else {
		assert(!r->tvarsized || !r->ttype);
		lvars = NULL;
	}
	/* offset to convert BUN to OID for value in right column */
	rseq = r->hseqbase;

	if (lstart == lend || rstart == rend)
		return nomatch(r1p, r2p, l, r, lstart, lend, lcand, lcandend,
			       nil_on_miss, only_misses, "hashjoin", t0);

	BUN maxsize = joininitresults(r1p, r2p, lcnt, rcnt,
				      l->tkey, r->tkey, semi, nil_on_miss,
				      only_misses, estimate);
	if (maxsize == BUN_NONE)
		return GDK_FAIL;
	BAT *r1 = *r1p;
	BAT *r2 = r2p ? *r2p : NULL;

	/* basic properties will be adjusted if necessary later on,
	 * they were initially set by joininitresults() */

	if (r2) {
		r2->tkey = l->tkey;
		/* r2 is not likely to be sorted (although it is
		 * certainly possible) */
		r2->tsorted = false;
		r2->trevsorted = false;
		r2->tseqbase = oid_nil;
	}

	if (sl && !BATtdense(sl))
		r1->tseqbase = oid_nil;

	rl = 0;
	if (phash) {
		BAT *b = BBPdescriptor(VIEWtparent(r));
		assert(sr == NULL);
		ALGODEBUG fprintf(stderr, "#hashjoin(%s): using "
				  "parent(" ALGOBATFMT ") for hash\n",
				  BATgetId(r), ALGOBATPAR(b));
		rl = (BUN) ((r->theap.base - b->theap.base) >> r->tshift);
		r = b;
	}
	rh = rl + rend;
	rl += rstart;
	rseq += rstart;

	if (sr) {
		if (BATtdense(sr) &&
		    BATcheckhash(r) &&
		    BATcount(r) / ((size_t *) r->thash->heap.base)[5] * lcnt < lcnt + rcnt) {
			ALGODEBUG fprintf(stderr, "#hashjoin(%s): using "
					  "existing hash with candidate list\n",
					  BATgetId(r));
			hsh = r->thash;
			sr = NULL;
		} else {
			char ext[32];
			assert(!phash);
			ALGODEBUG fprintf(stderr, "#hashjoin(%s): creating "
					  "hash for candidate list\n",
					  BATgetId(r));
			snprintf(ext, sizeof(ext), "thash%x", sr->batCacheid);
			if ((hsh = BAThash_impl(r, sr, ext)) == NULL)
				goto bailout;
		}
	} else {
		if (BAThash(r) != GDK_SUCCEED)
			goto bailout;
		hsh = r->thash;
	}
	ri = bat_iterator(r);
	t = ATOMbasetype(r->ttype);

	if (lcand) {
		while (lcand < lcandend) {
			lo = *lcand++;
			if (BATtvoid(l)) {
				if (BATtdense(l))
					lval = lo - l->hseqbase + l->tseqbase;
			} else {
				v = VALUE(l, lo - l->hseqbase);
			}
			nr = 0;
			if (!nil_matches && cmp(v, nil) == 0) {
				/* no match */
			} else if (sr) {
				for (rb = HASHget(hsh, HASHprobe(hsh, v));
				     rb != HASHnil(hsh);
				     rb = HASHgetlink(hsh, rb)) {
					ro = BUNtoid(sr, rb);
					if ((*cmp)(v, BUNtail(ri, ro - r->hseqbase)) != 0)
						continue;
					if (only_misses) {
						nr++;
						break;
					}
					HASHLOOPBODY();
					if (semi)
						break;
				}
			} else {
				HASHloop_bound(ri, hsh, rb, v, rl, rh) {
					ro = (oid) (rb - rl + rseq);
					if (only_misses) {
						nr++;
						break;
					}
					HASHLOOPBODY();
					if (semi)
						break;
				}
			}
			if (nr == 0) {
				if (only_misses) {
					nr = 1;
					if (BUNlast(r1) == BATcapacity(r1)) {
						newcap = BATgrows(r1);
						if (newcap > maxsize)
							newcap = maxsize;
						BATsetcount(r1, BATcount(r1));
						if (BATextend(r1, newcap) != GDK_SUCCEED)
							goto bailout;
					}
					APPEND(r1, lo);
					if (lskipped)
						r1->tseqbase = oid_nil;
				} else if (nil_on_miss) {
					nr = 1;
					r2->tnil = true;
					r2->tnonil = false;
					r2->tkey = false;
					if (BUNlast(r1) == BATcapacity(r1)) {
						newcap = BATgrows(r1);
						if (newcap > maxsize)
							newcap = maxsize;
						BATsetcount(r1, BATcount(r1));
						BATsetcount(r2, BATcount(r2));
						if (BATextend(r1, newcap) != GDK_SUCCEED ||
						    BATextend(r2, newcap) != GDK_SUCCEED)
							goto bailout;
						assert(BATcapacity(r1) == BATcapacity(r2));
					}
					APPEND(r1, lo);
					APPEND(r2, oid_nil);
				} else {
					lskipped = BATcount(r1) > 0;
				}
			} else if (only_misses) {
				lskipped = BATcount(r1) > 0;
			} else {
				if (lskipped) {
					/* note, we only get here in
					 * an iteration *after*
					 * lskipped was first set to
					 * true, i.e. we did indeed skip
					 * values in l */
					r1->tseqbase = oid_nil;
				}
				if (nr > 1) {
					r1->tkey = false;
					r1->tseqbase = oid_nil;
				}
			}
			if (nr > 0 && BATcount(r1) > nr)
				r1->trevsorted = false;
		}
	} else if (rcand == NULL && lvars == NULL && sr == NULL &&
		   !nil_matches && !nil_on_miss && !semi && !only_misses &&
		   !BATtvoid(l) && (t == TYPE_int || t == TYPE_lng)) {
		/* special case for a common way of calling this
		 * function */
		const void *restrict base = Tloc(r, 0);

		assert(lcand == NULL);

		if (t == TYPE_int) {
			switch (hsh->width) {
			case BUN2:
				HASHJOIN(int, 2);
				break;
			case BUN4:
				HASHJOIN(int, 4);
				break;
#ifdef BUN8
			case BUN8:
				HASHJOIN(int, 8);
				break;
#endif
			}
		} else {
			/* t == TYPE_lng */
			switch (hsh->width) {
			case BUN2:
				HASHJOIN(lng, 2);
				break;
			case BUN4:
				HASHJOIN(lng, 4);
				break;
#ifdef BUN8
			case BUN8:
				HASHJOIN(lng, 8);
				break;
#endif
			}
		}
	} else {
		for (lo = lstart + l->hseqbase; lstart < lend; lo++) {
			if (BATtvoid(l)) {
				if (BATtdense(l))
					lval = lo - l->hseqbase + l->tseqbase;
			} else {
				v = VALUE(l, lstart);
			}
			lstart++;
			nr = 0;
			if (sr) {
				if (nil_matches || cmp(v, nil) != 0) {
					for (rb = HASHget(hsh, HASHprobe(hsh, v));
					     rb != HASHnil(hsh);
					     rb = HASHgetlink(hsh, rb)) {
						ro = BUNtoid(sr, rb);
						if ((*cmp)(v, BUNtail(ri, ro - r->hseqbase)) != 0)
							continue;
						if (only_misses) {
							nr++;
							break;
						}
						HASHLOOPBODY();
						if (semi)
							break;
					}
				}
			} else {
				switch (t) {
				case TYPE_int:
					if (nil_matches || !is_int_nil(*(const int*)v)) {
						HASHloop_bound_TYPE(ri, hsh, rb, v, rl, rh, int) {
							ro = (oid) (rb - rl + rseq);
							if (only_misses) {
								nr++;
								break;
							}
							HASHLOOPBODY();
							if (semi)
								break;
						}
					}
					break;
				case TYPE_lng:
					if (nil_matches || !is_lng_nil(*(const lng*)v)) {
						HASHloop_bound_TYPE(ri, hsh, rb, v, rl, rh, lng) {
							ro = (oid) (rb - rl + rseq);
							if (only_misses) {
								nr++;
								break;
							}
							HASHLOOPBODY();
							if (semi)
								break;
						}
					}
					break;
#ifdef HAVE_HGE
				case TYPE_hge:
					if (nil_matches || !is_hge_nil(*(const hge*)v)) {
						HASHloop_bound_TYPE(ri, hsh, rb, v, rl, rh, hge) {
							ro = (oid) (rb - rl + rseq);
							if (only_misses) {
								nr++;
								break;
							}
							HASHLOOPBODY();
							if (semi)
								break;
						}
					}
					break;
#endif
				default:
					if (nil_matches || cmp(v, nil) != 0) {
						HASHloop_bound(ri, hsh, rb, v, rl, rh) {
							ro = (oid) (rb - rl + rseq);
							if (only_misses) {
								nr++;
								break;
							}
							HASHLOOPBODY();
							if (semi)
								break;
						}
					}
					break;
				}
			}
			if (nr == 0) {
				if (only_misses) {
					nr = 1;
					if (BUNlast(r1) == BATcapacity(r1)) {
						newcap = BATgrows(r1);
						if (newcap > maxsize)
							newcap = maxsize;
						BATsetcount(r1, BATcount(r1));
						if (BATextend(r1, newcap) != GDK_SUCCEED)
							goto bailout;
					}
					APPEND(r1, lo);
					if (lskipped)
						r1->tseqbase = oid_nil;
				} else if (nil_on_miss) {
					nr = 1;
					r2->tnil = true;
					r2->tnonil = false;
					r2->tkey = false;
					if (BUNlast(r1) == BATcapacity(r1)) {
						newcap = BATgrows(r1);
						if (newcap > maxsize)
							newcap = maxsize;
						BATsetcount(r1, BATcount(r1));
						BATsetcount(r2, BATcount(r2));
						if (BATextend(r1, newcap) != GDK_SUCCEED ||
						    BATextend(r2, newcap) != GDK_SUCCEED)
							goto bailout;
						assert(BATcapacity(r1) == BATcapacity(r2));
					}
					APPEND(r1, lo);
					APPEND(r2, oid_nil);
				} else {
					lskipped = BATcount(r1) > 0;
				}
			} else if (only_misses) {
				lskipped = BATcount(r1) > 0;
			} else {
				if (lskipped) {
					/* note, we only get here in
					 * an iteration *after*
					 * lskipped was first set to
					 * 1, i.e. we did indeed skip
					 * values in l */
					r1->tseqbase = oid_nil;
				}
				if (nr > 1) {
					r1->tkey = false;
					r1->tseqbase = oid_nil;
				}
			}
			if (nr > 0 && BATcount(r1) > nr)
				r1->trevsorted = false;
		}
	}
	if (sr) {
		HEAPfree(&hsh->heap, true);
		GDKfree(hsh);
	}
	/* also set other bits of heap to correct value to indicate size */
	BATsetcount(r1, BATcount(r1));
	if (BATcount(r1) <= 1) {
		r1->tsorted = true;
		r1->trevsorted = true;
		r1->tkey = true;
		r1->tseqbase = 0;
	}
	if (r2) {
		BATsetcount(r2, BATcount(r2));
		assert(BATcount(r1) == BATcount(r2));
		if (BATcount(r2) <= 1) {
			r2->tsorted = true;
			r2->trevsorted = true;
			r2->tkey = true;
			r2->tseqbase = 0;
		}
	}
	if (BATcount(r1) > 0) {
		if (BATtdense(r1))
			r1->tseqbase = ((oid *) r1->theap.base)[0];
		if (r2 && BATtdense(r2))
			r2->tseqbase = ((oid *) r2->theap.base)[0];
	} else {
		r1->tseqbase = 0;
		if (r2) {
			r2->tseqbase = 0;
		}
	}
	ALGODEBUG fprintf(stderr, "#hashjoin(l=%s,r=%s)=(" ALGOBATFMT ","
			  ALGOOPTBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;

  bailout:
	if (sr && hsh) {
		HEAPfree(&hsh->heap, true);
		GDKfree(hsh);
	}
	BBPreclaim(r1);
	BBPreclaim(r2);
	return GDK_FAIL;
}

#define MASK_EQ		1
#define MASK_LT		2
#define MASK_GT		4
#define MASK_LE		(MASK_EQ | MASK_LT)
#define MASK_GE		(MASK_EQ | MASK_GT)
#define MASK_NE		(MASK_LT | MASK_GT)

static gdk_return
thetajoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr, int opcode, BUN estimate, lng t0)
{
	BUN lstart, lend, lcnt;
	const oid *lcand = NULL, *lcandend = NULL;
	BUN rstart, rend, rcnt;
	const oid *rcand = NULL, *rcandend = NULL;
	const char *lvals, *rvals;
	const char *lvars, *rvars;
	int lwidth, rwidth;
	const void *nil = ATOMnilptr(l->ttype);
	int (*cmp)(const void *, const void *) = ATOMcompare(l->ttype);
	const char *vl, *vr;
	const oid *p;
	oid lastr = 0;		/* last value inserted into r2 */
	BUN n, nr;
	BUN newcap;
	oid lo, ro;
	int c;
	bool lskipped = false;	/* whether we skipped values in l */
	lng loff = 0, roff = 0;
	oid lval = oid_nil, rval = oid_nil;

	ALGODEBUG fprintf(stderr, "#thetajoin(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ",sl=" ALGOOPTBATFMT ","
			  "sr=" ALGOOPTBATFMT ",op=%s%s%s)\n",
			  ALGOBATPAR(l), ALGOBATPAR(r), ALGOOPTBATPAR(sl), ALGOOPTBATPAR(sr),
			  opcode & MASK_LT ? "<" : "",
			  opcode & MASK_GT ? ">" : "",
			  opcode & MASK_EQ ? "=" : "");

	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(sl == NULL || sl->tsorted);
	assert(sr == NULL || sr->tsorted);
	assert((opcode & (MASK_EQ | MASK_LT | MASK_GT)) != 0);

	CANDINIT(l, sl, lstart, lend, lcnt, lcand, lcandend);
	CANDINIT(r, sr, rstart, rend, rcnt, rcand, rcandend);
	lcnt = lcand ? (BUN) (lcandend - lcand) : lend - lstart;
	rcnt = rcand ? (BUN) (rcandend - rcand) : rend - rstart;

	lvals = BATtvoid(l) ? NULL : (const char *) Tloc(l, 0);
	rvals = BATtvoid(r) ? NULL : (const char *) Tloc(r, 0);
	if (l->tvarsized && l->ttype) {
		assert(r->tvarsized && r->ttype);
		lvars = l->tvheap->base;
		rvars = r->tvheap->base;
	} else {
		assert(!r->tvarsized || !r->ttype);
		lvars = rvars = NULL;
	}
	lwidth = l->twidth;
	rwidth = r->twidth;

	if (BATtvoid(l)) {
		if (!BATtdense(l)) {
			/* trivial: nils don't match anything */
			return nomatch(r1p, r2p, l, r, lstart, lend, lcand,
				       lcandend, false, false, "thetajoin", t0);
		}
		if (lcand) {
			lstart = 0;
			lend = (BUN) (lcandend - lcand);
			lvals = (const char *) lcand;
			lcand = NULL;
			lwidth = SIZEOF_OID;
		}
		loff = (lng) l->tseqbase - (lng) l->hseqbase;
	}
	if (BATtvoid(r)) {
		if (!BATtdense(r)) {
			/* trivial: nils don't match anything */
			return nomatch(r1p, r2p, l, r, lstart, lend, lcand,
				       lcandend, false, false, "thetajoin", t0);
		}
		if (rcand) {
			rstart = 0;
			rend = (BUN) (rcandend - rcand);
			rvals = (const char *) rcand;
			rcand = NULL;
			rwidth = SIZEOF_OID;
		}
		roff = (lng) r->tseqbase - (lng) r->hseqbase;
	}
	assert(lvals != NULL || lcand == NULL);
	assert(rvals != NULL || rcand == NULL);

	BUN maxsize = joininitresults(r1p, r2p, lcnt, rcnt, false, false,
				      false, false, false, estimate);
	if (maxsize == BUN_NONE)
		return GDK_FAIL;
	BAT *r1 = *r1p;
	BAT *r2 = *r2p;

	r1->tkey = true;
	r1->tsorted = true;
	r1->trevsorted = true;
	r2->tkey = true;
	r2->tsorted = true;
	r2->trevsorted = true;

	/* nested loop implementation for theta join */
	for (;;) {
		if (lcand) {
			if (lcand == lcandend)
				break;
			lo = *lcand++;
			vl = VALUE(l, lo - l->hseqbase);
		} else {
			if (lstart == lend)
				break;
			if (lvals) {
				vl = VALUE(l, lstart);
				if (loff != 0) {
					lval = (oid) (*(const oid *)vl + loff);
					vl = (const char *) &lval;
				}
			} else {
				lval = lstart + l->tseqbase;
				vl = (const char *) &lval;
			}
			lo = lstart++ + l->hseqbase;
		}
		nr = 0;
		if (cmp(vl, nil) != 0) {
			p = rcand;
			n = rstart;
			for (;;) {
				if (rcand) {
					if (p == rcandend)
						break;
					ro = *p++;
					vr = VALUE(r, ro - r->hseqbase);
				} else {
					if (n == rend)
						break;
					if (rvals) {
						vr = VALUE(r, n);
						if (roff != 0) {
							rval = (oid) (*(const oid *)vr + roff);
							vr = (const char *) &rval;
						}
					} else {
						rval = n + r->tseqbase;
						vr = (const char *) &rval;
					}
					ro = n++ + r->hseqbase;
				}
				if (cmp(vr, nil) == 0)
					continue;
				c = cmp(vl, vr);
				if (!((opcode & MASK_LT && c < 0) ||
				      (opcode & MASK_GT && c > 0) ||
				      (opcode & MASK_EQ && c == 0)))
					continue;
				if (BUNlast(r1) == BATcapacity(r1)) {
					newcap = BATgrows(r1);
					if (newcap > maxsize)
						newcap = maxsize;
					BATsetcount(r1, BATcount(r1));
					BATsetcount(r2, BATcount(r2));
					if (BATextend(r1, newcap) != GDK_SUCCEED ||
					    BATextend(r2, newcap) != GDK_SUCCEED)
						goto bailout;
					assert(BATcapacity(r1) == BATcapacity(r2));
				}
				if (BATcount(r2) > 0) {
					if (lastr + 1 != ro)
						r2->tseqbase = oid_nil;
					if (nr == 0) {
						r1->trevsorted = false;
						if (lastr > ro) {
							r2->tsorted = false;
							r2->tkey = false;
						} else if (lastr < ro) {
							r2->trevsorted = false;
						} else {
							r2->tkey = false;
						}
					}
				}
				APPEND(r1, lo);
				APPEND(r2, ro);
				lastr = ro;
				nr++;
			}
		}
		if (nr > 1) {
			r1->tkey = false;
			r1->tseqbase = oid_nil;
			r2->trevsorted = false;
		} else if (nr == 0) {
			lskipped = BATcount(r1) > 0;
		} else if (lskipped) {
			r1->tseqbase = oid_nil;
		}
	}
	assert(BATcount(r1) == BATcount(r2));
	/* also set other bits of heap to correct value to indicate size */
	BATsetcount(r1, BATcount(r1));
	BATsetcount(r2, BATcount(r2));
	if (BATcount(r1) > 0) {
		if (BATtdense(r1))
			r1->tseqbase = ((oid *) r1->theap.base)[0];
		if (BATtdense(r2))
			r2->tseqbase = ((oid *) r2->theap.base)[0];
	} else {
		r1->tseqbase = r2->tseqbase = 0;
	}
	ALGODEBUG fprintf(stderr, "#thetajoin(l=%s,r=%s)=(" ALGOBATFMT "," ALGOBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;

  bailout:
	BBPreclaim(r1);
	BBPreclaim(r2);
	return GDK_FAIL;
}

static gdk_return
bandjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	 const void *c1, const void *c2, bool li, bool hi, BUN estimate,
	 lng t0)
{
	BUN lstart, lend, lcnt;
	const oid *lcand = NULL, *lcandend = NULL;
	BUN rstart, rend, rcnt;
	const oid *rcand = NULL, *rcandend = NULL;
	const char *lvals, *rvals;
	int lwidth, rwidth;
	int t;
	const void *nil = ATOMnilptr(l->ttype);
	int (*cmp)(const void *, const void *) = ATOMcompare(l->ttype);
	const char *vl, *vr;
	const oid *p;
	oid lastr = 0;		/* last value inserted into r2 */
	BUN n, nr;
	BUN newcap;
	oid lo, ro;
	bool lskipped = false;	/* whether we skipped values in l */
	BUN nils = 0;		/* needed for XXX_WITH_CHECK macros */

	assert(ATOMtype(l->ttype) == ATOMtype(r->ttype));
	assert(sl == NULL || sl->tsorted);
	assert(sr == NULL || sr->tsorted);

	t = ATOMtype(l->ttype);
	t = ATOMbasetype(t);

	switch (t) {
	case TYPE_bte:
		if (is_bte_nil(*(const bte *)c1) ||
		    is_bte_nil(*(const bte *)c2) ||
		    -*(const bte *)c1 > *(const bte *)c2 ||
		    ((!hi || !li) && -*(const bte *)c1 == *(const bte *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
	case TYPE_sht:
		if (is_sht_nil(*(const sht *)c1) ||
		    is_sht_nil(*(const sht *)c2) ||
		    -*(const sht *)c1 > *(const sht *)c2 ||
		    ((!hi || !li) && -*(const sht *)c1 == *(const sht *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
	case TYPE_int:
		if (is_int_nil(*(const int *)c1) ||
		    is_int_nil(*(const int *)c2) ||
		    -*(const int *)c1 > *(const int *)c2 ||
		    ((!hi || !li) && -*(const int *)c1 == *(const int *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
	case TYPE_lng:
		if (is_lng_nil(*(const lng *)c1) ||
		    is_lng_nil(*(const lng *)c2) ||
		    -*(const lng *)c1 > *(const lng *)c2 ||
		    ((!hi || !li) && -*(const lng *)c1 == *(const lng *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
#ifdef HAVE_HGE
	case TYPE_hge:
		if (is_hge_nil(*(const hge *)c1) ||
		    is_hge_nil(*(const hge *)c2) ||
		    -*(const hge *)c1 > *(const hge *)c2 ||
		    ((!hi || !li) && -*(const hge *)c1 == *(const hge *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
#endif
	case TYPE_flt:
		if (is_flt_nil(*(const flt *)c1) ||
		    is_flt_nil(*(const flt *)c2) ||
		    -*(const flt *)c1 > *(const flt *)c2 ||
		    ((!hi || !li) && -*(const flt *)c1 == *(const flt *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
	case TYPE_dbl:
		if (is_dbl_nil(*(const dbl *)c1) ||
		    is_dbl_nil(*(const dbl *)c2) ||
		    -*(const dbl *)c1 > *(const dbl *)c2 ||
		    ((!hi || !li) && -*(const dbl *)c1 == *(const dbl *)c2))
			return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
				       false, false, "bandjoin", t0);
		break;
	default:
		GDKerror("BATbandjoin: unsupported type\n");
		return GDK_FAIL;
	}

	CANDINIT(l, sl, lstart, lend, lcnt, lcand, lcandend);
	CANDINIT(r, sr, rstart, rend, rcnt, rcand, rcandend);
	lcnt = lcand ? (BUN) (lcandend - lcand) : lend - lstart;
	rcnt = rcand ? (BUN) (rcandend - rcand) : rend - rstart;

	if (lcnt == 0 || rcnt == 0)
		return nomatch(r1p, r2p, l, r, 0, 0, NULL, NULL,
			       false, false, "bandjoin", t0);

	BUN maxsize = joininitresults(r1p, r2p, lcnt, rcnt, false, false,
				      false, false, false, estimate);
	if (maxsize == BUN_NONE)
		return GDK_FAIL;
	BAT *r1 = *r1p;
	BAT *r2 = *r2p;

	lvals = (const char *) Tloc(l, 0);
	rvals = (const char *) Tloc(r, 0);
	assert(!r->tvarsized);
	lwidth = l->twidth;
	rwidth = r->twidth;

	assert(lvals != NULL);
	assert(rvals != NULL);

	r1->tkey = true;
	r1->tsorted = true;
	r1->trevsorted = true;
	r2->tkey = true;
	r2->tsorted = true;
	r2->trevsorted = true;

	/* nested loop implementation for band join */
	for (;;) {
		if (lcand) {
			if (lcand == lcandend)
				break;
			lo = *lcand++;
			vl = FVALUE(l, lo - l->hseqbase);
		} else {
			if (lstart == lend)
				break;
			vl = FVALUE(l, lstart);
			lo = lstart++ + l->hseqbase;
		}
		if (cmp(vl, nil) == 0)
			continue;
		nr = 0;
		p = rcand;
		n = rstart;
		for (;;) {
			if (rcand) {
				if (p == rcandend)
					break;
				ro = *p++;
				vr = FVALUE(r, ro - r->hseqbase);
			} else {
				if (n == rend)
					break;
				vr = FVALUE(r, n);
				ro = n++ + r->hseqbase;
			}
			switch (ATOMtype(l->ttype)) {
			case TYPE_bte: {
				sht v1 = (sht) *(const bte *) vr, v2;

				if (is_bte_nil(v1))
					continue;
				v2 = v1;
				v1 -= *(const bte *)c1;
				if (*(const bte *)vl <= v1 &&
				    (!li || *(const bte *)vl != v1))
					continue;
				v2 += *(const bte *)c2;
				if (*(const bte *)vl >= v2 &&
				    (!hi || *(const bte *)vl != v2))
					continue;
				break;
			}
			case TYPE_sht: {
				int v1 = (int) *(const sht *) vr, v2;

				if (is_sht_nil(v1))
					continue;
				v2 = v1;
				v1 -= *(const sht *)c1;
				if (*(const sht *)vl <= v1 &&
				    (!li || *(const sht *)vl != v1))
					continue;
				v2 += *(const sht *)c2;
				if (*(const sht *)vl >= v2 &&
				    (!hi || *(const sht *)vl != v2))
					continue;
				break;
			}
			case TYPE_int: {
				lng v1 = (lng) *(const int *) vr, v2;

				if (is_int_nil(v1))
					continue;
				v2 = v1;
				v1 -= *(const int *)c1;
				if (*(const int *)vl <= v1 &&
				    (!li || *(const int *)vl != v1))
					continue;
				v2 += *(const int *)c2;
				if (*(const int *)vl >= v2 &&
				    (!hi || *(const int *)vl != v2))
					continue;
				break;
			}
#ifdef HAVE_HGE
			case TYPE_lng: {
				hge v1 = (hge) *(const lng *) vr, v2;

				if (is_lng_nil(v1))
					continue;
				v2 = v1;
				v1 -= *(const lng *)c1;
				if (*(const lng *)vl <= v1 &&
				    (!li || *(const lng *)vl != v1))
					continue;
				v2 += *(const lng *)c2;
				if (*(const lng *)vl >= v2 &&
				    (!hi || *(const lng *)vl != v2))
					continue;
				break;
			}
#else
#ifdef HAVE___INT128
			case TYPE_lng: {
				__int128 v1 = (__int128) *(const lng *) vr, v2;

				if (is_lng_nil(v1))
					continue;
				v2 = v1;
				v1 -= *(const lng *)c1;
				if (*(const lng *)vl <= v1 &&
				    (!li || *(const lng *)vl != v1))
					continue;
				v2 += *(const lng *)c2;
				if (*(const lng *)vl >= v2 &&
				    (!hi || *(const lng *)vl != v2))
					continue;
				break;
			}
#else
			case TYPE_lng: {
				lng v1, v2;
				bool abort_on_error = true;

				if (is_lng_nil(*(const lng *)vr))
					continue;
				SUB_WITH_CHECK(*(const lng *)vr,
					       *(const lng *)c1,
					       lng, v1,
					       GDK_lng_max,
					       do{if(*(const lng*)c1<0)goto nolmatch;else goto lmatch1;}while(false));
				if (*(const lng *)vl <= v1 &&
				    (!li || *(const lng *)vl != v1))
					continue;
			  lmatch1:
				ADD_WITH_CHECK(*(const lng *)vr,
					       *(const lng *)c2,
					       lng, v2,
					       GDK_lng_max,
					       do{if(*(const lng*)c2>0)goto nolmatch;else goto lmatch2;}while(false));
				if (*(const lng *)vl >= v2 &&
				    (!hi || *(const lng *)vl != v2))
					continue;
			  lmatch2:
				break;
			  nolmatch:
				continue;
			}
#endif
#endif
#ifdef HAVE_HGE
			case TYPE_hge: {
				hge v1, v2;
				bool abort_on_error = true;

				if (is_hge_nil(*(const hge *)vr))
					continue;
				SUB_WITH_CHECK(*(const hge *)vr,
					       *(const hge *)c1,
					       hge, v1,
					       GDK_hge_max,
					       do{if(*(const hge*)c1<0)goto nohmatch;else goto hmatch1;}while(false));
				if (*(const hge *)vl <= v1 &&
				    (!li || *(const hge *)vl != v1))
					continue;
			  hmatch1:
				ADD_WITH_CHECK(*(const hge *)vr,
					       *(const hge *)c2,
					       hge, v2,
					       GDK_hge_max,
					       do{if(*(const hge*)c2>0)goto nohmatch;else goto hmatch2;}while(false));
				if (*(const hge *)vl >= v2 &&
				    (!hi || *(const hge *)vl != v2))
					continue;
			  hmatch2:
				break;
			  nohmatch:
				continue;
			}
#endif
			case TYPE_flt: {
				dbl v1 = (dbl) *(const flt *) vr, v2;

				if (is_flt_nil(v1))
					continue;
				v2 = v1;
				v1 -= *(const flt *)c1;
				if (*(const flt *)vl <= v1 &&
				    (!li || *(const flt *)vl != v1))
					continue;
				v2 += *(const flt *)c2;
				if (*(const flt *)vl >= v2 &&
				    (!hi || *(const flt *)vl != v2))
					continue;
				break;
			}
			case TYPE_dbl: {
				dbl v1, v2;
				bool abort_on_error = true;

				if (is_dbl_nil(*(const dbl *)vr))
					continue;
				SUB_WITH_CHECK(*(const dbl *)vr,
					       *(const dbl *)c1,
					       dbl, v1,
					       GDK_dbl_max,
					       do{if(*(const dbl*)c1<0)goto nodmatch;else goto dmatch1;}while(false));
				if (*(const dbl *)vl <= v1 &&
				    (!li || *(const dbl *)vl != v1))
					continue;
			  dmatch1:
				ADD_WITH_CHECK(*(const dbl *)vr,
					       *(const dbl *)c2,
					       dbl, v2,
					       GDK_dbl_max,
					       do{if(*(const dbl*)c2>0)goto nodmatch;else goto dmatch2;}while(false));
				if (*(const dbl *)vl >= v2 &&
				    (!hi || *(const dbl *)vl != v2))
					continue;
			  dmatch2:
				break;
			  nodmatch:
				continue;
			}
			}
			if (BUNlast(r1) == BATcapacity(r1)) {
				newcap = BATgrows(r1);
				if (newcap > maxsize)
					newcap = maxsize;
				BATsetcount(r1, BATcount(r1));
				BATsetcount(r2, BATcount(r2));
				if (BATextend(r1, newcap) != GDK_SUCCEED ||
				    BATextend(r2, newcap) != GDK_SUCCEED)
					goto bailout;
				assert(BATcapacity(r1) == BATcapacity(r2));
			}
			if (BATcount(r2) > 0) {
				if (lastr + 1 != ro)
					r2->tseqbase = oid_nil;
				if (nr == 0) {
					r1->trevsorted = false;
					if (lastr > ro) {
						r2->tsorted = false;
						r2->tkey = false;
					} else if (lastr < ro) {
						r2->trevsorted = false;
					} else {
						r2->tkey = false;
					}
				}
			}
			APPEND(r1, lo);
			APPEND(r2, ro);
			lastr = ro;
			nr++;
		}
		if (nr > 1) {
			r1->tkey = false;
			r1->tseqbase = oid_nil;
			r2->trevsorted = false;
		} else if (nr == 0) {
			lskipped = BATcount(r1) > 0;
		} else if (lskipped) {
			r1->tseqbase = oid_nil;
		}
	}
	assert(BATcount(r1) == BATcount(r2));
	/* also set other bits of heap to correct value to indicate size */
	BATsetcount(r1, BATcount(r1));
	BATsetcount(r2, BATcount(r2));
	if (BATcount(r1) > 0) {
		if (BATtdense(r1))
			r1->tseqbase = ((oid *) r1->theap.base)[0];
		if (BATtdense(r2))
			r2->tseqbase = ((oid *) r2->theap.base)[0];
	} else {
		r1->tseqbase = r2->tseqbase = 0;
	}
	ALGODEBUG fprintf(stderr, "#BATbandjoin(l=%s,r=%s)=(" ALGOBATFMT "," ALGOBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;

  bailout:
	BBPreclaim(r1);
	BBPreclaim(r2);
	return GDK_FAIL;
}

/* small ordered right, dense left, oid's only, do fetches */
static gdk_return
fetchjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, lng t0)
{
	oid lo = l->tseqbase, hi = lo + BATcount(l);
	BUN b = SORTfndfirst(r, &lo), e = SORTfndlast(r, &hi), p;

	ALGODEBUG fprintf(stderr, "#fetchjoin(l=" ALGOBATFMT ","
			  "r=" ALGOBATFMT ")\n",
			  ALGOBATPAR(l), ALGOBATPAR(r));

	if (r2p &&
	    (*r2p = BATdense(0, e == b ? 0 : r->hseqbase + b, e - b)) == NULL) {
		return GDK_FAIL;
	}
	BAT *r2 = r2p ? *r2p : NULL;
	BAT *r1 = *r1p = COLnew(0, TYPE_oid, e - b, TRANSIENT);
	if (r1 == NULL) {
		if (r2p)
			BBPreclaim(*r2p);
		return GDK_FAIL;
	}
	oid *op = (oid *) Tloc(r1, 0);
	const oid *rp = (const oid *) Tloc(r, 0);
	for (p = b; p < e; p++) {
		*op++ = rp[p] + l->hseqbase - l->tseqbase;
	}
	BATsetcount(r1, e - b);
	r1->tkey = r->tkey;
	r1->tsorted = r->tsorted || e - b <= 1;
	r1->trevsorted = r->trevsorted || e - b <= 1;
	r1->tseqbase = e == b ? 0 : e - b == 1 ? *(const oid *)Tloc(r1, 0) : oid_nil;
	ALGODEBUG fprintf(stderr, "#fetchjoin(l=%s,r=%s)=(" ALGOBATFMT "," ALGOOPTBATFMT ") " LLFMT "us\n",
			  BATgetId(l), BATgetId(r),
			  ALGOBATPAR(r1), ALGOOPTBATPAR(r2),
			  GDKusec() - t0);
	return GDK_SUCCEED;
}


/* Make the implementation choices for various left joins. */
static gdk_return
leftjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	 bool nil_matches, bool nil_on_miss, bool semi, bool only_misses,
	 BUN estimate, const char *name, lng t0)
{
	BUN lstart, lend, lcnt;
	const oid *lcand, *lcandend;
	BUN rstart, rend, rcnt;
	const oid *rcand, *rcandend;
	bool phash = false;

	/* only_misses implies left output only */
	assert(!only_misses || r2p == NULL);
	/* only no right output allowed for semijoin and diff */
	assert(r2p != NULL || (semi | only_misses));
	/* if nil_on_miss is set, we really need a right output */
	assert(!nil_on_miss || r2p != NULL);
	*r1p = NULL;
	if (r2p)
		*r2p = NULL;
	if (joinparamcheck(l, r, NULL, sl, sr, name) != GDK_SUCCEED)
		return GDK_FAIL;

	CANDINIT(l, sl, lstart, lend, lcnt, lcand, lcandend);
	CANDINIT(r, sr, rstart, rend, rcnt, rcand, rcandend);
	lcnt = lcand ? (BUN) (lcandend - lcand) : lend - lstart;
	rcnt = rcand ? (BUN) (rcandend - rcand) : rend - rstart;

	if (lcnt == 0 || (!only_misses && !nil_on_miss && rcnt == 0))
		return nomatch(r1p, r2p, l, r, lstart, lend, lcand, lcandend,
			       nil_on_miss, only_misses, "leftjoin", t0);

	if (!nil_on_miss && !semi && !only_misses &&
	    (lcnt == 1 || (BATordered(l) && BATordered_rev(l)))) {
		/* single value to join, use select */
		return selectjoin(r1p, r2p, l, r, sl, sr,
				  lstart, lend, lcnt, lcand, lcandend,
				  nil_matches, t0, false);
	} else if (BATtdense(r) && (sr == NULL || BATtdense(sr)) &&
		   lcnt > 0 && rcnt > 0) {
		/* use special implementation for dense right-hand side */
		return mergejoin_void(r1p, r2p, l, r, sl, sr,
				      lstart, lend, lcnt, lcand, lcandend,
				      nil_on_miss, only_misses, t0, false);
	} else if ((BATordered(r) || BATordered_rev(r))
		   && (BATordered(l)
		       || BATordered_rev(l)
		       || BATtdense(r)
		       || lcnt < 1024
		       || BATcount(r) * (Tsize(r) + (r->tvheap ? r->tvheap->size : 0) + 2 * sizeof(BUN)) > GDK_mem_maxsize / (GDKnr_threads ? GDKnr_threads : 1)))
		return mergejoin(r1p, r2p, l, r, sl, sr,
				 lstart, lend, lcnt, lcand, lcandend,
				 rstart, rend, rcnt, rcand, rcandend,
				 nil_matches, nil_on_miss, semi, only_misses,
				 estimate, t0, false);
	if (BATtdense(l)
	    && ATOMtype(l->ttype) == TYPE_oid
	    && sl == NULL
	    && sr == NULL
	    && !semi
	    && !nil_matches
	    && !only_misses
	    && (rcnt * 1024) < lcnt
	    && BATordered(r))
		return fetchjoin(r1p, r2p, l, r, t0);
	phash = sr == NULL &&
		VIEWtparent(r) != 0 &&
		BATcount(BBPquickdesc(VIEWtparent(r), false)) == BATcount(r);
	return hashjoin(r1p, r2p, l, r, sl, sr,
			lstart, lend, lcnt, lcand, lcandend,
			rstart, rend, rcnt, rcand, rcandend,
			nil_matches, nil_on_miss, semi, only_misses,
			estimate, t0, false, phash, "leftjoin");
}

/* Perform an equi-join over l and r.  Returns two new, aligned, bats
 * with the oids of matching tuples.  The result is in the same order
 * as l (i.e. r1 is sorted). */
gdk_return
BATleftjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr, bool nil_matches, BUN estimate)
{
	return leftjoin(r1p, r2p, l, r, sl, sr, nil_matches,
			false, false, false, estimate, "BATleftjoin",
			GDKdebug & ALGOMASK ? GDKusec() : 0);
}

/* Performs a left outer join over l and r.  Returns two new, aligned,
 * bats with the oids of matching tuples, or the oid in the first
 * output bat and nil in the second output bat if the value in l does
 * not occur in r.  The result is in the same order as l (i.e. r1 is
 * sorted). */
gdk_return
BATouterjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr, bool nil_matches, BUN estimate)
{
	return leftjoin(r1p, r2p, l, r, sl, sr, nil_matches,
			true, false, false, estimate, "BATouterjoin",
			GDKdebug & ALGOMASK ? GDKusec() : 0);
}

/* Perform a semi-join over l and r.  Returns one or two new, bats
 * with the oids of matching tuples.  The result is in the same order
 * as l (i.e. r1 is sorted).  If a single bat is returned, it is a
 * candidate list. */
gdk_return
BATsemijoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr, bool nil_matches, BUN estimate)
{
	return leftjoin(r1p, r2p, l, r, sl, sr, nil_matches,
			false, true, false, estimate, "BATsemijoin",
			GDKdebug & ALGOMASK ? GDKusec() : 0);
}

/* Return a candidate list with the list of rows in l whose value also
 * occurs in r.  This is just the left output of a semi-join. */
BAT *
BATintersect(BAT *l, BAT *r, BAT *sl, BAT *sr, bool nil_matches, BUN estimate)
{
	BAT *bn;

	if (leftjoin(&bn, NULL, l, r, sl, sr, nil_matches,
		     false, true, false, estimate, "BATintersect",
		     GDKdebug & ALGOMASK ? GDKusec() : 0) == GDK_SUCCEED)
		return virtualize(bn);
	return NULL;
}

/* Return the difference of l and r.  The result is a BAT with the
 * oids of those values in l that do not occur in r.  This is what you
 * might call an anti-semi-join.  The result is a candidate list. */
BAT *
BATdiff(BAT *l, BAT *r, BAT *sl, BAT *sr, bool nil_matches, BUN estimate)
{
	BAT *bn;

	if (leftjoin(&bn, NULL, l, r, sl, sr, nil_matches,
		     false, false, true, estimate, "BATdiff",
		     GDKdebug & ALGOMASK ? GDKusec() : 0) == GDK_SUCCEED)
		return virtualize(bn);
	return NULL;
}

gdk_return
BATthetajoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr, int op, bool nil_matches, BUN estimate)
{
	int opcode = 0;
	lng t0 = 0;

	/* encode operator as a bit mask into opcode */
	switch (op) {
	case JOIN_EQ:
		return BATjoin(r1p, r2p, l, r, sl, sr, nil_matches, estimate);
	case JOIN_NE:
		opcode = MASK_NE;
		break;
	case JOIN_LT:
		opcode = MASK_LT;
		break;
	case JOIN_LE:
		opcode = MASK_LE;
		break;
	case JOIN_GT:
		opcode = MASK_GT;
		break;
	case JOIN_GE:
		opcode = MASK_GE;
		break;
	default:
		GDKerror("BATthetajoin: unknown operator %d.\n", op);
		return GDK_FAIL;
	}

	ALGODEBUG t0 = GDKusec();
	*r1p = NULL;
	*r2p = NULL;
	if (joinparamcheck(l, r, NULL, sl, sr, "BATthetajoin") != GDK_SUCCEED)
		return GDK_FAIL;

	return thetajoin(r1p, r2p, l, r, sl, sr, opcode, estimate, t0);
}

gdk_return
BATjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr, bool nil_matches, BUN estimate)
{
	BUN lstart, lend, lcnt;
	const oid *lcand, *lcandend;
	BUN rstart, rend, rcnt;
	const oid *rcand, *rcandend;
	BUN lsize, rsize;
	bool lhash = false, rhash = false;
	bool plhash = false, prhash = false;
	BUN lslots = 0, rslots = 0;
	bool swap;
	bat parent;
	size_t mem_size;
	lng t0 = 0;
	const char *reason = "";

	ALGODEBUG t0 = GDKusec();

	CANDINIT(l, sl, lstart, lend, lcnt, lcand, lcandend);
	CANDINIT(r, sr, rstart, rend, rcnt, rcand, rcandend);
	lcnt = lcand ? (BUN) (lcandend - lcand) : lend - lstart;
	rcnt = rcand ? (BUN) (rcandend - rcand) : rend - rstart;

	*r1p = NULL;
	*r2p = NULL;
	if (joinparamcheck(l, r, NULL, sl, sr, "BATjoin") != GDK_SUCCEED)
		return GDK_FAIL;

	if (lcnt == 0 || rcnt == 0)
		return nomatch(r1p, r2p, l, r, lstart, lend, lcand, lcandend,
			       false, false, "BATjoin", t0);

	swap = false;

	/* some statistics to help us decide */
	lsize = (BUN) (BATcount(l) * (Tsize(l)) + (l->tvheap ? l->tvheap->size : 0) + 2 * sizeof(BUN));
	rsize = (BUN) (BATcount(r) * (Tsize(r)) + (r->tvheap ? r->tvheap->size : 0) + 2 * sizeof(BUN));
	mem_size = GDK_mem_maxsize / (GDKnr_threads ? GDKnr_threads : 1);

	if (lcnt == 1 || (BATordered(l) && BATordered_rev(l))) {
		/* single value to join, use select */
		return selectjoin(r1p, r2p, l, r, sl, sr,
				  lstart, lend, lcnt, lcand, lcandend,
				  nil_matches, t0, false);
	} else if (rcnt == 1 || (BATordered(r) && BATordered_rev(r))) {
		/* single value to join, use select */
		return selectjoin(r2p, r1p, r, l, sr, sl,
				  rstart, rend, rcnt, rcand, rcandend,
				  nil_matches, t0, true);
	} else if (BATtdense(r) && rcand == NULL) {
		/* use special implementation for dense right-hand side */
		return mergejoin_void(r1p, r2p, l, r, sl, sr,
				      lstart, lend, lcnt, lcand, lcandend,
				      false, false, t0, false);
	} else if (BATtdense(l) && lcand == NULL) {
		/* use special implementation for dense right-hand side */
		return mergejoin_void(r2p, r1p, r, l, sr, sl,
				      rstart, rend, rcnt, rcand, rcandend,
				      false, false, t0, true);
	} else if ((BATordered(l) || BATordered_rev(l)) &&
		   (BATordered(r) || BATordered_rev(r))) {
		/* both sorted */
		return mergejoin(r1p, r2p, l, r, sl, sr,
				 lstart, lend, lcnt, lcand, lcandend,
				 rstart, rend, rcnt, rcand, rcandend,
				 nil_matches, false, false, false,
				 estimate, t0, false);
	}
	if (sl == NULL) {
		lhash = BATcheckhash(l);
		if (lhash) {
			lslots = ((size_t *) l->thash->heap.base)[5];
		} else if ((parent = VIEWtparent(l)) != 0) {
			BAT *b = BBPdescriptor(parent);
			/* use hash on parent if the average chain
			 * length times the number of required probes
			 * is less than the cost for creating and
			 * probing a new hash on the view */
			if (BATcheckhash(b)) {
				lslots = ((size_t *) b->thash->heap.base)[5];
				lhash = (BATcount(b) == BATcount(l) ||
					 BATcount(b) / lslots * rcnt < lcnt + rcnt);
			}
			plhash = lhash;
		}
	} else if (BATtdense(sl) && BATcheckhash(l)) {
		lslots = ((size_t *) l->thash->heap.base)[5];
		lhash = BATcount(l) / lslots * rcnt < lcnt + rcnt;
	}
	if (sr == NULL) {
		rhash = BATcheckhash(r);
		if (rhash) {
			rslots = ((size_t *) r->thash->heap.base)[5];
		} else if ((parent = VIEWtparent(r)) != 0) {
			BAT *b = BBPdescriptor(parent);
			/* use hash on parent if the average chain
			 * length times the number of required probes
			 * is less than the cost for creating and
			 * probing a new hash on the view */
			if (BATcheckhash(b)) {
				rslots = ((size_t *) b->thash->heap.base)[5];
				rhash = (BATcount(b) == BATcount(r) ||
					 BATcount(b) / rslots * lcnt < lcnt + rcnt);
			}
			prhash = rhash;
		}
	} else if (BATtdense(sr) && BATcheckhash(r)) {
		rslots = ((size_t *) r->thash->heap.base)[5];
		rhash = BATcount(r) / rslots * rcnt < lcnt + rcnt;
	}
	if (lhash && rhash) {
		if (lcnt == lslots && rcnt == rslots) {
			/* both perfect hashes, smallest on right */
			swap = lcnt < rcnt;
		} else if (lcnt == lslots) {
			/* left is perfect (right isn't): swap */
			swap = true;
		} else if (rcnt != rslots) {
			/* neither is perfect, shortest chains on right */
			swap = lcnt / lslots < rcnt / rslots;
		} /* else: right is perfect */
		reason = "both have hash";
	} else if (lhash) {
		/* only left has hash, swap */
		swap = true;
		reason = "left has hash";
	} else if (rhash) {
		/* only right has hash, don't swap */
		swap = false;
		reason = "right has hash";
	} else if ((BATordered(l) || BATordered_rev(l)) &&
		   (BATtvoid(l) || rcnt < 1024 || MIN(lsize, rsize) > mem_size)) {
		/* only left is sorted, swap; but only if right is
		 * "large" and the smaller of the two isn't too large
		 * (i.e. prefer hash over binary search, but only if
		 * the hash table doesn't cause thrashing) */
		return mergejoin(r2p, r1p, r, l, sr, sl,
				 rstart, rend, rcnt, rcand, rcandend,
				 lstart, lend, lcnt, lcand, lcandend,
				 nil_matches, false, false, false,
				 estimate, t0, true);
	} else if ((BATordered(r) || BATordered_rev(r)) &&
		   (BATtvoid(r) || lcnt < 1024 || MIN(lsize, rsize) > mem_size)) {
		/* only right is sorted, don't swap; but only if left
		 * is "large" and the smaller of the two isn't too
		 * large (i.e. prefer hash over binary search, but
		 * only if the hash table doesn't cause thrashing) */
		return mergejoin(r1p, r2p, l, r, sl, sr,
				 lstart, lend, lcnt, lcand, lcandend,
				 rstart, rend, rcnt, rcand, rcandend,
				 nil_matches, false, false, false,
				 estimate, t0, false);
	} else if (l->batPersistence == PERSISTENT &&
		   r->batPersistence != PERSISTENT) {
		/* l is persistent and r is not, create hash on l
		 * since it may be reused */
		swap = true;
		reason = "left is persistent";
	} else if (l->batPersistence != PERSISTENT &&
		   r->batPersistence == PERSISTENT) {
		/* l is not persistent but r is, create hash on r
		 * since it may be reused */
		/* nothing */;
		reason = "right is persistent";
	} else if (lcnt < rcnt) {
		/* no hashes, not sorted, create hash on smallest BAT */
		swap = true;
		reason = "left is smaller";
	}
	if (swap) {
		return hashjoin(r2p, r1p, r, l, sr, sl,
				rstart, rend, rcnt, rcand, rcandend,
				lstart, lend, lcnt, lcand, lcandend,
				nil_matches, false, false, false,
				estimate, t0, true, plhash, reason);
	} else {
		return hashjoin(r1p, r2p, l, r, sl, sr,
				lstart, lend, lcnt, lcand, lcandend,
				rstart, rend, rcnt, rcand, rcandend,
				nil_matches, false, false, false,
				estimate, t0, false, prhash, reason);
	}
}

gdk_return
BATbandjoin(BAT **r1p, BAT **r2p, BAT *l, BAT *r, BAT *sl, BAT *sr,
	       const void *c1, const void *c2, bool li, bool hi, BUN estimate)
{
	lng t0 = 0;

	ALGODEBUG t0 = GDKusec();

	ALGODEBUG fprintf(stderr, "#BATbandjoin("
			  "l=" ALGOBATFMT ",r=" ALGOBATFMT ","
			  "sl=" ALGOOPTBATFMT ",sr=" ALGOOPTBATFMT ")\n",
			  ALGOBATPAR(l), ALGOBATPAR(r),
			  ALGOOPTBATPAR(sl), ALGOOPTBATPAR(sr));

	*r1p = NULL;
	*r2p = NULL;
	if (joinparamcheck(l, r, NULL, sl, sr, "BATbandjoin") != GDK_SUCCEED)
		return GDK_FAIL;
	return bandjoin(r1p, r2p, l, r, sl, sr, c1, c2, li, hi, estimate, t0);
}

gdk_return
BATrangejoin(BAT **r1p, BAT **r2p, BAT *l, BAT *rl, BAT *rh,
		BAT *sl, BAT *sr, bool li, bool hi, BUN estimate)
{
	BAT *r1, *r2;
	BUN maxsize;

	*r1p = NULL;
	*r2p = NULL;
	if (joinparamcheck(l, rl, rh, sl, sr, "BATrangejoin") != GDK_SUCCEED)
		return GDK_FAIL;
	if ((maxsize = joininitresults(&r1, &r2, sl ? BATcount(sl) : BATcount(l), sr ? BATcount(sr) : BATcount(rl), false, false, false, false, false, estimate)) == BUN_NONE)
		return GDK_FAIL;
	*r1p = r1;
	*r2p = r2;
	if (maxsize == 0)
		return GDK_SUCCEED;

	/* note, the rangejoin implementation is in gdk_select.c since
	 * it uses the imprints code there */
	return rangejoin(r1, r2, l, rl, rh, sl, sr, li, hi, maxsize);
}
