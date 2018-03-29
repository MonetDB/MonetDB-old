#include "monetdb_config.h"
#include "gdk.h"
#include "weld_udfs.h"
#include "mal_weld.h"
#include "algebra.h"

#define denseFor(PTR, TYPE, START, END) \
	do {                                \
		TYPE *tptr = (TYPE *)PTR;       \
		oid i;                          \
		for (i = START; i < END; i++) { \
			tptr[i - START] = i;        \
		}                               \
	} while (0)

static BAT *weldVecToBat(char *data, lng length, int type) {
	lng seq = (lng)data;
	if (seq <= 0) {
		/* Dense BAT hack */
		return BATdense(0, -seq, length);
	} else {
		BAT *b = COLnew(0, type, 0, TRANSIENT);
		b->theap.base = data;
		b->batCount = length;
		if (b->batCount == 0) b->theap.base = NULL;
		b->batCapacity = b->batCount;
		b->theap.storage = STORE_NOWN;
		b->tsorted = b->trevsorted = 0;
		b->theap.free = b->batCount << b->tshift;
		b->theap.size = b->batCount << b->tshift;
		return b;
	}
}

static BAT *replaceDenseBat(BAT *b, int type) {
	if (!BATtdense(b)) return b;
	BAT *bn = COLnew(0, type, b->batCount, TRANSIENT);
	if (type == TYPE_bte) denseFor(Tloc(bn, 0), char, b->tseqbase, b->tseqbase + b->batCount);
	if (type == TYPE_int) denseFor(Tloc(bn, 0), int, b->tseqbase, b->tseqbase + b->batCount);
	if (type == TYPE_lng) denseFor(Tloc(bn, 0), lng, b->tseqbase, b->tseqbase + b->batCount);
	if (type == TYPE_oid) denseFor(Tloc(bn, 0), oid, b->tseqbase, b->tseqbase + b->batCount);
	if (type == TYPE_flt) denseFor(Tloc(bn, 0), flt, b->tseqbase, b->tseqbase + b->batCount);
	if (type == TYPE_dbl) denseFor(Tloc(bn, 0), dbl, b->tseqbase, b->tseqbase + b->batCount);
	BATsetcount(bn, b->batCount);
	bn->tsorted = BATcount(bn) < 2;
	bn->trevsorted = BATcount(bn) < 2;
	BBPunfix(b->batCacheid);
	BBPkeepref(bn->batCacheid);
	return bn;
}

/* TODO - for now it only works for non string BATs */
static void weldJoin(void *l, int ltype, void *r, int rtype, void *sl, void *sr, bit *nilmatches,
					 lng *estimate, void *result) {
	BAT *lbat = weldVecToBat(((WeldVec *)l)->data, ((WeldVec *)l)->length, ltype);
	BAT *rbat = weldVecToBat(((WeldVec *)r)->data, ((WeldVec *)r)->length, rtype);
	BAT *slbat =
		sl == NULL ? NULL : weldVecToBat(((WeldVec *)sl)->data, ((WeldVec *)sl)->length, TYPE_oid);
	BAT *srbat =
		sr == NULL ? NULL : weldVecToBat(((WeldVec *)sr)->data, ((WeldVec *)sr)->length, TYPE_oid);
	bat r1, r2;
	bat lid = lbat->batCacheid;
	bat rid = rbat->batCacheid;
	bat slid = slbat == NULL ? bat_nil : slbat->batCacheid;
	bat srid = srbat == NULL ? bat_nil : srbat->batCacheid;
	ALGjoin(&r1, &r2, &lid, &rid, &slid, &srid, nilmatches, estimate);

	/* Tell gdk not to free weld's data */
	lbat->theap.base = NULL;
	rbat->theap.base = NULL;
	if (slbat != NULL) slbat->theap.base = NULL;
	if (srbat != NULL) srbat->theap.base = NULL;

	BAT *r1bat = BATdescriptor(r1);
	r1bat = replaceDenseBat(r1bat, TYPE_oid);
	BAT *r2bat = BATdescriptor(r2);
	r2bat = replaceDenseBat(r2bat, TYPE_oid);

	/* result is a pointer to a struct with 2 weld vectors and 2 ints*/
	char *resultPtr = result;
	getOrSetStructMember(&resultPtr, TYPE_ptr, &r1bat->theap.base, OP_SET);
	getOrSetStructMember(&resultPtr, TYPE_lng, &r1bat->batCount, OP_SET);
	getOrSetStructMember(&resultPtr, TYPE_ptr, &r2bat->theap.base, OP_SET);
	getOrSetStructMember(&resultPtr, TYPE_lng, &r2bat->batCount, OP_SET);
	getOrSetStructMember(&resultPtr, TYPE_bat, &r1bat->batCacheid, OP_SET);
	getOrSetStructMember(&resultPtr, TYPE_bat, &r2bat->batCacheid, OP_SET);
}

#define joinImpl(MTYPE, WTYPE)                                                                \
	mal_export void weldJoinNoCandList##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r,           \
											  bit *nilmatches, lng *estimate, void *result) { \
		weldJoin(l, MTYPE, r, MTYPE, NULL, NULL, nilmatches, estimate, result);               \
	}                                                                                         \
	mal_export void weldJoin##WTYPE(WeldVec##WTYPE *l, WeldVec##WTYPE *r, WeldVec##WTYPE *sl, \
									WeldVec##WTYPE *sr, bit *nilmatches, lng *estimate,       \
									void *result) {                                           \
		weldJoin(l, MTYPE, r, MTYPE, sl, sr, nilmatches, estimate, result);                   \
	}

joinImpl(TYPE_bte, i8);
joinImpl(TYPE_int, i32);
joinImpl(TYPE_oid, i64);
joinImpl(TYPE_flt, f32);
joinImpl(TYPE_dbl, f64);
