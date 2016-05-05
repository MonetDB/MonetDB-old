/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/*
 * @a Kostis Kyzirakos, Foteini Alvanaki
 * @* A grid based index
 */

#include "grid.h"

static size_t
countSetBits(uint64_t *resBitvector, size_t vectorSize)
{
	size_t num = 0 ;
	for(size_t k = 0; k < vectorSize; k++) {
		uint64_t b = resBitvector[k];
		for(uint64_t j = 0; j < BITSNUM; j++) {
			num += b & 0x01;
			b >>= 1;
		}
	}

	return num;
}

#define TP bte
#include "grid_create_impl.h"
#undef TP
#define TP sht
#include "grid_create_impl.h"
#undef TP
#define TP int
#include "grid_create_impl.h"
#undef TP
#define TP lng
#include "grid_create_impl.h"
#undef TP
#ifdef HAVE_HGE
#define TP hge
#include "grid_create_impl.h"
#undef TP
#endif
#define TP flt
#include "grid_create_impl.h"
#undef TP
#define TP dbl
#include "grid_create_impl.h"
#undef TP
#define TP1 bte
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 bte
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 bte
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 bte
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 bte
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 bte
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 bte
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 sht
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 sht
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 sht
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 sht
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 sht
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 sht
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 sht
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 int
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 int
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 int
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 int
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 int
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 int
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 int
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 lng
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 lng
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 lng
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 lng
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 lng
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 lng
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 lng
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#ifdef HAVE_HGE
#define TP1 hge
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 flt
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 flt
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 flt
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 flt
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 flt
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 flt
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 flt
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 dbl
#define TP2 bte
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 dbl
#define TP2 sht
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 dbl
#define TP2 int
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 dbl
#define TP2 lng
#include "grid_impl.h"
#undef TP1
#undef TP2
#ifdef HAVE_HGE
#define TP1 dbl
#define TP2 hge
#include "grid_impl.h"
#undef TP1
#undef TP2
#endif
#define TP1 dbl
#define TP2 flt
#include "grid_impl.h"
#undef TP1
#undef TP2
#define TP1 dbl
#define TP2 dbl
#include "grid_impl.h"
#undef TP1
#undef TP2

#define GRIDextend(g1, g2, cellR, cellS, r1, r2, msg)                             \
do {                                                                              \
/* make space to cater for the worst case where all points qualify */             \
BUN maxSize = BATcount((r1)) +                                                    \
              ((g1)->dir[(cellR)+1]-(g1)->dir[(cellR)])*                          \
              ((g2)->dir[(cellS)+1]-(g2)->dir[(cellS)]);                          \
	if (maxSize > GDK_oid_max) {                                                  \
			(msg) = createException(MAL,"grid.distance","overflow of head value");\
			goto distancejoin_fail;                                               \
	}                                                                             \
	while (maxSize > BATcapacity((r1))) {                                         \
		if ((BATextend((r1), BATgrows((r1))) != GDK_SUCCEED) ||                   \
			(BATextend((r2), BATgrows((r2))) != GDK_SUCCEED)) {                   \
			(msg) = createException(MAL, "grid.distance",                         \
					"could not extend BATs for storing the join results");        \
			goto distancejoin_fail;                                               \
		}                                                                         \
	}                                                                             \
} while (0)

#define GRIDdist(r1Vals, oid1, seq1, r1b, x1v, y1v,                               \
                 r2Vals, oid2, seq2, r2b, x2v, y2v)                               \
do {                                                                              \
	dbl ddist = ((x2v)-(x1v))*((x2v)-(x1v))+((y2v)-(y1v))*((y2v)-(y1v));          \
	(r1Vals)[(r1b)] = (oid1) + (seq1);                                            \
	(r2Vals)[(r2b)] = (oid2) + (seq2);                                            \
	(r1b) += ddist <= distsqr;                                                    \
	(r2b) += ddist <= distsqr;                                                    \
} while(0)

#define GRIDcmp(tpe, x1BAT, y1BAT, g1,                                            \
                     x2BAT, y2BAT, g2,                                            \
                cellR, cellS, r1, r2, seq1, seq2, msg)                            \
do {                                                                              \
	BUN r1b, r2b;                                                                 \
	tpe *x1Vals, *y1Vals, *x2Vals, *y2Vals;                                       \
	oid *r1Vals, *r2Vals;                                                         \
	oid m;                                                                        \
	if ((cellR) >= (g1)->cellsNum || (cellS) >= (g2)->cellsNum)                   \
		continue;                                                                 \
	GRIDextend(g1, g2, cellR, cellS, r1, r2, msg);                                \
	x1Vals = (tpe*)Tloc(x1BAT, BUNfirst(x1BAT));                                  \
	y1Vals = (tpe*)Tloc(y1BAT, BUNfirst(y1BAT));                                  \
	x2Vals = (tpe*)Tloc(x2BAT, BUNfirst(x2BAT));                                  \
	y2Vals = (tpe*)Tloc(y2BAT, BUNfirst(y2BAT));                                  \
	r1Vals = (oid*)Tloc(r1, BUNfirst(r1));                                        \
	r2Vals = (oid*)Tloc(r2, BUNfirst(r2));                                        \
	r1b = BATcount(r1);                                                           \
	r2b = BATcount(r2);                                                           \
	m = (g1)->dir[(cellR)];                                                       \
	if (GRIDcount(g1, cellR) > 16) {                                              \
		/* compare points of R in cellR with points of S in cellS */              \
		for (; m < (g1)->dir[(cellR)+1]-16; m+=16) {                              \
			for (oid n = (g2)->dir[(cellS)]; n < (g2)->dir[(cellS)+1]; n++) {     \
				oid oid2 = (g2)->oids[n];                                         \
				tpe x2v = (x2Vals)[oid2];                                         \
				tpe y2v = (y2Vals)[oid2];                                         \
				for(oid o1 = m; o1 < m+16; o1++) {                                \
					oid oid1 = (g1)->oids[o1];                                    \
					tpe x1v = (x1Vals)[oid1];                                     \
					tpe y1v = (y1Vals)[oid1];                                     \
					GRIDdist(r1Vals, oid1, seq1, r1b, x1v, y1v,                   \
							 r2Vals, oid2, seq2, r2b, x2v, y2v);                  \
				}                                                                 \
			}                                                                     \
		}                                                                         \
	}                                                                             \
	for (; m < (g1)->dir[(cellR)+1]; m++) {                                       \
		oid oid1 = (g1)->oids[m];                                                 \
		tpe x1v = (x1Vals)[oid1];                                                 \
		tpe y1v = (y1Vals)[oid1];                                                 \
		for (oid n = (g2)->dir[(cellS)]; n < (g2)->dir[(cellS)+1]; n++) {         \
			oid oid2 = (g2)->oids[n];                                             \
			tpe x2v = (x2Vals)[oid2];                                             \
			tpe y2v = (y2Vals)[oid2];                                             \
			GRIDdist(r1Vals, oid1, seq1, r1b, x1v, y1v,                           \
					 r2Vals, oid2, seq2, r2b, x2v, y2v);                          \
		}                                                                         \
	}                                                                             \
	BATsetcount(r1, r1b);                                                         \
	BATsetcount(r2, r2b);                                                         \
} while (0)

#define GRIDjoin(tpe,                                                             \
                 x1BAT, y1BAT, g1, x2BAT, y2BAT, g2,                              \
                 R, S, r1, r2, seq1, seq2, msg)                                   \
do {                                                                              \
	/* perform the distance join */                                               \
	for (size_t i = minCellx; i <= maxCellx; i++) {                               \
		for (size_t j = minCelly; j <= maxCelly; j++) {                           \
			/* define which cells should be compared */                           \
			size_t min = i + g1->cellsX*j;                                        \
			size_t R[] = {min,            min,              min,   min, min+1, min+g1->cellsX, min+g1->cellsX+1}; \
			size_t S[] = {min+g1->cellsX, min+g1->cellsX+1, min+1, min, min,   min,            min             }; \
			for (size_t k = 0; k < 7; k++) {                                      \
				if (GRIDcount(g1,R[k]) > GRIDcount(g2,S[k])) {                    \
					GRIDcmp(tpe, x1BAT, y1BAT, g1, x2BAT, y2BAT,                  \
                                 g2, R[k], S[k], r1, r2, seq1, seq2, msg);        \
				} else {                                                          \
					GRIDcmp(tpe, x2BAT, y2BAT, g2, x1BAT, y1BAT,                  \
                                 g1, S[k], R[k], r2, r1, seq2, seq1, msg);        \
				}                                                                 \
			}                                                                     \
		}                                                                         \
	}                                                                             \
} while(0)

#if 0
static void
grid_print(Grid * g)
{
	int m = 0, n = 0, o = 0;
	fprintf(stderr, "GRID %p (shift %d). batx: %d, baty: %d\n", g, g->shift, g->xbat, g->ybat);
	fprintf(stderr, "- Universe: [%f %f, %f %f]\n", g->xmin, g->ymin, g->xmax, g->ymax);
	fprintf(stderr, "- MBB     : [%f %f, %f %f]\n", g->mbb.xmin, g->mbb.ymin, g->mbb.xmax, g->mbb.ymax);
	fprintf(stderr, "- Cells  X: %zu, Y: %zu, total: %zu, per axis: %zu\n", g->cellsX, g->cellsY, g->cellsNum, g->cellsPerAxis);
	m = ceil(log10(g->dir[g->cellsNum]));
	n = ceil(log10(g->cellsY-1));
	o = ceil(log10(g->cellsX-1));
	m = m > o ? m : o;

	fprintf(stderr, "- Directory\n");
	for (size_t i = 0; i < g->cellsX*(m+1)+n+4; i++)
		fprintf(stderr,"-");
	fprintf(stderr, "\n");
	for (size_t k = 0; k < g->cellsY; k++) {
		size_t j = g->cellsY - k - 1;
		fprintf(stderr,"||%*zu||",n,j);
		for (size_t i = 0; i < g->cellsX; i++) {
			oid cell = i + j*g->cellsX;
			size_t v = g->dir[cell+1]-g->dir[cell];
			if (v == 0)
				fprintf(stderr, "%*s|", m, "");
			else
				fprintf(stderr, "%*zu|", m, v);
		}
		fprintf(stderr,"\n");
	}
	for (size_t i = 0; i < g->cellsX*(m+1)+n+4; i++)
		fprintf(stderr,"-");
	fprintf(stderr, "\n");
	fprintf(stderr, "||%*s||", n, "");
	for (size_t i = 0; i < g->cellsX; i++)
		fprintf(stderr, "%*zu|", m, i);
	fprintf(stderr, "\n");

	fprintf(stderr, "- Directory\n[");
	for (size_t i = 0; i <= g->cellsNum; i++)
		fprintf(stderr, "%zu,",g->dir[i]);
	fprintf(stderr, "\r]\n");
	fprintf(stderr, "- Directory\n");
	for (size_t i = 0; i < g->cellsNum; i++)
		if (g->dir[i] != g->dir[i+1])
			fprintf(stderr, "[%zu] %zu,", i, g->dir[i]);
	fprintf(stderr, "\r\n");

	fprintf(stderr, "- OIDs\n[");
	for (size_t i = 0; i < g->dir[g->cellsNum]; i++) {
		fprintf(stderr, "%zu,", g->oids[i]);
	}
	fprintf(stderr, "\r]\n");
}
#endif

static str
grid_create_mbr(Grid * g, BAT *bx, BAT *by, mbr *m, dbl * d)
{
	lng *xVals, *yVals;
	size_t i, cnt;
	dbl fxa, fxb, fya, fyb;

	assert(BATcount(bx) == BATcount(by));
	assert(BATcount(bx) > 0);
	assert((*d) > 0);

	g->xbat = bx->batCacheid;
	g->ybat = by->batCacheid;
	xVals = (lng*)Tloc(bx, BUNfirst(bx));
	yVals = (lng*)Tloc(by, BUNfirst(by));

	g->mbb.xmin = m->xmin;
	g->mbb.ymin = m->ymin;
	g->mbb.xmax = m->xmax;
	g->mbb.ymax = m->ymax;
	g->xmin = m->xmin;
	g->ymin = m->ymin;
	g->xmax = m->xmax;
	g->ymax = m->ymax;

#if 0
#ifndef NDEBUG
	fprintf(stderr, "grid borders: [%f %f, %f %f]\n", g->mbb.xmin, g->mbb.ymin, g->mbb.xmax, g->mbb.ymax);
#endif
#endif
	/* determine the appropriate number of cells */
	g->cellsX = (size_t)ceil((m->xmax - m->xmin)/(*d))+1;
	g->cellsY = (size_t)ceil((m->ymax - m->ymin)/(*d))+1;
	if ((size_t)(m->xmax-m->xmin)/(*d) == (size_t)g->cellsX*(*d))
		g->cellsX++;
	if ((m->ymax-m->ymin)/(*d) == g->cellsY*(*d))
		g->cellsY++;
	g->mbb.xmax = g->mbb.xmin + g->cellsX*(*d);
	g->mbb.ymax = g->mbb.ymin + g->cellsY*(*d);

	/* how many bits do we need? */
	g->shift = (bte)ceil(log2(g->cellsX>g->cellsY?g->cellsX:g->cellsY));
	cnt = BATcount(bx);
	g->cellsNum = g->cellsX * g->cellsY;
#if 0
#ifndef NDEBUG
	fprintf(stderr, "shift: %d\n", g->shift);
	fprintf(stderr, "Cells: x=%zu, y=%zu (total: %zu)\n", g->cellsX, g->cellsY, g->cellsNum);
#endif
#endif

	/* allocate space for the directory */
	if ((g->dir = GDKmalloc((g->cellsNum+1)*sizeof(oid))) == NULL) {
		GDKfree(g);
		g = NULL;
		return createException(MAL, "grid.create_mbr", MAL_MALLOC_FAIL);
	}
	for (i = 0; i <= g->cellsNum; i++)
		g->dir[i] = 0;

	/* allocate space for the index */
	if((g->oids = GDKmalloc(BATcount(bx)*sizeof(oid))) == NULL) {
		GDKfree(g);
		g = NULL;
		return createException(MAL, "grid.create_mbr", MAL_MALLOC_FAIL);
	}

	/* compute the index */
	/* step 1: compute the histogram of cell frequencies */
	fxa = ((double)g->cellsX/(g->mbb.xmax-g->mbb.xmin));
	fxb = (double)g->mbb.xmin*fxa;
	fya = ((double)g->cellsY/(g->mbb.ymax-g->mbb.ymin));
	fyb = (double)g->mbb.ymin*fya;
#if 0
#ifndef NDEBUG
	fprintf(stderr, "Coefficients: fxa=%f, fxb=%f\n", fxa, fxb);
	fprintf(stderr, "Coefficients: fya=%f, fyb=%f\n", fya, fyb);
#endif
#endif

	cnt = BATcount(bx);
	for (i = 0; i < cnt; i++) {
		oid cellx = (double)xVals[i]*fxa - fxb;
		oid celly = (double)yVals[i]*fya - fyb;
		//oid cell = ((cellx << g->shift) | celly);
		oid cell = cellx + g->cellsX*celly;
		assert(cell < g->cellsNum);
		g->dir[cell+1]++;
	}

	/* step 2: compute the directory pointers */
	for (i = 1; i < g->cellsNum; i++)
		g->dir[i] += g->dir[i-1];

	/* step 3: fill in the oid array */
	for (size_t i = 0; i < cnt; i++) {
		oid cellx = (double)xVals[i]*fxa - fxb;
		oid celly = (double)yVals[i]*fya - fyb;
		//oid cell = ((cellx << g->shift) | celly);
		oid cell = cellx + g->cellsX*celly;
		g->oids[g->dir[cell]++] = i;
	}

	/* step 4: adjust the directory pointers */
	for (size_t i = g->cellsNum; i > 0; i--)
		g->dir[i+1] = g->dir[i];
	g->dir[0] = 0;

	/* TODO: move here the code for compressing the index */

	g->cellsPerAxis = 0;

#if 0
#ifndef NDEBUG
	grid_print(g);
#endif
#endif
	return MAL_SUCCEED;
}


static str
grid_create_bats(Grid ** gg1, Grid **gg2, mbr ** common, BAT *bx1, BAT *by1, BAT *bx2, BAT *by2, dbl * d)
{
	Grid * g1 = NULL, * g2 = NULL;
	lng *x1Vals, *y1Vals, *x2Vals, *y2Vals;
	size_t i, cnt1, cnt2;
	mbr *m1 = NULL, *m2 = NULL, *mi = NULL, *mu = NULL;
	str msg = MAL_SUCCEED;

	assert(BATcount(bx1) == BATcount(by1));
	assert(BATcount(bx2) == BATcount(by2));
	assert(BATcount(bx1) > 0);
	assert(BATcount(bx2) > 0);

	if (((mi = GDKmalloc(sizeof(mbr))) == NULL) ||
		((mu = GDKmalloc(sizeof(mbr))) == NULL) ||
		((m1 = GDKmalloc(sizeof(mbr))) == NULL) ||
		((m2 = GDKmalloc(sizeof(mbr))) == NULL) ||
		((g1 = GDKmalloc(sizeof(Grid))) == NULL) ||
		((g2 = GDKmalloc(sizeof(Grid))) == NULL)) {
		msg=createException(MAL, "grid.create_bats", MAL_MALLOC_FAIL);
		goto grid_create_bats_fail;
	}

	g1->xbat = bx1->batCacheid;
	g1->ybat = by1->batCacheid;
	x1Vals = (lng*)Tloc(bx1, BUNfirst(bx1));
	y1Vals = (lng*)Tloc(by1, BUNfirst(by1));
	g2->xbat = bx2->batCacheid;
	g2->ybat = by2->batCacheid;
	x2Vals = (lng*)Tloc(bx2, BUNfirst(bx2));
	y2Vals = (lng*)Tloc(by2, BUNfirst(by2));

	/* find min and max values for X and y coordinates */
	cnt1 = BATcount(bx1);
	g1->xmin = g1->xmax = x1Vals[0];
	for (i = 1; i < cnt1; i++) {
		lng val = x1Vals[i];
		if(g1->xmin > val)
			g1->xmin = val;
		if(g1->xmax < val)
			g1->xmax = val;
	}
	g1->ymin = g1->ymax = y1Vals[0];
	for (i = 1; i < cnt1; i++) {
		lng val = y1Vals[i];
		if(g1->ymin > val)
			g1->ymin = val;
		if(g1->ymax < val)
			g1->ymax = val;
	}
	g1->mbb.xmin=g1->xmin;
	g1->mbb.ymin=g1->ymin;
	g1->mbb.xmax=g1->xmax;
	g1->mbb.ymax=g1->ymax;
#if 0
#ifndef NDEBUG
	fprintf(stderr, "Outer relation limits: [%f %f, %f %f]\n", g1->mbb.xmin, g1->mbb.ymin, g1->mbb.xmax, g1->mbb.ymax);
	fprintf(stderr, "Outer relation limits: [%f %f, %f %f]\n", g1->xmin, g1->ymin, g1->xmax, g1->ymax);
#endif
#endif
	cnt2 = BATcount(bx2);
	g2->xmin = g2->xmax = x2Vals[0];
	for (i = 1; i < cnt2; i++) {
		lng val = x2Vals[i];
		if(g2->xmin > val)
			g2->xmin = val;
		if(g2->xmax < val)
			g2->xmax = val;
	}
	g2->ymin = g2->ymax = y2Vals[0];
	for (i = 1; i < cnt2; i++) {
		lng val = y2Vals[i];
		if(g2->ymin > val)
			g2->ymin = val;
		if(g2->ymax < val)
			g2->ymax = val;
	}
	g2->mbb.xmin=g2->xmin;
	g2->mbb.ymin=g2->ymin;
	g2->mbb.xmax=g2->xmax;
	g2->mbb.ymax=g2->ymax;
#if 0
#ifndef NDEBUG
	fprintf(stderr, "Outer relation limits: [%f %f, %f %f]\n", g2->mbb.xmin, g2->mbb.ymin, g2->mbb.xmax, g2->mbb.ymax);
	fprintf(stderr, "Outer relation limits: [%f %f, %f %f]\n", g2->xmin, g2->ymin, g2->xmax, g2->ymax);
#endif
#endif
	m1->xmin = g1->mbb.xmin - *d; m1->ymin = g1->mbb.ymin - *d; m1->xmax = g1->mbb.xmax + *d; m1->ymax = g1->mbb.ymax + *d;
	m2->xmin = g2->mbb.xmin - *d; m2->ymin = g2->mbb.ymin - *d; m2->xmax = g2->mbb.xmax + *d; m2->ymax = g2->mbb.ymax + *d;

	/* compute the intersection between the two coverages */
	mbrIntersection(&mi, &m1, &m2);  /* query the grid only on the intersecting mbr */
	mbrUnion(&mu, &m1, &m2);         /* create the grid index on the union of the two universes */
#if 0
#ifndef NDEBUG
	fprintf(stderr, "outer relation mbb: [%f %f, %f %f]\n", m1->xmin, m1->ymin, m1->xmax, m1->ymax);
	fprintf(stderr, "inner relation mbb: [%f %f, %f %f]\n", m2->xmin, m2->ymin, m2->xmax, m2->ymax);
	fprintf(stderr, "  intersection mbb: [%f %f, %f %f]\n", mi->xmin, mi->ymin, mi->xmax, mi->ymax);
	fprintf(stderr, "         union mbb: [%f %f, %f %f]\n", mu->xmin, mu->ymin, mu->xmax, mu->ymax);
#endif
#endif
	/* if the two mbbs do not intersect, return an empty result */
	if (mbr_isnil(mi))
		goto grid_create_bats_return; 

	/* create grid index using a common grid */
	if ( grid_create_mbr(g1, bx1, by1, mu, d) != MAL_SUCCEED ||
		 grid_create_mbr(g2, bx2, by2, mu, d) != MAL_SUCCEED) {
		msg = createException(MAL, "grid.create_bats", "Could not create the grid index");
		goto grid_create_bats_return;
	}

	/* TODO: move here the code for compressing the index */
	*gg1 = g1;
	*gg2 = g2;
	*common = mi;
	GDKfree(mu);
	GDKfree(m1);
	GDKfree(m2);

grid_create_bats_return:
	return msg;

grid_create_bats_fail:
	if (mi) GDKfree(mi);
	if (mu) GDKfree(mu);
	if (m1) GDKfree(m1);
	if (m2) GDKfree(m2);
	if (g1) GDKfree(g1);
	if (g2) GDKfree(g2);
	g1 = NULL;
	g2 = NULL;
	goto grid_create_bats_return;
}

static str
distance_typesswitchloop(BUN * ret, const void *lft, int tp1, int incr1,
						 const void *rgt, int tp2, int incr2,
						 void *restrict dst, int tp, BUN cnt,
						 BUN start, BUN end, const oid *restrict cand,
						 const oid *candend, oid candoff,
						 int abort_on_error, const char *func)
{

	BUN r;
	tp1 = ATOMbasetype(tp1);
	tp2 = ATOMbasetype(tp2);
	tp = ATOMbasetype(tp);

	switch (tp1) {
		case TYPE_bte :
			switch (tp2) {
				case TYPE_bte :
					r = distance_bte_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_bte_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_bte_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_bte_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#ifdef HAVE_HGE
				case TYPE_hge :
					r = distance_bte_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#endif
				case TYPE_flt :
					r = distance_bte_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_bte_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
		case TYPE_sht :
			switch (tp2) {
				case TYPE_bte :
					r = distance_sht_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_sht_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_sht_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_sht_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#ifdef HAVE_HGE
				case TYPE_hge :
					r = distance_sht_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#endif
				case TYPE_flt :
					r = distance_sht_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_sht_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
		case TYPE_int :
			switch (tp2) {
				case TYPE_bte :
					r = distance_int_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_int_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_int_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_int_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#ifdef HAVE_HGE
				case TYPE_hge :
					r = distance_int_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#endif
				case TYPE_flt :
					r = distance_int_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_int_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
		case TYPE_lng :
			switch (tp2) {
				case TYPE_bte :
					r = distance_lng_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_lng_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_lng_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_lng_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#ifdef HAVE_HGE
				case TYPE_hge :
					r = distance_lng_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#endif
				case TYPE_flt :
					r = distance_lng_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_lng_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
#ifdef HAVE_HGE
		case TYPE_hge :
			switch (tp2) {
				case TYPE_bte :
					r = distance_hge_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_hge_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_hge_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_hge_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_hge :
					r = distance_hge_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_flt :
					r = distance_hge_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_hge_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
#endif
		case TYPE_flt :
			switch (tp2) {
				case TYPE_bte :
					r = distance_flt_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_flt_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_flt_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_flt_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#ifdef HAVE_HGE
				case TYPE_hge :
					r = distance_flt_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#endif
				case TYPE_flt :
					r = distance_flt_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_flt_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
		case TYPE_dbl :
			switch (tp2) {
				case TYPE_bte :
					r = distance_dbl_bte(lft, incr1, rgt, incr2,
							dst, GDK_bte_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_sht :
					r = distance_dbl_sht(lft, incr1, rgt, incr2,
							dst, GDK_sht_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_int :
					r = distance_dbl_int(lft, incr1, rgt, incr2,
							dst, GDK_int_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_lng :
					r = distance_dbl_lng(lft, incr1, rgt, incr2,
							dst, GDK_lng_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#ifdef HAVE_HGE
				case TYPE_hge :
					r = distance_dbl_hge(lft, incr1, rgt, incr2,
							dst, GDK_hge_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
#endif
				case TYPE_flt :
					r = distance_dbl_flt(lft, incr1, rgt, incr2,
							dst, GDK_flt_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				case TYPE_dbl :
					r = distance_dbl_dbl(lft, incr1, rgt, incr2,
							dst, GDK_dbl_max, cnt, start, end,
							cand, candend, candoff, abort_on_error);
					break;
				default:
					goto unsupported;
			}
		default:
			goto unsupported;
	}

	*ret = r;
	return MAL_SUCCEED;

unsupported:
	return createException(MAL, "GRIDdistance", "%s: type combination (add(%s,%s)->%s) not supported.\n", func, ATOMname(tp1), ATOMname(tp2), ATOMname(tp))
}

static str
GRIDdistance_(ValPtr ret, const ValRecord *lft, const ValRecord *rgt, int abort_on_error)
{
	return distance_typeswitchloop(VALptr(lft), lft->vtype, 0,
					VALptr(rgt), rgt->vtype, 0,
					VALget(ret), ret->vtype, 1,
					0, 1, NULL, NULL, 0,
					abort_on_error, "GRIDdistance");
}

str
GRIDdistance(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;

	return GRIDdistance_(&stk->stk[getArg(pci, 0)], &stk->stk[getArg(pci, 1)], &stk->stk[getArg(pci, 2)], 1) != MAL_SUCCEED;
#if 0
	str msg = MAL_SUCCEED;
	BUN nils;
	ValPtr ret;
	ValRecord *lft1;
	ValRecord *rgt1;
	int abort_on_error;

	void *lft;
	int tp1;
	int incr1;
	void *rgt;
	int tp2;
	int incr2;
	void *dst;
	int tp;
	BUN cnt;
	BUN start;
	BUN end;
	oid * cand;
	oid *candend;
	oid candoff;
	char *func;

	(void)cntxt;
	(void)mb;

	ret = stk->stk[getArg(pci, 0)];
	lft1 = stk->stk[getArg(pci, 1)];
	rgt1 = stk->stk[getArg(pci, 2)];
	abort_on_error = 1;
	
	lft = VALptr(lft1);
	tp1 = lft->vtype;
	incr1 = 0;
	rgt = VALptr(rgt1);
	tp2 = rgt->vtype;
	incr2 = 0;
	dst = VALget(ret);
	tp = ret->vtype;
	cnt = 1;
	start = 0;
	end = 1;
	cand = NULL;
	candend = NULL;
	candoff = 0;
	func = "VARcalcadd";

	tp1 = ATOMbasetype(tp1);
	tp2 = ATOMbasetype(tp2);
	tp = ATOMbasetype(tp);

	switch (tp1) {
	case TYPE_bte:
		switch (tp2) {
		case TYPE_bte:
			switch (tp) {
			case TYPE_bte:
				nils = add_bte_bte_bte(lft, incr1, rgt, incr2,
								       dst, GDK_bte_max, cnt,
								       start, end,
								       cand, candend, candoff,
								       abort_on_error);
				break;
			default:
				goto unsupported;
			}
			break;
		default:
			goto unsupported;
		}
		break;
	default:
		goto unsupported;
	}


distance_return;
	if (nils == BUN_NONE)
		msg = mythrow(MAL, "calc.+", OPERATION_FAILED);

	return msg;

unsupported:
	nils = BUN_NONE;
	goto distance_return;
#endif 
}

#if 0
str
GRIDdistancesubselect(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void)cntxt;
	(void)mb;
	(void)stk;
	(void)pci;

	return MAL_SUCCEED;
}
#endif

#if 0
static str
GRIDdistance_(bit * res, void * x1, void * y1, void * x2, void * y2, dbl * d)
{
	(void)res;
	(void)x1;
	(void)y1;
	(void)x2;
	(void)y2;
	(void)d;

	return MAL_SUCCEED;
}

static str
GRIDdistancesubselect_(bat * res, bat * x1, bat * y1, bat * cand1, void * x2, void * y2, dbl * d, bit * anti)
{
	(void)res;
	(void)x1;
	(void)y1;
	(void)cand1;
	(void)x2;
	(void)y2;
	(void)d;
	(void)anti;

	return MAL_SUCCEED;
}
#endif

str
GRIDdistancesubjoin(bat *res1, bat * res2,bat * x1, bat * y1, bat * x2, bat * y2, dbl * d, bat * s1, bat * s2, bit * nil, lng * estimate)
{
	size_t num1, num2;
	size_t minCellx, minCelly, maxCellx, maxCelly;
	BAT *x1BAT = NULL, *y1BAT = NULL, *x2BAT = NULL, *y2BAT = NULL;
	Grid * g1 = NULL, * g2 = NULL;
	BAT *r1 = NULL, *r2 = NULL;
	mbr * mbb = NULL;
	oid seq1, seq2;
	double distance = (double)*d;
	double distsqr = (distance)*(distance);
	str msg = MAL_SUCCEED;

	assert (distance > 0);

	(void)nil;
	(void)estimate;
	(void)num1;
	(void)num2;
	(void)s1;
	(void)s2;

	/* allocate space for the results */
	if (((r1 = BATnew(TYPE_void, TYPE_oid, 0, TRANSIENT)) == NULL) ||
		((r2 = BATnew(TYPE_void, TYPE_oid, 0, TRANSIENT)) == NULL)) {
		msg = createException(MAL, "grid.distance", "could not create BATs for storing the join results");
		goto distancejoin_fail;
	}

	/* get the X and Y BATs*/
	if((x1BAT = BATdescriptor(*x1)) == NULL) {
		msg = createException(MAL, "grid.distance", "runtime object missing");
		goto distancejoin_fail;
	}
	if((y1BAT = BATdescriptor(*y1)) == NULL) {
		msg = createException(MAL, "grid.distance", "runtime object missing");
		goto distancejoin_fail;
	}
	if((x2BAT = BATdescriptor(*x2)) == NULL) {
		msg = createException(MAL, "grid.distance", "runtime object missing");
		goto distancejoin_fail;
	}
	if((y2BAT = BATdescriptor(*y2)) == NULL) {
		msg = createException(MAL, "grid.distance", "runtime object missing");
		goto distancejoin_fail;
	}

	/* check if the BATs have dense heads and are aligned */
	if (!BAThdense(x1BAT) || !BAThdense(y1BAT)) {
		msg = createException(MAL, "grid.distance", "BATs must have dense heads");
		goto distancejoin_fail;
	}
	if(x1BAT->hseqbase != y1BAT->hseqbase || BATcount(x1BAT) != BATcount(y1BAT)) {
		/* hack for hanlding the Cartesian product introduced by mitosis        */
		/* Compare only bats with the same seqbase                              */
		/* TODO: make mitosis aware of filter functions utilizing multiple BATs */
		goto distancejoin_returnempty;
	}
	if (!BAThdense(x2BAT) || !BAThdense(y2BAT)) {
		msg = createException(MAL, "grid.distance", "BATs must have dense heads");
		goto distancejoin_fail;
	}
	if(x2BAT->hseqbase != y2BAT->hseqbase || BATcount(x2BAT) != BATcount(y2BAT)) {
		/* hack for mitosis */
		goto distancejoin_returnempty;
	}
	assert(x1BAT->ttype == y1BAT->ttype);
	assert(y2BAT->ttype == y2BAT->ttype);
	assert(x1BAT->ttype == x1BAT->ttype);
	num1 = BATcount(x1BAT);
	num2 = BATcount(x2BAT);
	seq1 = x1BAT->hseqbase;
	seq2 = x2BAT->hseqbase;
	BATsetaccess(r1, BAT_APPEND);
	BATsetaccess(r2, BAT_APPEND);

	if ((mbb = GDKmalloc(sizeof(mbr))) == NULL) {
		msg = createException(MAL, "grid.distance", "malloc failed");
		goto distancejoin_fail;
	}

	/* compute the grid index */
	if (grid_create_bats(&g1, &g2, &mbb, x1BAT, y1BAT, x2BAT, y2BAT, &distance) != MAL_SUCCEED) {
		msg = createException(MAL, "grid.distance", "could not compute the grid index");
		goto distancejoin_fail;
	}

	/* find which cells have to be examined */
	minCellx = (size_t)((mbb->xmin)/(*d) - g1->mbb.xmin/(*d));
	maxCellx = (size_t)((mbb->xmax)/(*d) - g1->mbb.xmin/(*d));
	minCelly = (size_t)((mbb->ymin)/(*d) - g1->mbb.ymin/(*d));
	maxCelly = (size_t)((mbb->ymax)/(*d) - g1->mbb.ymin/(*d));

	assert(maxCellx < g1->cellsX && maxCellx < g2->cellsX);
	assert(maxCelly < g1->cellsY && maxCelly < g2->cellsY);

	switch (x1BAT->ttype) {
		case TYPE_bte:
			GRIDjoin(bte, x1BAT, y1BAT, g1, x2BAT, y2BAT, g2, R, S, r1, r2, seq1, seq2, msg);
			break
			;;
		case TYPE_sht:
			GRIDjoin(sht, x1BAT, y1BAT, g1, x2BAT, y2BAT, g2, R, S, r1, r2, seq1, seq2, msg);
			break
			;;
		case TYPE_int:
			GRIDjoin(int, x1BAT, y1BAT, g1, x2BAT, y2BAT, g2, R, S, r1, r2, seq1, seq2, msg);
			break
			;;
		case TYPE_lng:
			GRIDjoin(lng, x1BAT, y1BAT, g1, x2BAT, y2BAT, g2, R, S, r1, r2, seq1, seq2, msg);
			break
			;;
		case TYPE_flt:
			GRIDjoin(flt, x1BAT, y1BAT, g1, x2BAT, y2BAT, g2, R, S, r1, r2, seq1, seq2, msg);
			break
			;;
		case TYPE_dbl:
			GRIDjoin(dbl, x1BAT, y1BAT, g1, x2BAT, y2BAT, g2, R, S, r1, r2, seq1, seq2, msg);
			break
			;;
		default:
			msg = createException(MAL, "grid.subjoin", "Unsupported data type");
			goto distancejoin_fail;
	}


distancejoin_returnempty:
	/* keep the results */
	//BATderiveProps(r1, false);
	//BATderiveProps(r2, false);
	r1->tsorted = false;
	r1->trevsorted = false;
	r2->tsorted = false;
	r2->trevsorted = false;
	*res1 = r1->batCacheid;
	*res2 = r2->batCacheid;
	BBPkeepref(*res1);
	BBPkeepref(*res2);
distancejoin_clean:
	/* clean up */
	if (x1BAT) BBPunfix(x1BAT->batCacheid);
	if (y1BAT) BBPunfix(y1BAT->batCacheid);
	if (x2BAT) BBPunfix(x2BAT->batCacheid);
	if (y2BAT) BBPunfix(y2BAT->batCacheid);
	if (g1) {
		if (g1->dir)  GDKfree(g1->dir);
		if (g1->oids) GDKfree(g1->oids);
		GDKfree(g1);
	}
	if (g2) {
		if (g2->dir)  GDKfree(g2->dir);
		if (g2->oids) GDKfree(g2->oids);
		GDKfree(g2);
	}

	return msg;

distancejoin_fail:
	*res1 = 0;
	*res2 = 0;
	if (r1) BBPunfix(r1->batCacheid);
	if (r2) BBPunfix(r2->batCacheid);
	goto distancejoin_clean;
}

