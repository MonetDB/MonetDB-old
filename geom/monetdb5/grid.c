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

#define GRID_VERSION 1
#define POINTSPERCELL 100000

#define BITSNUM 64
#define SHIFT 6  /* division with 64 */
#define ONES ((1<<(SHIFT))-1) /* 63 */
#define get(bitVector, bitPos) (((bitVector) >> (bitPos)) & 0x01) 
#define set(bitVector, bitPos, value) ((bitVector) |= ((value)<<(bitPos))) 
#define setbv(bitVector, bitPos, value) set((bitVector)[(bitPos) >> SHIFT], (bitPos) & ONES, (value))
#define unset(bitVector, bitPos) ((bitVector) &= ~((uint64_t)1<<(bitPos)))
#define common(bitVector, bitPos, value) ((bitVector) &= (0xFFFFFFFFFFFFFFFF ^ (((1-value))<<(bitPos))))
#define GRIDcount(g, c) (g)->dir[(c)+1] - (g)->dir[(c)]

#define maximumNumberOfCells(max, bitsNum, add) \
do {                                            \
    int i = 0;                                  \
    size_t res = 0;                             \
    for(i = 0; i < bitsNum; i++)                \
        res = (res << 1) | 1;                   \
    max = res + add;                            \
} while (0)

#define GRIDextend(g1, g2, cellR, cellS, r1, r2, msg)                              \
do {                                                                               \
/* make space to cater for the worst case where all points qualify */              \
BUN maxSize = BATcount((r1)) +                                                     \
              ((g1)->dir[(cellR)+1]-(g1)->dir[(cellR)])*                           \
              ((g2)->dir[(cellS)+1]-(g2)->dir[(cellS)]);                           \
if (maxSize > GDK_oid_max) {                                                       \
		(msg) = createException(MAL, "grid.distance", "overflow of head value");   \
		goto distancejoin_fail;                                                    \
}                                                                                  \
while (maxSize > BATcapacity((r1))) {                                              \
fprintf(stderr, "maxSize: %zu capacity: %zu\n", maxSize, BATcapacity((r1)));       \
	if ((BATextend((r1), BATgrows((r1))) != GDK_SUCCEED) ||                        \
		(BATextend((r2), BATgrows((r2))) != GDK_SUCCEED)) {                        \
		(msg) = createException(MAL, "grid.distance",                              \
                            "could not extend BATs for storing the join results"); \
		goto distancejoin_fail;                                                    \
	}                                                                              \
}                                                                                  \
} while (0)



#define GRIDcmp(x1Vals, y1Vals, g1,                                               \
                x2Vals, y2Vals, g2,                                               \
                cellR, cellS, r1, r2, seq1, seq2, msg)                            \
do {                                                                              \
BUN r1b, r2b;                                                                     \
lng * r1Vals, * r2Vals;                                                           \
if ((cellR) >= (g1)->cellsNum || (cellS) >= (g2)->cellsNum)                       \
	continue;                                                                     \
GRIDextend(g1, g2, cellR, cellS, r1, r2, msg);                                    \
r1Vals = (lng*)Tloc(r1, BUNfirst(r1));                                            \
r2Vals = (lng*)Tloc(r2, BUNfirst(r2));                                            \
r1b = BATcount(r1);                                                               \
r2b = BATcount(r2);                                                               \
/* compare points of R in cellR with points of S in cellS */                      \
for (size_t m = (g1)->dir[(cellR)]; m < (g1)->dir[(cellR)+1]; m++) {              \
	oid oid1 = m;                                                                 \
	lng x1v = (x1Vals)[oid1];                                                     \
	lng y1v = (y1Vals)[oid1];                                                     \
	for (size_t n = (g2)->dir[(cellS)]; n < (g2)->dir[(cellS)+1]; n++) {          \
		size_t oid2 = n;                                                          \
		lng x2v = (x2Vals)[oid2];                                                 \
		lng y2v = (y2Vals)[oid2];                                                 \
		double ddist = (x2v-x1v)*(x2v-x1v)+(y2v-y1v)*(y2v-y1v);                   \
		r1Vals[r1b] = oid1 + seq1;                                                \
		r2Vals[r2b] = oid2 + seq2;                                                \
		r1b += ddist <= distsqr;                                                  \
		r2b += ddist <= distsqr;                                                  \
	}                                                                             \
}                                                                                 \
BATsetcount(r1, r1b);                                                             \
BATsetcount(r2, r2b);                                                             \
} while (0)

typedef struct Grid Grid;
struct Grid {
	dbl xmin;			/* minimum X value of input BATs   */
	dbl ymin;			/* minimum Y value of input BATs   */
	dbl xmax;			/* maximum X value of input BATs   */
	dbl ymax;			/* maximum Y value of input BATs   */
	mbr mbb;			/* grid universe (might differ from the input values) */
	bte shift;
	size_t cellsNum;	/* number of cells                 */
	size_t cellsPerAxis;/* number of cells per axis        */
	size_t cellsX;		/* number of cells in X axis       */
	size_t cellsY;		/* number of cells in Y axis       */
	bat xbat;			/* bat id for X coordinates        */
	bat ybat;			/* bat id for Y coordinates        */
	oid * dir;			/* the grid directory              */
	oid * oids;			/* heap where the index is stored  */
};

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
	fprintf(stderr, "- OIDs\n[");
	for (size_t i = 0; i < g->dir[g->cellsNum]; i++) {
		fprintf(stderr, "%zu,", g->oids[i]);
	}
	fprintf(stderr, "\r]\n");
}
#endif

static Grid *
grid_create(BAT *bx, BAT *by)
{
	Grid * g;
	lng *xVals, *yVals;
	size_t i, cnt;
	dbl fxa, fxb, fya, fyb;

	assert(BATcount(bx) == BATcount(by));
	assert(BATcount(bx) > 0);

	if ((g = GDKmalloc(sizeof(Grid))) == NULL)
		return g;

	g->xbat = bx->batCacheid;
	g->ybat = by->batCacheid;
	xVals = (lng*)Tloc(bx, BUNfirst(bx));
	yVals = (lng*)Tloc(by, BUNfirst(by));

	/* determine the appropriate number of cells */
	g->shift = 2;
	maximumNumberOfCells(g->cellsNum, g->shift*2, 1);
	maximumNumberOfCells(g->cellsPerAxis, g->shift, 0);

	cnt = BATcount(bx);
	while(cnt/g->cellsNum > POINTSPERCELL) {
		/* use one more bit per axis */
		g->shift++;
		maximumNumberOfCells(g->cellsNum, g->shift*2, 1);
		maximumNumberOfCells(g->cellsPerAxis, g->shift, 0);
	}

	/* find min and max values for X and y coordinates */
	g->xmin = g->xmax = xVals[0];
	for (i = 1; i < cnt; i++) {
		lng val = xVals[i];
		if(g->xmin > val)
			g->xmin = val;
		if(g->xmax < val)
			g->xmax = val;
	}
	g->ymin = g->ymax = yVals[0];
	for (i = 1; i < cnt; i++) {
		lng val = yVals[i];
		if(g->ymin > val)
			g->ymin = val;
		if(g->ymax < val)
			g->ymax = val;
	}

	/* allocate space for the directory */
	if ((g->dir = GDKmalloc((g->cellsNum+1)*sizeof(oid))) == NULL) {
		GDKfree(g);
		g = NULL;
		return g;
	}
	for (i = 0; i < g->cellsNum; i++)
		g->dir[i] = 0;

	/* allocate space for the index */
	if((g->oids = GDKmalloc(BATcount(bx)*sizeof(oid))) == NULL) {
		GDKfree(g);
		g = NULL;
		return g;
	}

	/* compute the index */
	/* step 1: compute the histogram of cell frequencies */
	fxa = ((double)g->cellsPerAxis/(double)(g->xmax-g->xmin));
	fxb = (double)g->xmin*fxa;
	fya = ((double)g->cellsPerAxis/(double)(g->ymax-g->ymin));
	fyb = (double)g->ymin*fya;

	cnt = BATcount(bx);
	for (i = 0; i < cnt; i++) {
		oid cellx = (double)xVals[i]*fxa - fxb;
		oid celly = (double)yVals[i]*fya - fyb;
		oid cell = ((cellx << g->shift) | celly);
		g->dir[cell+1]++;
	}

	/* step 2: compute the directory pointers */
	for (i = 1; i < g->cellsNum; i++)
		g->dir[i] += g->dir[i-1];

	/* step 3: fill in the oid array */
	for (size_t i = 0; i < cnt; i++) {
		oid cellx = (double)xVals[i]*fxa - fxb;
		oid celly = (double)yVals[i]*fya - fyb;
		oid cell = ((cellx << g->shift) | celly);
		g->oids[g->dir[cell]++] = i;
	}

	/* step 4: adjust the directory pointers */
	for (size_t i = g->cellsNum; i > 0; i--)
		g->dir[i+1] = g->dir[i];
	g->dir[0] = 0;

	/* TODO: move here the code for compressing the index */

	return g;
}

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

str
GRIDdistance(bit * res, lng * x1, lng * y1, lng * x2, lng * y2, dbl * d)
{
	str r = MAL_SUCCEED;

	if ((res = GDKmalloc(sizeof(bit))) == NULL)
		r = createException(MAL, "grid.distance", MAL_MALLOC_FAIL);
	else {
		*res = ((*x2-*x1)*(*x2-*x1)+(*y2-*y1)*(*y2-*y1))<(*d)*(*d);
	}

	return r;
}

str
GRIDdistancesubselect(bat * res, bat * x1, bat * y1, bat * cand1, lng * x2, lng * y2, dbl * d, bit * anti)
{
	size_t minCellx, minCelly, maxCellx, maxCelly, cellx, celly;
	size_t *borderCells, *internalCells;
	size_t borderCellsNum, internalCellsNum, totalCellsNum;
	size_t i, j, bvsize, num;
	uint64_t * bv, * cbv;
	double fxa, fxb, fya, fyb;
	BAT *x1BAT = NULL, *y1BAT = NULL, *cBAT = NULL;
	lng * x1Vals = NULL, * y1Vals = NULL;
	oid * resVals = NULL;
	oid seq;
	Grid * g = NULL;
	mbr mbb;
	BAT *r;
	BUN p, q;
	BATiter pi;
	BUN resNum = 0;
	assert (*d > 0);

	/* get the X and Y BATs*/
	if((x1BAT = BATdescriptor(*x1)) == NULL)
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	if((y1BAT = BATdescriptor(*y1)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	}
	if((cBAT = BATdescriptor(*cand1)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	}

	/* check if the BATs have dense heads and are aligned */
	if (!BAThdense(x1BAT) || !BAThdense(y1BAT)) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		BBPunfix(cBAT->batCacheid);
		return createException(MAL, "grid.distance", "BATs must have dense heads");
	}
	if(x1BAT->hseqbase != y1BAT->hseqbase
		|| BATcount(x1BAT) != BATcount(y1BAT)
		|| x1BAT->hseqbase != cBAT->hseqbase) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		BBPunfix(cBAT->batCacheid);
		return createException(MAL, "grid.distance", "BATs must be aligned");
	}
	num = BATcount(x1BAT);
	seq = x1BAT->hseqbase;

	assert(x1BAT->ttype == TYPE_lng);
	assert(y1BAT->ttype == TYPE_lng);
	x1Vals = (lng*)Tloc(x1BAT, BUNfirst(x1BAT));
	y1Vals = (lng*)Tloc(y1BAT, BUNfirst(y1BAT));

	/* initialize the bit vectors */
	bvsize = (num >> SHIFT) + ((num & ONES) > 0);
	if ((bv = GDKmalloc(bvsize * sizeof(uint64_t))) == NULL)
		throw(MAL, "grid.distance", MAL_MALLOC_FAIL);
	for (i = 0; i < bvsize; i++)
		bv[i] = 0;

	if ((cbv = GDKmalloc(bvsize * sizeof(uint64_t))) == NULL) {
		GDKfree(bv);
		throw(MAL, "grid.distance", MAL_MALLOC_FAIL);
	}
	for (i = 0; i < bvsize; i++)
		cbv[i] = 0;

	pi = bat_iterator(cBAT);
	BATloop(cBAT, p, q) {
		oid o = *(oid*)BUNtail(pi, p) - seq;
		size_t blockNum = o >> SHIFT;
		uint64_t bitPos = o & ONES;
		set(cbv[blockNum], bitPos, 1);
	}
	BBPunfix(cBAT->batCacheid);

	/* compute the grid index */
	if((g = grid_create(x1BAT, y1BAT)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		GDKfree(bv);
		GDKfree(cbv);
		return createException(MAL, "grid.distance", "Could not compute the grid index");
	}

	/* find which cells have to be examined */
	fxa = ((double)g->cellsPerAxis/(double)(g->xmax-g->xmin));
	fxb = (double)g->xmin*fxa; 
	fya = ((double)g->cellsPerAxis/(double)(g->ymax-g->ymin));
	fyb = (double)g->ymin*fya; 
	//mbb = (mbr) {.xmin = 0, .ymin = 0, .xmax = 0, .ymax = 0}
	mbb = (mbr) { .xmin = *x2 - *d, .ymin = *y2 - *d, .xmax = *x2 + *d, .ymax = *y2 + *d};
	if (mbb.xmin > g->xmax || mbb.xmax < g->xmin ||
		mbb.ymin > g->ymax || mbb.ymax < g->ymin) {

		/* no results */
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		GDKfree(bv);
		GDKfree(cbv);
		if ((r = BATnew(TYPE_void, TYPE_oid, 0, TRANSIENT)) == NULL)
			return createException(MAL, "grid.distance", "could not create a BAT for storing the results");
		*res = r->batCacheid;
		BBPkeepref(*res);

		return MAL_SUCCEED;
	}

	mbb.xmin = (mbb.xmin < g->xmin) ? g->xmin : mbb.xmin;
	mbb.xmax = (mbb.xmax > g->xmax) ? g->xmax : mbb.xmax; 
	mbb.ymin = (mbb.ymin < g->ymin) ? g->ymin : mbb.ymin;
	mbb.ymax = (mbb.ymax > g->ymax) ? g->ymax : mbb.ymax; 

	minCellx = (double)(mbb.xmin)*fxa - fxb;
	maxCellx = (double)(mbb.xmax)*fxa - fxb;
	minCelly = (double)(mbb.ymin)*fya - fyb;
	maxCelly = (double)(mbb.ymax)*fya - fyb;

	/* split the cells in border and internal ones */
	totalCellsNum = (maxCellx - minCellx + 1)*(maxCelly - minCelly + 1);
	borderCellsNum = (maxCellx - minCellx + 1) + (maxCelly - minCelly + 1) - 1; /* per axis, remove the corner cell that has been added twice */
	if(maxCellx > minCellx && maxCelly > minCelly)
		borderCellsNum = borderCellsNum*2 - 2; /* subtract the two corner cells that have been added twice */
	internalCellsNum = totalCellsNum - borderCellsNum;

	if((borderCells = (size_t*)GDKmalloc((borderCellsNum + 1) * sizeof(size_t*))) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		GDKfree(bv);
		GDKfree(cbv);
		return createException(MAL, "grid.distance", MAL_MALLOC_FAIL);
	}
	if((internalCells = (size_t*)GDKmalloc((internalCellsNum + 1) * sizeof(size_t*))) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		GDKfree(borderCells);
		GDKfree(bv);
		GDKfree(cbv);
		return createException(MAL, "grid.distance", MAL_MALLOC_FAIL);
	}

	borderCellsNum = 0;
	internalCellsNum = 0;
	for(cellx = minCellx ; cellx <= maxCellx; cellx++) {
		for(celly = minCelly ; celly <= maxCelly ; celly++) {
			size_t cellId = (cellx << g->shift) | celly;
			unsigned short border = (cellx == minCellx) | (cellx == maxCellx) | (celly == minCelly) | (celly == maxCelly);
			borderCells[borderCellsNum] = cellId;
			internalCells[internalCellsNum] = cellId;
			borderCellsNum += border;
			internalCellsNum += 1 - border;
		}
	}

	/* process cells along the border */
	for (i = 0; i < borderCellsNum; i++) {
		size_t cellId = borderCells[i];
		size_t offsetStartIdx = g->dir[cellId];
		size_t offsetEndIdx = g->dir[cellId+1]; /* exclusive */
		for(j = offsetStartIdx; j < offsetEndIdx; j++) {
			size_t o = g->oids[j];
			size_t blockNum = o >> SHIFT;
			uint64_t bitPos = o & ONES;
			size_t mask = (mbb.xmin <= x1Vals[o]) & (mbb.xmax >= x1Vals[o]) & 
				(mbb.ymin <= y1Vals[o]) & (mbb.ymax >= y1Vals[o]);
			set(bv[blockNum], bitPos, mask);
		}
	} 
	GDKfree(borderCells);
	BBPunfix(x1BAT->batCacheid);
	BBPunfix(y1BAT->batCacheid);

	/* process internal cells */
	for (i = 0; i < internalCellsNum; i++) {
		size_t cellId = internalCells[i];
		size_t offsetStartIdx = g->dir[cellId];
		size_t offsetEndIdx = g->dir[cellId+1]; /* exclusive */
		for(j = offsetStartIdx; j < offsetEndIdx; j++) {
			size_t o = g->oids[j];
			size_t blockNum = o >> SHIFT;
			uint64_t bitPos = o & ONES;
			set(bv[blockNum], bitPos, 0x01);
		}
	}
	GDKfree(internalCells);
	GDKfree(g->oids);
	GDKfree(g->dir);
	GDKfree(g);

	/* anti */
	if (*anti) {
		for (i = 0; i < bvsize; i++)
			bv[i] = ~bv[i];
		for (i = num % BITSNUM; i < BITSNUM; i++)
			set(bv[bvsize-1], i, 0);
	}

	/* & the bit vectors*/
	for (i = 0; i < bvsize; i++)
		bv[i] &= cbv[i];
	GDKfree(cbv);

	/* allocate a BAT for the results */
	resNum = countSetBits(bv, bvsize);
	if ((r = BATnew(TYPE_void, TYPE_oid, resNum+1, TRANSIENT)) == NULL) {
		GDKfree(bv);
		return createException(MAL, "grid.distance", "could not create a BAT for storing the results");
	}

	/* project the bit vector */
	resVals = (oid*)Tloc(r, BUNfirst(r));
	j = 0;
	for(i = 0; i < bvsize; i++) {
		uint64_t b = bv[i];
		oid o = i * BITSNUM + seq;
		for(short l = 0; l < BITSNUM; l++) {
			resVals[j] = o;
			j += b & 0x01;
			b >>= 1;
			o++;
		}
	}

	GDKfree(bv);
	//BATderiveProps(r, false);
	BATsetcount(r, resNum);
	//BATseqbase(r, 0);
	r->tsorted = true;
	r->trevsorted = false;
	*res = r->batCacheid;
	BBPkeepref(*res);

	return MAL_SUCCEED;
}

str
GRIDdistancesubjoin(bat *res1, bat * res2,bat * x1, bat * y1, bat * x2, bat * y2, dbl * d, bat * s1, bat * s2, bit * nil, lng * estimate)
{
	size_t num1, num2;
	size_t minCellx, minCelly, maxCellx, maxCelly;
	BAT *x1BAT = NULL, *y1BAT = NULL, *x2BAT = NULL, *y2BAT = NULL;
	lng * x1Vals = NULL, * y1Vals = NULL, *x2Vals = NULL, *y2Vals = NULL;
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
	assert(x1BAT->ttype == TYPE_lng);
	assert(y1BAT->ttype == TYPE_lng);
	assert(x2BAT->ttype == TYPE_lng);
	assert(y2BAT->ttype == TYPE_lng);
	num1 = BATcount(x1BAT);
	num2 = BATcount(x2BAT);
	seq1 = x1BAT->hseqbase;
	seq2 = x2BAT->hseqbase;
	x1Vals = (lng*)Tloc(x1BAT, BUNfirst(x1BAT));
	y1Vals = (lng*)Tloc(y1BAT, BUNfirst(y1BAT));
	x2Vals = (lng*)Tloc(x2BAT, BUNfirst(x2BAT));
	y2Vals = (lng*)Tloc(y2BAT, BUNfirst(y2BAT));
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

	/* perform the distance join */
	for (size_t i = minCellx; i <= maxCellx; i++) {
		for (size_t j = minCelly; j <= maxCelly; j++) {
			/* define which cells should be compared */
			size_t min = i + g1->cellsX*j;
			size_t R[] = {min,            min,              min,   min, min+1, min+g1->cellsX, min+g1->cellsX+1};
			size_t S[] = {min+g1->cellsX, min+g1->cellsX+1, min+1, min, min,   min,            min             };
			for (size_t k = 0; k < 7; k++) {
				if (GRIDcount(g1,R[k]) > GRIDcount(g2,S[k])) {
					GRIDcmp(x1Vals, y1Vals, g1, x2Vals, y2Vals, g2, R[k], S[k], r1, r2, seq1, seq2, msg);
				} else {
					GRIDcmp(x2Vals, y2Vals, g2, x1Vals, y1Vals, g1, S[k], R[k], r2, r1, seq2, seq1, msg);
				}
			}
		}
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
