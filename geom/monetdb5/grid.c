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

#define maximumNumberOfCells(max, bitsNum, add) \
do {                                            \
    int i = 0;                                  \
    size_t res = 0;                             \
    for(i = 0; i < bitsNum; i++)                \
        res = (res << 1) | 1;                   \
    max = res + add;                            \
} while (0)

typedef struct grid {
	lng xmin;			/* grid universe: minimum X value */
	lng ymin;			/* grid universe: minimum Y value */
	lng xmax;			/* grid universe: maximum X value */
	lng ymax;			/* grid universe: maximum Y value */
	bte shift;
	size_t cellsNum;		/* number of cells                 */
	size_t cellsPerAxis;	/* number of cells per axis        */
	bat xbat;				/* bat id for X coordinates        */
	bat ybat;				/* bat id for Y coordinates        */
	size_t * dir;			/* the grid directory              */
	oid * oids;			/* heap where the index is stored  */
} grid;

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

static grid *
grid_create(BAT *bx, BAT *by)
{
	grid * g;
	lng *xVals, *yVals;
	size_t i, cnt;
	dbl fxa, fxb, fya, fyb;

	assert(BATcount(bx) == BATcount(by));
	assert(BATcount(bx) > 0);

	if ((g = GDKmalloc(sizeof(grid))) == NULL)
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
	if ((g->dir = GDKmalloc((g->cellsNum+1)*sizeof(size_t))) == NULL) {
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
//		assert(cell < g->cellsNum);
//		assert(position < offsetVals[g->cellsNum]);
		g->oids[g->dir[cell]++] = i;
	}

	/* step 4: adjust the directory pointers */
	for (size_t i = g->cellsNum; i > 0; i--)
		g->dir[i+1] = g->dir[i];
	g->dir[0] = 0;

	/* TODO: move here the code for compressing the index */

	return g;
}

str
GRIDdistance(bit * res, lng * x1, lng * y1, lng * x2, lng * y2, int * d)
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
GRIDdistancesubselect(bat * res, bat * x1, bat * y1, bat * cand1, lng * x2, lng * y2, int * d, bit * anti)
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
	grid * g = NULL;
	mbr mbb = (mbr) {.xmin = *x2 - *d, .ymin = *y2 - *d, .xmax = *x2 + *d, .ymax = *y2 + *d};
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
	if((y1BAT = BATdescriptor(*y1)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	}
	if((cBAT = BATdescriptor(*cand1)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	}
	num = BATcount(x1BAT);

	/* check if the BATs have dense heads and are aligned */
	if (!BAThdense(x1BAT) || !BAThdense(y1BAT)) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		BBPunfix(cBAT->batCacheid);
		return createException(MAL, "grid.distance", "BATs must have dense heads");
	}
	if(x1BAT->hseqbase != y1BAT->hseqbase || BATcount(x1BAT) != BATcount(y1BAT)) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		BBPunfix(cBAT->batCacheid);
		return createException(MAL, "grid.distance", "BATs must be aligned");
	}

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
		oid o = *(oid*)BUNtail(pi, p);
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

	minCellx = (double)(mbb.xmin<g->xmin?g->xmin:mbb.xmin)*fxa - fxb;
	maxCellx = (double)(mbb.xmax>g->xmax?g->xmax:mbb.xmax)*fxa - fxb;
	minCelly = (double)(mbb.ymin<g->ymin?g->ymin:mbb.ymin)*fya - fyb;
	maxCelly = (double)(mbb.ymax>g->ymax?g->ymax:mbb.ymax)*fya - fyb;

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
		oid o = i * BITSNUM;
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
	r->tsorted = true;
	r->trevsorted = false;
	*res = r->batCacheid;
	BBPkeepref(*res);

	return MAL_SUCCEED;
}

str
GRIDdistancesubjoin(bat *res1, bat * res2,bat * x1, bat * y1, bat * x2, bat * y2, int * distance, bat * s1, bat * s2, bit * nil, lng * estimate)
{
	(void)res1;
	(void)res2;
	(void)x1;
	(void)y1;
	(void)x2;
	(void)y2;
	(void)s1;
	(void)s2;
	(void)distance;
	(void)nil;
	(void)estimate;

	return MAL_SUCCEED;
}

/* The commented version below:
 * - does not use a bit vector for storing the results
 * - does not use candidates 
 * - ignores the anti parameter 
 * - the index is stored as an OID array */
#if 0
str
GRIDdistancesubselect(bat * res, bat * x1, bat * y1, bat * cand1, lng * x2, lng * y2, int * d, bit * anti)
{
	size_t minCellx, minCelly, maxCellx, maxCelly, cellx, celly;
	size_t *borderCells, *internalCells;
	size_t borderCellsNum, internalCellsNum, totalCellsNum;
	size_t i, j;
	double fxa, fxb, fya, fyb;
	BAT *x1BAT = NULL, *y1BAT = NULL;
	lng * x1Vals = NULL, * y1Vals = NULL, * resVals = NULL;
	grid * g = NULL;
	mbr mbb = (mbr) {.xmin = *x2 - *d, .ymin = *y2 - *d, .xmax = *x2 + *d, .ymax = *y2 + *d};
	BAT *r;
	BUN resNum = 0;
	assert (*d > 0);

	(void)anti; //TODO: anti

	/* get the X and Y BATs*/
	if((x1BAT = BATdescriptor(*x1)) == NULL)
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	if((y1BAT = BATdescriptor(*y1)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		throw(MAL, "grid.distance", RUNTIME_OBJECT_MISSING);
	}
	x1Vals = (lng*)Tloc(x1BAT, BUNfirst(x1BAT));
	y1Vals = (lng*)Tloc(y1BAT, BUNfirst(y1BAT));

	/* check if the BATs have dense heads and are aligned */
	if (!BAThdense(x1BAT) || !BAThdense(y1BAT)) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		return createException(MAL, "grid.distance", "BATs must have dense heads");
	}
	if(x1BAT->hseqbase != y1BAT->hseqbase || BATcount(x1BAT) != BATcount(y1BAT)) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		return createException(MAL, "grid.distance", "BATs must be aligned");
	}

	/* compute the grid index */
	if((g = grid_create(x1BAT, y1BAT)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		return createException(MAL, "grid.distance", "Could not compute the grid index");
	}

	/* find which cells have to be examined */
	fxa = ((double)g->cellsPerAxis/(double)(g->xmax-g->xmin));
	fxb = (double)g->xmin*fxa; 
	fya = ((double)g->cellsPerAxis/(double)(g->ymax-g->ymin));
	fyb = (double)g->ymin*fya; 

	minCellx = (double)(mbb.xmin<g->xmin?g->xmin:mbb.xmin)*fxa - fxb;
	maxCellx = (double)(mbb.xmax>g->xmax?g->xmax:mbb.xmax)*fxa - fxb;
	minCelly = (double)(mbb.ymin<g->ymin?g->ymin:mbb.ymin)*fya - fyb;
	maxCelly = (double)(mbb.ymax>g->ymax?g->ymax:mbb.ymax)*fya - fyb;

	/* split the cells in border and internal ones */
	totalCellsNum = (maxCellx - minCellx + 1)*(maxCelly - minCelly + 1);
	borderCellsNum = (maxCellx - minCellx + 1) + (maxCelly - minCelly + 1) - 1; /* per axis, remove the corner cell that has been added twice */
	if(maxCellx > minCellx && maxCelly > minCelly)
		borderCellsNum = borderCellsNum*2 - 2; /* subtract the two corner cells that have been added twice */
	internalCellsNum = totalCellsNum - borderCellsNum;

	if((borderCells = (size_t*)GDKmalloc((borderCellsNum + 1) * sizeof(size_t*))) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		return createException(MAL, "grid.distance", MAL_MALLOC_FAIL);
	}
	if((internalCells = (size_t*)GDKmalloc((internalCellsNum + 1) * sizeof(size_t*))) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		GDKfree(borderCells);
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

	/* count number of results from internal cells */
	for (i = 0; i < internalCellsNum; i++)
		resNum += g->dir[i+1] - g->dir[i];

	/* count number of results from border cells */
	for (i = 0; i < borderCellsNum; i++) {
		size_t cellId = borderCells[i];
			size_t offsetStartIdx = g->dir[cellId];
			size_t offsetEndIdx = g->dir[cellId+1]; /* exclusive */
			for(j = offsetStartIdx; j < offsetEndIdx; j++) {
				size_t oid = g->oids[j];
				resNum += (mbb.xmin <= x1Vals[oid]) & (mbb.xmax >= x1Vals[oid]) & 
						  (mbb.ymin <= y1Vals[oid]) & (mbb.ymax >= y1Vals[oid]);
			}
	} 

	/* allocate a BAT for the results */
	if ((r = BATnew(TYPE_void, TYPE_oid, resNum, TRANSIENT)) == NULL) {
		BBPunfix(x1BAT->batCacheid);
		BBPunfix(y1BAT->batCacheid);
		GDKfree(borderCells);
		GDKfree(internalCells);
		return createException(MAL, "grid.distance", "could not create a BAT for storing the results");
	}
	resVals = (lng*)Tloc(r, BUNfirst(r));

	/* process cells along the border */
	resNum = 0;
	for (i = 0; i < borderCellsNum; i++) {
		size_t cellId = borderCells[i];
			size_t offsetStartIdx = g->dir[cellId];
			size_t offsetEndIdx = g->dir[cellId+1]; /* exclusive */
			for(j = offsetStartIdx; j < offsetEndIdx; j++) {
				size_t oid = g->oids[j];
				resVals[resNum] = oid;
				resNum += (mbb.xmin <= x1Vals[oid]) & (mbb.xmax >= x1Vals[oid]) & 
						  (mbb.ymin <= y1Vals[oid]) & (mbb.ymax >= y1Vals[oid]);
			}
	} 

	/* process internal cells */
	for (i = 0; i < internalCellsNum; i++) {
		size_t cellId = internalCells[i];
			size_t offsetStartIdx = g->dir[cellId];
			size_t offsetEndIdx = g->dir[cellId+1]; /* exclusive */
			memcpy(resVals+resNum, g->oids+offsetStartIdx, (offsetEndIdx-offsetStartIdx)*sizeof(oid));
	}
	GDKqsort(resVals, NULL, NULL, (size_t) resNum, sizeof(oid), 0, TYPE_oid);
	BATsetcount(r, resNum);

	/* clean up */
	BBPunfix(x1BAT->batCacheid);
	BBPunfix(y1BAT->batCacheid);
	GDKfree(borderCells);
	GDKfree(internalCells);
	GDKfree(g->oids);
	GDKfree(g->dir);
	GDKfree(g);
	//BATderiveProps(r, false);
	r->tsorted = true;
	r->trevsorted = false;
	*res = r->batCacheid;
	BBPkeepref(*res);


	return MAL_SUCCEED;
}
#endif
