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

#define CONCAT2(a,b) a##_##b
#define U2(a,b)      CONCAT2(a,b)

static Grid *
U2(grid_create,TP) (BAT *bx, BAT *by)
{
	Grid * g;
	TP *xVals, *yVals;
	size_t i, cnt;
	dbl fxa, fxb, fya, fyb;

	assert(BATcount(bx) == BATcount(by));
	assert(BATcount(bx) > 0);

	if ((g = GDKmalloc(sizeof(Grid))) == NULL)
		return g;

	g->xbat = bx->batCacheid;
	g->ybat = by->batCacheid;
	xVals = (TP*)Tloc(bx, BUNfirst(bx));
	yVals = (TP*)Tloc(by, BUNfirst(by));

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
		dbl val = (dbl)xVals[i];
		if(g->xmin > val)
			g->xmin = val;
		if(g->xmax < val)
			g->xmax = val;
	}
	g->ymin = g->ymax = yVals[0];
	for (i = 1; i < cnt; i++) {
		dbl val = (dbl)yVals[i];
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
		oid cellx = (dbl)xVals[i]*fxa - fxb;
		oid celly = (dbl)yVals[i]*fya - fyb;
		oid cell = ((cellx << g->shift) | celly);
		g->dir[cell+1]++;
	}

	/* step 2: compute the directory pointers */
	for (i = 1; i < g->cellsNum; i++)
		g->dir[i] += g->dir[i-1];

	/* step 3: fill in the oid array */
	for (size_t i = 0; i < cnt; i++) {
		oid cellx = (dbl)xVals[i]*fxa - fxb;
		oid celly = (dbl)yVals[i]*fya - fyb;
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

#undef U2
#undef CONCAT2
