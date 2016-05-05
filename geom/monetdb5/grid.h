/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/*
 * @a Kostis Kyzirakos, Foteini Alvanaki
 */

#include <monetdb_config.h>
#include <geom.h>

#ifdef WIN32
#ifndef LIBGEOM
#define geom_export extern __declspec(dllimport)
#else
#define geom_export extern __declspec(dllexport)
#endif
#else
#define geom_export extern
#endif

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

#define GRID_dist(TP1,TP2) \
str GRIDdistance_##TP1##_##TP2 (bit * res, TP1 * x1, TP1 * y1, TP2 * x2, TP2 * y2, dbl * d); \
str GRIDdistancesubselect_##TP1##_##TP2 (bat * res, bat * x1, bat * y1, bat * cand1, TP2 * x2, TP2 * y2, dbl * d, bit * anti);

GRID_dist(bte,bte);
GRID_dist(bte,sht);
GRID_dist(bte,int);
GRID_dist(bte,lng);
GRID_dist(bte,flt);
GRID_dist(bte,dbl);
GRID_dist(sht,bte);
GRID_dist(sht,sht);
GRID_dist(sht,int);
GRID_dist(sht,lng);
GRID_dist(sht,flt);
GRID_dist(sht,dbl);
GRID_dist(int,bte);
GRID_dist(int,sht);
GRID_dist(int,int);
GRID_dist(int,lng);
GRID_dist(int,flt);
GRID_dist(int,dbl);
GRID_dist(lng,bte);
GRID_dist(lng,sht);
GRID_dist(lng,int);
GRID_dist(lng,lng);
GRID_dist(lng,flt);
GRID_dist(lng,dbl);
GRID_dist(flt,bte);
GRID_dist(flt,sht);
GRID_dist(flt,int);
GRID_dist(flt,lng);
GRID_dist(flt,flt);
GRID_dist(flt,dbl);
GRID_dist(dbl,bte);
GRID_dist(dbl,sht);
GRID_dist(dbl,int);
GRID_dist(dbl,lng);
GRID_dist(dbl,flt);
GRID_dist(dbl,dbl);
#ifdef HAVE_HGE
GRID_dist(bte,hge);
GRID_dist(sht,hge);
GRID_dist(int,hge);
GRID_dist(lng,hge);
GRID_dist(hge,bte);
GRID_dist(hge,sht);
GRID_dist(hge,int);
GRID_dist(hge,lng);
GRID_dist(hge,hge);
GRID_dist(hge,flt);
GRID_dist(hge,dbl);
GRID_dist(flt,hge);
GRID_dist(dbl,hge);
#endif

// geom_export str GRIDdistance(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
// geom_export str GRIDdistancesubselect(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
geom_export str GRIDdistancesubjoin(bat *res1, bat * res2,bat * x1, bat * y1, bat * x2, bat * y2, dbl * distance, bat * s1, bat * s2, bit * nil, lng * estimate);
