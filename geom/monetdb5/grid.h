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

#define GRID_distance_scalar(TPE) \
	geom_export str GRIDdistance_##TPE (bit * res, TPE * x1, TPE * y1, TPE * x2, TPE * y2, dbl * distance)
#define GRID_distancesubselect_scalar(TPE) \
	geom_export str GRIDdistancesubselect_##TPE (bat * res, bat * x1, bat * y1, bat * cand1, TPE * x2, TPE * y2, dbl * distance, bit * anti)

GRID_distance_scalar(bte);
GRID_distancesubselect_scalar(bte);
GRID_distance_scalar(sht);
GRID_distancesubselect_scalar(sht);
GRID_distance_scalar(int);
GRID_distancesubselect_scalar(int);
GRID_distance_scalar(lng);
GRID_distancesubselect_scalar(lng);
#ifdef HAVE_HGE
GRID_distance_scalar(hge);
GRID_distancesubselect_scalar(hge);
#endif
GRID_distance_scalar(flt);
GRID_distancesubselect_scalar(flt);
GRID_distance_scalar(dbl);
GRID_distancesubselect_scalar(dbl);

geom_export str GRIDdistancesubjoin(bat *res1, bat * res2,bat * x1, bat * y1, bat * x2, bat * y2, dbl * distance, bat * s1, bat * s2, bit * nil, lng * estimate);
