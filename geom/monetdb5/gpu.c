/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/*
 * @a Romulo Goncalves
 * @* The simple geom module
 */

#include "gpu.h"
#define GEOMBULK_DEBUG

/*TODO: better conversion from WKB*/
/*TODO: Check if the allocations are working*/
static str
getVerts(wkb *geom, vertexWKBF **res)
{
	str err = NULL;
	str geom_str = NULL, str_pt = NULL;
	char *str2, *token, *subtoken;
	char *saveptr1 = NULL, *saveptr2 = NULL;
    vertexWKBF *verts = NULL;

    /*Check if it is a Polygon*/

	if ((err = wkbAsText(&geom_str, &geom, NULL)) != MAL_SUCCEED) {
		return err;
	}

    str_pt = geom_str;
    if ( (verts = (vertexWKBF*) GDKzalloc(sizeof(vertexWKBF))) == NULL) {
        throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
    }

	geom_str = strchr(geom_str, '(');
	geom_str += 2;

	/*Lets get the polygon */
	token = strtok_r(geom_str, ")", &saveptr1);
    if ( (verts->vert_x = GDKmalloc(POLY_NUM_VERT * sizeof(float))) == NULL) {
        GDKfree(verts);
        throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
    }
	if ( (verts->vert_y = GDKmalloc(POLY_NUM_VERT * sizeof(float))) == NULL) {
        GDKfree(verts->vert_x);
        GDKfree(verts);
        throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
    }

	for (str2 = token;; str2 = NULL) {
		subtoken = strtok_r(str2, ",", &saveptr2);
		if (subtoken == NULL)
			break;
		//sscanf(subtoken, "%lf %lf", &(verts->vert_x[verts->nvert]), &(verts->vert_y[verts->nvert]));
		sscanf(subtoken, "%f %f", &(verts->vert_x[verts->nvert]), &(verts->vert_y[verts->nvert]));
		verts->nvert++;
		if ((verts->nvert % POLY_NUM_VERT) == 0) {
			if ( (verts->vert_x = GDKrealloc(verts->vert_x, verts->nvert * 2 * sizeof(float))) == NULL) {
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }
			if ( (verts->vert_y = GDKrealloc(verts->vert_y, verts->nvert * 2 * sizeof(float))) == NULL) {
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }
		}
	}

	token = strtok_r(NULL, ")", &saveptr1);
	if (token) {
		if ( (verts->holes_x = GDKzalloc(POLY_NUM_HOLE * sizeof(float *))) == NULL) {
            GDKfree(verts->vert_x);
            GDKfree(verts->vert_y);
            GDKfree(verts);
            throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
        }
		if ( (verts->holes_y = GDKzalloc(POLY_NUM_HOLE * sizeof(float *))) == NULL) {
            GDKfree(verts->holes_x);
            GDKfree(verts->vert_x);
            GDKfree(verts->vert_y);
            GDKfree(verts);
            throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
        }
		if ( (verts->holes_n = GDKzalloc(POLY_NUM_HOLE * sizeof(int *))) == NULL) {
            GDKfree(verts->holes_x);
            GDKfree(verts->holes_y);
            GDKfree(verts->vert_x);
            GDKfree(verts->vert_y);
            GDKfree(verts);
            throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
        }
	}
	/*Lets get all the holes */
	while (token) {
		int nhole = 0;
		token = strchr(token, '(');
		if (!token)
			break;
		token++;

		if (!verts->holes_x[verts->nholes])
			if ( (verts->holes_x[verts->nholes] = GDKzalloc(POLY_NUM_VERT * sizeof(float))) == NULL) {
                GDKfree(verts->holes_x);
                GDKfree(verts->holes_y);
                GDKfree(verts->holes_n);
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }
		if (!verts->holes_y[verts->nholes])
			if ( (verts->holes_y[verts->nholes] = GDKzalloc(POLY_NUM_VERT * sizeof(float))) == NULL) {
                GDKfree(verts->holes_x[verts->nholes]);
                GDKfree(verts->holes_x);
                GDKfree(verts->holes_y);
                GDKfree(verts->holes_n);
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }

		for (str2 = token;; str2 = NULL) {
			subtoken = strtok_r(str2, ",", &saveptr2);
			if (subtoken == NULL)
				break;
			//sscanf(subtoken, "%lf %lf", &(verts->holes_x[verts->nholes][nhole]), &(verts->holes_y[verts->nholes][nhole]));
			sscanf(subtoken, "%f %f", &(verts->holes_x[verts->nholes][nhole]), &(verts->holes_y[verts->nholes][nhole]));
			nhole++;
			if ((nhole % POLY_NUM_VERT) == 0) {
                if ( (verts->holes_x[verts->nholes] = GDKrealloc(verts->holes_x[verts->nholes], nhole * 2 * sizeof(float))) == NULL) {
                    GDKfree(verts->holes_x[verts->nholes]);
                    GDKfree(verts->holes_y[verts->nholes]);
                    GDKfree(verts->holes_x);
                    GDKfree(verts->holes_y);
                    GDKfree(verts->holes_n);
                    GDKfree(verts->vert_x);
                    GDKfree(verts->vert_y);
                    GDKfree(verts);
                    throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
                }
                if ( (verts->holes_y[verts->nholes] = GDKrealloc(verts->holes_y[verts->nholes], nhole * 2 * sizeof(float))) == NULL) {
                    GDKfree(verts->holes_x[verts->nholes]);
                    GDKfree(verts->holes_y[verts->nholes]);
                    GDKfree(verts->holes_x);
                    GDKfree(verts->holes_y);
                    GDKfree(verts->holes_n);
                    GDKfree(verts->vert_x);
                    GDKfree(verts->vert_y);
                    GDKfree(verts);
                    throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
                }
			}
		}

		verts->holes_n[verts->nholes] = nhole;
        verts->nholes++;
        if ((verts->nholes % POLY_NUM_HOLE) == 0) {
            if ( (verts->holes_x = GDKrealloc(verts->holes_x, verts->nholes * 2 * sizeof(float *))) == NULL) {
                GDKfree(verts->holes_x[verts->nholes]);
                GDKfree(verts->holes_y[verts->nholes]);
                GDKfree(verts->holes_x);
                GDKfree(verts->holes_y);
                GDKfree(verts->holes_n);
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }
            if ( (verts->holes_y = GDKrealloc(verts->holes_y, verts->nholes * 2 * sizeof(float *))) == NULL) {
                GDKfree(verts->holes_x[verts->nholes]);
                GDKfree(verts->holes_y[verts->nholes]);
                GDKfree(verts->holes_x);
                GDKfree(verts->holes_y);
                GDKfree(verts->holes_n);
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }
            if ( (verts->holes_n = GDKrealloc(verts->holes_n, verts->nholes * 2 * sizeof(int))) == NULL ) {
                GDKfree(verts->holes_x[verts->nholes]);
                GDKfree(verts->holes_y[verts->nholes]);
                GDKfree(verts->holes_x);
                GDKfree(verts->holes_y);
                GDKfree(verts->holes_n);
                GDKfree(verts->vert_x);
                GDKfree(verts->vert_y);
                GDKfree(verts);
                throw(MAL, "geom.getVerts", MAL_MALLOC_FAIL);
            }
		}
		token = strtok_r(NULL, ")", &saveptr1);
	}

    if (str_pt)
        GDKfree(str_pt);

    *res = verts;
    return MAL_SUCCEED;
}

static void
freeVerts(vertexWKBF *verts)
{
    int j = 0;

	GDKfree(verts->vert_x);
	GDKfree(verts->vert_y);
	if (verts->holes_x && verts->holes_y && verts->holes_n) {
		for (j = 0; j < verts->nholes; j++) {
			GDKfree(verts->holes_x[j]);
			GDKfree(verts->holes_y[j]);
		}
		GDKfree(verts->holes_x);
		GDKfree(verts->holes_y);
		GDKfree(verts->holes_n);
	}

    GDKfree(verts);
}

static str
GpnpolyWithHoles(bat *out, int nvert, flt *vx, flt *vy, int nholes, flt **hx, flt **hy, int *hn, bat *point_x, bat *point_y)
{
	BAT *bo = NULL, *bpx = NULL, *bpy;
	flt *px = NULL, *py = NULL;
	BUN i = 0, cnt = 0;
	bit *cs = NULL;
	char **mc = NULL;
	int npoint = 0;
	float *mpx = NULL, *mpy = NULL, *mvx = NULL, *mvy = NULL;

#ifdef GEOMBULK_DEBUG
    static struct timeval start, stop;
    unsigned long long t;
#endif

	/*Get the BATs */
	if ((bpx = BATdescriptor(*point_x)) == NULL) {
		throw(MAL, "geom.point", RUNTIME_OBJECT_MISSING);
	}
	if ((bpy = BATdescriptor(*point_y)) == NULL) {
		BBPunfix(bpx->batCacheid);
		throw(MAL, "geom.point", RUNTIME_OBJECT_MISSING);
	}

	/*Check BATs alignment */
	if (bpx->hseqbase != bpy->hseqbase || BATcount(bpx) != BATcount(bpy)) {
		BBPunfix(bpx->batCacheid);
		BBPunfix(bpy->batCacheid);
		throw(MAL, "geom.point", "both point bats must have dense and aligned heads");
	}

	/*Create output BAT */
	if ((bo = COLnew(bpx->hseqbase, TYPE_bit, BATcount(bpx), TRANSIENT)) == NULL) {
		BBPunfix(bpx->batCacheid);
		BBPunfix(bpy->batCacheid);
		throw(MAL, "geom.point", MAL_MALLOC_FAIL);
	}

	/*Iterate over the Point BATs and determine if they are in Polygon represented by vertex BATs */
	px = (flt *) Tloc(bpx, 0);
	py = (flt *) Tloc(bpy, 0);
	cnt = BATcount(bpx);
	cs = (bit *) Tloc(bo, 0);
#ifdef GEOMBULK_DEBUG
    gettimeofday(&start, NULL);
#endif

	/*Call to the GPU function*/
	if (nvert && cnt)
		pnpoly_GPU(&cs, nvert, cnt, px, py, vx, vy);

	/*
	 * Verify if GPU has enough memory to get all the data 
	 * otherwise do it in waves.
	 */

	/*Lock until all the results are available*/

#ifdef GEOMBULK_DEBUG
    gettimeofday(&stop, NULL);
    t = 1000 * (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000;
    fprintf(stdout, "pnpolyWithHoles %llu ms\n", t);
#endif

	bo->tsorted = bo->trevsorted = 0;
	bo->tkey = 0;
	BATrmprops(bo)
	BATsetcount(bo, cnt);
	BATsettrivprop(bo);
	BBPunfix(bpx->batCacheid);
	BBPunfix(bpy->batCacheid);
	BBPkeepref(*out = bo->batCacheid);
	return MAL_SUCCEED;
}

str
geom_gpu_gcontains(bit *res, wkb **a, float *x, float *y, float *z, int *srid)
{
    vertexWKBF *verts = NULL;
	wkb *geom = NULL;
	str msg = NULL;
	(void) z;

	geom = (wkb *) *a;
    if ((msg = getVerts(geom, &verts)) != MAL_SUCCEED) {
        return msg;
    }
         
	//msg = GpnpolyWithHoles(out, (int) verts->nvert, verts->vert_x, verts->vert_y, verts->nholes, verts->holes_x, verts->holes_y, verts->holes_n, point_x, point_y);

    if (verts)
        freeVerts(verts);

	return msg;
}

str
geom_gpu_gcontains_bat(bat *out, wkb **a, bat *point_x, bat *point_y, bat *point_z, int *srid)
{
    vertexWKBF *verts = NULL;
	wkb *geom = NULL;
	str msg = NULL;
	(void) point_z;

	geom = (wkb *) *a;
    if ((msg = getVerts(geom, &verts)) != MAL_SUCCEED) {
        return msg;
    }
         
	msg = GpnpolyWithHoles(out, (int) verts->nvert, verts->vert_x, verts->vert_y, verts->nholes, verts->holes_x, verts->holes_y, verts->holes_n, point_x, point_y);

    if (verts)
        freeVerts(verts);

	return msg;
}

str
geom_gpu_setup(bit *res, int *flag) {
	str msg = MAL_SUCCEED;
	(void) res;

	setup_GPU();

	return msg;
}

str
geom_gpu_reset(bit *res, int *flag) {
	str msg = MAL_SUCCEED;
	(void) res;

	reset_GPU();

	return msg;
}
