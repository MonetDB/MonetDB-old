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


geom_export str GRIDdistance(bit * res, lng * x1, lng * y1, lng * x2, lng * y2, int * distance);
geom_export str GRIDdistancesubselect(bat * res, bat * x1, bat * y1, bat * cand1, lng * x2, lng * y2, int * distance, bit * anti);
geom_export str GRIDdistancesubjoin(bat *res1, bat * res2,bat * x1, bat * y1, bat * x2, bat * y2, int * distance, bat * s1, bat * s2, bit * nil, lng * estimate);
