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

str
GRIDdistance(bit * res, lng * x1, lng * y1, lng * x2, lng * y2, int * distance)
{
	(void)res;
	(void)x1;
	(void)y1;
	(void)x2;
	(void)y2;
	(void)distance;

	return MAL_SUCCEED;
}

str
GRIDdistancesubselect(bat * res, bat * x1, bat * y1, bat * cand1, lng * x2, lng * y2, int * distance, bit * anti)
{
	(void)res;
	(void)x1;
	(void)y1;
	(void)cand1;
	(void)x2;
	(void)y2;
	(void)distance;
	(void)anti;

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
