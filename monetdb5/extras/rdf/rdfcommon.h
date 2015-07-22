/*
 * The contents of this file are subject to the MonetDB Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.monetdb.org/Legal/MonetDBLicense
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is the MonetDB Database System.
 *
 * The Initial Developer of the Original Code is CWI.
 * Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
 * Copyright August 2008-2013 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * This contains common functions using in 
 * RDF module. E.g., merging list, and/or list of oid
 * */
#ifndef _RDFCOMMON_H_
#define _RDFCOMMON_H_

#include "rdf.h"

rdf_export void copyOidSet(oid* dest, oid* orig, int len); 
rdf_export void copyIntSet(int* dest, int* orig, int len); 
rdf_export void copybatSet(bat *dest, bat* orig, int len); 

rdf_export void initCharArray(char* inputArr, int num, char defaultValue);
rdf_export void initArray(oid* inputArr, int num, oid defaultValue);
rdf_export void initIntArray(int* inputArr, int num, oid defaultValue);

rdf_export void getNumCombinedP(oid* arr1, oid* arr2, int m, int n, int *numCombineP);
rdf_export void mergeOidSets(oid* arr1, oid* arr2, oid* mergeArr, int m, int n, int *numCombineP);
rdf_export void intersect_oidsets(oid** lists, int* listcount, int num, oid** interlist, int *internum); 
rdf_export void intersect_intsets(int** lists, int* listcount, int num, int** interlist, int *internum); 

rdf_export void get_sorted_distinct_set(oid* src, oid** des, int numsrc, int *numdesc);

rdf_export void appendArrayToBat(BAT *b, BUN* inArray, int num);
rdf_export void appendIntArrayToBat(BAT *b, int* inArray, int num);
rdf_export void appendbatArrayToBat(BAT *b, bat* inArray, int num);

#endif /* _RDFCOMMON_H_ */
