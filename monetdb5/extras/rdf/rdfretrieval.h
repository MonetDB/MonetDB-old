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

#ifndef _RDFRETRIEVAL_H_
#define _RDFRETRIEVAL_H_

#include "rdfschema.h"

typedef struct NodeStat {
	int	weight;		// summed weights of all nodes on the path
	int	steps;		// length of the path
	int	predecessor;	// last node on the path to this node
	int	origWeight;	// = CS frequency
} NodeStat;

#define SUBSCHEMA_HEURISTIC 4

int* retrieval(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, long int* adjacency_freq, int adjacencyCount);

#endif /* _RDFRETRIEVAL_H_ */
