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

#include "monetdb_config.h"
#include "rdf.h"
#include "rdfretrieval.h"
#include "rdfschema.h"
#include "rdflabels.h"

static
int edgeExists(long int from, long int to, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	int	i;
	for (i = 0; i < adjacencyCount; ++i) {
		if (adjacency_from[i] == from && adjacency_to[i] == to) return 1;
	}
	return 0;
}

static
NodeStat* initNodeStats1(long int* table_freq, int tableCount) {
	NodeStat*	nodeStats = NULL;
	int		i;

	nodeStats = (NodeStat *) malloc(sizeof(NodeStat) * tableCount);
	if (!nodeStats) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < tableCount; ++i) {
		nodeStats[i].origWeight = table_freq[i];
		nodeStats[i].weight = table_freq[i]; // weight = origWeight
		nodeStats[i].steps = -1;
		nodeStats[i].predecessor = -1;
	}

	return nodeStats;
}

static
void bfs1(int root, long int* table_id, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount, int* queue, int* visited, int* isInQueue, int* queuePosition, int* queueLength, NodeStat* nodeStats) {
	int	i;

	for (i = 0; i < tableCount; ++i) {
		if (edgeExists(table_id[root], table_id[i], adjacency_from, adjacency_to, adjacencyCount)) {
			if (nodeStats[i].steps == -1) {
				// no previous path to this node
				nodeStats[i].weight = nodeStats[root].weight + nodeStats[i].origWeight;
				nodeStats[i].steps = nodeStats[root].steps + 1;
				nodeStats[i].predecessor = root;
			} else {
				// previous path to this node
				// test if values should be updated

				// cycle detection
				int cycle = 0;
				int pathId = root;
				while (pathId != -1) {
					if (pathId == i) {
						// cycle found
						cycle = 1;
						break;
					}
					pathId = nodeStats[pathId].predecessor;
				}
				if (cycle) continue;

				if (nodeStats[i].predecessor == root) {
					// path to 'i' used 'root', has to be updated if the weight changes
					if (((float) (nodeStats[root].weight + nodeStats[i].origWeight)) / (nodeStats[root].steps + 1) != ((float) nodeStats[i].weight) / nodeStats[i].steps) {
						// set new weight and path
						nodeStats[i].weight = nodeStats[root].weight + nodeStats[i].origWeight;
						nodeStats[i].steps = nodeStats[root].steps + 1;
						nodeStats[i].predecessor = root;
						// update values for subsequent nodes
						visited[i] = 0;
					}
				} else if (((float) (nodeStats[root].weight + nodeStats[i].origWeight)) / (nodeStats[root].steps + 1) > ((float) nodeStats[i].weight) / nodeStats[i].steps) {
					// improved weight when accessing node 'i' via 'root'
					// set new weight and path
					nodeStats[i].weight = nodeStats[root].weight + nodeStats[i].origWeight;
					nodeStats[i].steps = nodeStats[root].steps + 1;
					nodeStats[i].predecessor = root;
					// update values for subsequent nodes
					visited[i] = 0;
				}
			}

			if (!visited[i] && !isInQueue[i]) {
				// add to queue
				queue[((*queueLength + *queuePosition) % tableCount)] = i;
				*queueLength += 1;
				isInQueue[i] = 1;
			}

		}
	}

	if (*queueLength > 0) {
		visited[queue[(*queuePosition % tableCount)]] = 1;
		isInQueue[queue[(*queuePosition % tableCount)]] = 0;
		*queuePosition += 1;
		*queueLength -= 1;
		bfs1(queue[((*queuePosition + tableCount - 1) % tableCount)], table_id, tableCount, adjacency_from, adjacency_to, adjacencyCount, queue, visited, isInQueue, queuePosition, queueLength, nodeStats);
	}
}

static
void addNode1(long int* adjacency_from, long int* adjacency_to, int adjacencyCount, NodeStat* nodeStats, long int* table_id, int tableCount, int root, char initial) {
	int	queue[tableCount]; // cyclic array
	int	visited[tableCount];
	int	isInQueue[tableCount];
	int	queuePosition; // next element in queue to view at
	int	queueLength;
	int	pathId, pathIdTmp;
	int	i;

	// init
	for (i = 0; i < tableCount; ++i) {
		queue[i] = -1;
		visited[i] = 0;
		isInQueue[i] = 0;
	}
	visited[root] = 1;
	queuePosition = 0;
	queueLength = 0;

	if (initial) {
		// mark root as a "chosen node"
		nodeStats[root].steps = 0;
		nodeStats[root].predecessor = -1;
		nodeStats[root].weight = 0;
	} else {
		// add nodes on path to queue
		int steps = nodeStats[root].steps;
		int i = 0;

		pathId = root;
		while (pathId != -1) {
			++i;
			pathIdTmp = nodeStats[pathId].predecessor; // save predecessor

			// mark node as a "chosen node"
			nodeStats[pathId].steps = 0;
			nodeStats[pathId].predecessor = -1;
			nodeStats[pathId].weight = 0;

			if (nodeStats[pathIdTmp].steps == 0) break; // found the end of the path of new nodes
			queue[queueLength] = pathIdTmp;
			queueLength += 1;
			isInQueue[pathIdTmp] = 1;

			pathId = pathIdTmp; // move to predecessor
		}
		assert(steps == i);
	}

	bfs1(root, table_id, tableCount, adjacency_from, adjacency_to, adjacencyCount, queue, visited, isInQueue, &queuePosition, &queueLength, nodeStats);
}

static
int* retrieval1(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	NodeStat	*nodeStats = NULL;
	int		numNodes;
	int		*chosenNodes = NULL;
	int		i, j;
	int		sumSubjects = 0;
	int		csCount = 0;
	int		sumChosenSubjects = 0;

	if (numNodesMax < 1) fprintf(stderr, "ERROR: numNodesMax < 1!\n");

	chosenNodes = (int *) malloc(sizeof(int) * numNodesMax);
	if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	nodeStats = initNodeStats1(table_freq, tableCount);
	numNodes = 1;

	// add root node
	addNode1(adjacency_from, adjacency_to, adjacencyCount, nodeStats, table_id, tableCount, root, 1);

	// add nodes
	while (numNodes < numNodesMax) {
		// get top node (highest fraction (weight/steps))
		int top = -1;
		for (i = 0; i < tableCount; ++i) {
			int topWeight, testWeight;
			if (nodeStats[i].steps == -1) continue; // non-reachable
			if (nodeStats[i].steps == 0) continue; // already chosen
			if (numNodes + nodeStats[i].steps > numNodesMax) continue; // path too long
			if (top == -1) {
				top = i; // set first value
				continue;
			}
			topWeight = ((float) nodeStats[top].weight) / nodeStats[top].steps;
			testWeight = ((float) nodeStats[i].weight) / nodeStats[i].steps;
			if (testWeight > topWeight) {
				top = i;
			}
		}
		if (top == -1) break; // not enough nodes found
		numNodes += nodeStats[top].steps;
		addNode1(adjacency_from, adjacency_to, adjacencyCount, nodeStats, table_id, tableCount, top, 0); // add node(s)
	}

	// store list of chosen nodes
	printf("SUBSCHEMA:\n");
	j = 0; // counter for chosenNodes[]
	for (i = 0; i < tableCount; ++i) {
		if (nodeStats[i].steps != 0) continue; // non-chosen
		chosenNodes[j++] = i;
		printf("CS %s\n", table_name[i]);
	}

	// statistics
	for (i = 0; i < tableCount; ++i) {
		csCount += 1;
		sumSubjects += table_freq[i];
		if (nodeStats[i].steps == 0) {
			sumChosenSubjects += table_freq[i];
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	// free
	free(nodeStats);

	*numNodesActual = numNodes;
	return chosenNodes;
}

static
NodeStat* initNodeStats23(long int* table_freq, int tableCount) {
	NodeStat*	nodeStats = NULL;
	int		i;

	nodeStats = (NodeStat *) malloc(sizeof(NodeStat) * tableCount);
	if (!nodeStats) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < tableCount; ++i) {
		nodeStats[i].origWeight = table_freq[i];
		nodeStats[i].weight = 0;
		nodeStats[i].steps = -1; // not used
		nodeStats[i].predecessor = 0; // not used
	}

	return nodeStats;
}

static
void assignWeightToChildren2(long int* adjacency_from, long int* adjacency_to, int adjacencyCount, NodeStat* nodeStats, long int* table_id, int tableCount, int root) {
	int		i, j;

	// mark root as a "chosen node"
	nodeStats[root].steps = 0;
	nodeStats[root].weight = 0;

	// set summed weight for children
	for (i = 0; i < tableCount; ++i) {
		if (edgeExists(table_id[root], table_id[i], adjacency_from, adjacency_to, adjacencyCount)) {
			if (nodeStats[i].steps == 0) continue; // already in list
			nodeStats[i].weight = 0;
			for (j = 0; j < tableCount; ++j) {
				if (edgeExists(table_id[i], table_id[j], adjacency_from, adjacency_to, adjacencyCount)) {
					if (nodeStats[j].steps == 0) continue; // already in list
					nodeStats[i].weight += nodeStats[j].origWeight;
				}
			}
			nodeStats[i].weight += nodeStats[i].origWeight;
		}
	}
}

static
int* retrieval2(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	NodeStat	*nodeStats = NULL;
	int		numNodes;
	int		*chosenNodes = NULL;
	int		i, j;
	int		sumSubjects = 0;
	int		csCount = 0;
	int		sumChosenSubjects = 0;

	if (numNodesMax < 1) fprintf(stderr, "ERROR: numNodesMax < 1!\n");

	chosenNodes = (int *) malloc(sizeof(int) * numNodesMax);
	if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	nodeStats = initNodeStats23(table_freq, tableCount);
	numNodes = 1;

	// add root node
	numNodes = 1;
	assignWeightToChildren2(adjacency_from, adjacency_to, adjacencyCount, nodeStats, table_id, tableCount, root);

	// add nodes
	for (i = numNodes; i < numNodesMax; ++i) {
		// get top node (highest weight)
		int top = -1;
		for (j = 0; j < tableCount; ++j) {
			if (nodeStats[j].weight != 0) {
				if (top == -1) {
					top = j; // set first value
					continue;
				}
				if (nodeStats[j].weight > nodeStats[top].weight) {
					top = j;
				}
			}
		}
		if (top == -1) break; // not enough nodes found
		numNodes += 1;
		assignWeightToChildren2(adjacency_from, adjacency_to, adjacencyCount, nodeStats, table_id, tableCount, top);
	}

	// store list of chosen nodes
	printf("SUBSCHEMA:\n");
	j = 0; // counter for chosenNodes[]
	for (i = 0; i < tableCount; ++i) {
		if (nodeStats[i].steps != 0) continue; // non-chosen
		chosenNodes[j++] = i;
		printf("CS %s\n", table_name[i]);
	}

	// statistics
	for (i = 0; i < tableCount; ++i) {
		csCount += 1;
		sumSubjects += table_freq[i];
		if (nodeStats[i].steps == 0) {
			sumChosenSubjects += table_freq[i];
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	// free
	free(nodeStats);

	*numNodesActual = numNodes;
	return chosenNodes;
}

static
void assignWeightToChildren3(long int* adjacency_from, long int* adjacency_to, int adjacencyCount, NodeStat* nodeStats, long int* table_id, int tableCount, int root) {
	int		i, j, k;
	char		visited[tableCount];

	// mark root as a "chosen node"
	nodeStats[root].steps = 0;
	nodeStats[root].weight = 0;

	// set summed weight for children
	for (i = 0; i < tableCount; ++i) {
		if (edgeExists(table_id[root], table_id[i], adjacency_from, adjacency_to, adjacencyCount)) {
			if (nodeStats[i].steps == 0) continue; // already in list
			nodeStats[i].weight = 0;

			for (j = 0; j < tableCount; ++j) {
				visited[j] = 0;
			}
			visited[i] = 1;

			for (j = 0; j < tableCount; ++j) {
				if (edgeExists(table_id[i], table_id[j], adjacency_from, adjacency_to, adjacencyCount)) {
					if (nodeStats[j].steps == 0) continue; // already in list
					for (k = 0; k < tableCount; ++k) {
						if (edgeExists(table_id[j], table_id[k], adjacency_from, adjacency_to, adjacencyCount)) {
							if (nodeStats[k].steps == 0) continue; // already in list
							if (visited[k]) continue; // cycle or reachable over multiple pathes
							visited[k] = 1;
							nodeStats[i].weight += nodeStats[k].origWeight;
						}
					}
					if (!visited[j]) {
						// visited[j] means that this j-node is also an k-node for another j-node. In this case the j-node-origWeight must not be added
						nodeStats[i].weight += nodeStats[j].origWeight;
						visited[j] = 1;
					}
				}
			}
			nodeStats[i].weight += nodeStats[i].origWeight;
		}
	}
}

static
int* retrieval3(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	NodeStat	*nodeStats = NULL;
	int		numNodes;
	int		*chosenNodes = NULL;
	int		i, j;
	int		sumSubjects = 0;
	int		csCount = 0;
	int		sumChosenSubjects = 0;

	if (numNodesMax < 1) fprintf(stderr, "ERROR: numNodesMax < 1!\n");

	chosenNodes = (int *) malloc(sizeof(int) * numNodesMax);
	if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	nodeStats = initNodeStats23(table_freq, tableCount);
	numNodes = 1;

	// add root node
	numNodes = 1;
	assignWeightToChildren3(adjacency_from, adjacency_to, adjacencyCount, nodeStats, table_id, tableCount, root);

	// add nodes
	for (i = numNodes; i < numNodesMax; ++i) {
		// get top node (highest weight)
		int top = -1;
		for (j = 0; j < tableCount; ++j) {
			if (nodeStats[j].weight != 0) {
				if (top == -1) {
					top = j; // set first value
					continue;
				}
				if (nodeStats[j].weight > nodeStats[top].weight) {
					top = j;
				}
			}
		}
		if (top == -1) break; // not enough nodes found
		numNodes += 1;
		assignWeightToChildren3(adjacency_from, adjacency_to, adjacencyCount, nodeStats, table_id, tableCount, top);
	}

	// store list of chosen nodes
	printf("SUBSCHEMA:\n");
	j = 0; // counter for chosenNodes[]
	for (i = 0; i < tableCount; ++i) {
		if (nodeStats[i].steps != 0) continue; // non-chosen
		chosenNodes[j++] = i;
		printf("CS %s\n", table_name[i]);
	}

	// statistics
	for (i = 0; i < tableCount; ++i) {
		csCount += 1;
		sumSubjects += table_freq[i];
		if (nodeStats[i].steps == 0) {
			sumChosenSubjects += table_freq[i];
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	// free
	free(nodeStats);

	*numNodesActual = numNodes;
	return chosenNodes;
}

int* retrieval(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	if (SUBSCHEMA_HEURISTIC == 3) {
		return retrieval3(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
	} else if (SUBSCHEMA_HEURISTIC == 2) {
		return retrieval2(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
	}
	// SUBSCHEMA_HEURISTIC == 1 or other value
	return retrieval1(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
};
