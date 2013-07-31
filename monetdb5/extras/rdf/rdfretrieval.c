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
int getTableIndex(long int id, long int* table_id, int tableCount) {
	int i;
	for (i = 0; i < tableCount; ++i) {
		if (table_id[i] == id) return i;
	}
	return -1;
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

static
int* retrieval4(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, long int* adjacency_freq, int adjacencyCount) {
	int		numNodes;
	int		*chosenNodes = NULL;
	int		i, j;
	int		sumSubjects = 0;
	int		csCount = 0;
	int		sumChosenSubjects = 0;

	if (numNodesMax < 1) fprintf(stderr, "ERROR: numNodesMax < 1!\n");

	chosenNodes = (int *) malloc(sizeof(int) * numNodesMax);
	if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	numNodes = 0;

	// add root node
	chosenNodes[numNodes] = root;
	numNodes += 1;

	// add nodes
	while (numNodes < numNodesMax) {
		int bestNextEdge = -1;
		for (i = 0; i < adjacencyCount; ++i) {
			char foundFrom = 0;
			char foundTo = 0;
			for (j = 0; j < numNodes; ++j) {
				if (chosenNodes[j] == getTableIndex(adjacency_to[i], table_id, tableCount)) {
					foundTo = 1;
					break;
				}
			}
			for (j = 0; j < numNodes; ++j) {
				if (chosenNodes[j] == getTableIndex(adjacency_from[i], table_id, tableCount)) {
					foundFrom = 1;
					break;
				}
			}
			if (foundFrom && !foundTo) {
				// set or update
				if (bestNextEdge == -1) {
					// first edge
					bestNextEdge = i;
				} else {
					if (adjacency_freq[i] > adjacency_freq[bestNextEdge]) bestNextEdge = i;
				}
			}
		}
		if (bestNextEdge == -1) {
			// no more edges
			break;
		} else {
			chosenNodes[numNodes] = getTableIndex(adjacency_to[bestNextEdge], table_id, tableCount);
			numNodes += 1;
		}
	}

	printf("SUBSCHEMA:\n");
	for (i = 0; i < numNodes; ++i) {
		str name = table_name[chosenNodes[i]];
		printf("CS %s\n", name);
	}

	// statistics
	for (i = 0; i < tableCount; ++i) {
		csCount += 1;
		sumSubjects += table_freq[i];
		for (j = 0; j < numNodes; ++j) {
			if (chosenNodes[j] == i) sumChosenSubjects += table_freq[i];
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	*numNodesActual = numNodes;
	return chosenNodes;
}

static
char** initEdgesOverview(long int* table_id, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	char		**edges;
	int		i, j;

	edges = (char **) malloc(sizeof(char *) * tableCount);
	if (!edges) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < tableCount; ++i) {
		edges[i] = (char *) malloc(sizeof(char) * tableCount);
		if (!edges[i]) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");
		for (j = 0; j < tableCount; ++j) {
			edges[i][j] = 0;
		}
		edges[i][i] = 1; // self-reachability
	}

	for (i = 0; i < adjacencyCount; ++i) {
		long int from = adjacency_from[i];
		long int to = adjacency_to[i];
		int fromIdx = -1;
		int toIdx = -1;

		// index lookup
		for (j = 0; j < tableCount; ++j) {
			if (table_id[j] == from) {fromIdx = j;}
			if (table_id[j] == to) {toIdx = j;}
			if (fromIdx > -1 && toIdx > -1) {break;}
		}
		assert(fromIdx > -1);
		assert(toIdx > -1);

		// set edge
		edges[fromIdx][toIdx] = 1;
	}

	return edges;
}

static
int compareOverviewNodes (const void * a, const void * b) {
  return ( (*(Node*)b).reachabilityCount - (*(Node*)a).reachabilityCount ); // sort descending
}

static
int* retrievalOverview(int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, int adjacencyCount) {
	int		i, j, k;
	char		**edges;
	int		sumSubjects = 0;
	int		csCount = 0;
	int		sumChosenSubjects = 0;

	int		queue[tableCount]; // cyclic array
	int		isInQueue[tableCount];
	int		queuePosition; // next element in queue to view at
	int		queueLength;
	char		visited[tableCount];
	int		subgraphSize;
	Groups		groups;
	int		*chosenNodes = NULL;

	groups.count = 0;
	groups.groups = NULL;

	edges = initEdgesOverview(table_id, tableCount, adjacency_from, adjacency_to, adjacencyCount);

	for (i = 0; i < tableCount; ++i) {
		visited[i] = 0;
	}

	// split into disconnected subgraph (ignoring the direction of the edges) using BFS
	while (1) {
		int root = -1;
		for (i = 0; i < tableCount; ++i) {
			if (!visited[i]) {
				root = i;
				break;
			}
		}
		if (root == -1) break; // all nodes have been visited, all subgraphs have been found
		// init
		subgraphSize = 0;

		for (i = 0; i < tableCount; ++i) {
			queue[i] = -1;
			isInQueue[i] = 0;
		}

		// add root node
		queue[0] = root;
		queuePosition = 0;
		queueLength = 1;

		visited[root] = 1;
		isInQueue[root] = 1;

		// bfs
		while (queueLength > 0) {
			// dequeue next value
			int node = queue[queuePosition % tableCount];
			visited[node] = 1;
			subgraphSize++;
			isInQueue[node] = 0;
			queuePosition += 1;
			queueLength -= 1;

			// for all adjacent edges
			for (i = 0; i < tableCount; ++i) {
				if (visited[i] || isInQueue[i]) continue;
				if (edges[node][i] || edges[i][node]) {
					// ignore direction of edge

					// enqueue
					queue[((queueLength + queuePosition) % tableCount)] = i;
					queueLength += 1;
					isInQueue[i] = 1;
				}
			}
		}

		// store subgraph/group
		j = 0;
		// add group
		groups.count++;
		groups.groups = realloc(groups.groups, sizeof(Group) * groups.count);
		if (!groups.groups) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
		groups.groups[groups.count - 1].size = subgraphSize;
		groups.groups[groups.count - 1].nodes = (Node *) malloc(sizeof(Node) * subgraphSize);

		for (i = 0; i < tableCount; ++i) {
			if (visited[i] == 1) {
				// add to group
				groups.groups[groups.count - 1].nodes[j].idx = i;
				j++;
				visited[i] = 2; // node is still marked, but can be distinguished from nodes visited in other iterations
			}
		}
		assert(j == subgraphSize);
	}

	// transitive closure (Floyd-Warshall-Algorithm)
	for (k = 0; k < tableCount; ++k) {
		for (i = 0; i < tableCount; ++i) {
			for (j = 0; j < tableCount; ++j) {
				if (i == j || i == k || j == k) continue;
				if (edges[i][k] && edges[k][j]) edges[i][j] = 1;
			}
		}
	}

	// select nodes to be shown in the overview schema
	for (i = 0; i < groups.count; ++i) {
		int found = 0;

		// count how many group members can be reached from each node
		for (j = 0; j < groups.groups[i].size; ++j) {
			int node = groups.groups[i].nodes[j].idx;
			int reachabilityCount = 0;
			for (k = 0; k < tableCount; ++k) {
				if (edges[node][k]) {
					reachabilityCount++;
				}
			}
			groups.groups[i].nodes[j].reachabilityCount = reachabilityCount;

			if (reachabilityCount == groups.groups[i].size) {
				// found a node that links to all other nodes in that group --> add to overview schema
				(*numNodesActual) += 1;
				chosenNodes = realloc(chosenNodes, sizeof(int) * (*numNodesActual));
				if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				chosenNodes[*numNodesActual - 1] = node;
				found = 1;
				break;
			}
		}
		if (!found) {
			int node;
			char reachability[tableCount];
			int reachabilityCount = 0;
			int nextNode; // position in the (sorted) list of nodes to look at next

			// greedy
			qsort(groups.groups[i].nodes, groups.groups[i].size, sizeof(Node), compareOverviewNodes);

			// take first node (covers the most nodes)
			node = groups.groups[i].nodes[0].idx;
			(*numNodesActual) += 1;
			chosenNodes = realloc(chosenNodes, sizeof(int) * (*numNodesActual));
			if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
			chosenNodes[*numNodesActual - 1] = node;
			nextNode = 1;
			// store reachability vector
			for (j = 0; j < tableCount; ++j) {
				if (edges[node][j]) {
					reachability[j] = 1;
					reachabilityCount++;
				} else {
					reachability[j] = 0;
				}
			}
			assert (groups.groups[i].nodes[0].reachabilityCount == reachabilityCount);

			// take more nodes
			for (j = nextNode; j < tableCount; ++j) {
				int node = groups.groups[i].nodes[j].idx;;
				if (reachabilityCount == groups.groups[i].size) break;
				if (reachability[node]) continue;

				// take this node
				(*numNodesActual) += 1;
				chosenNodes = realloc(chosenNodes, sizeof(int) * (*numNodesActual));
				if (!chosenNodes) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");
				chosenNodes[*numNodesActual - 1] = node;
				nextNode = (j + 1);

				// update reachability vector
				for (k = 0; k < tableCount; ++k) {
					if (edges[node][k] && !reachability[k]) {
						reachability[k] = 1;
						reachabilityCount++;
					}
				}
			}
		}
	}

	printf("SUBSCHEMA:\n");
	for (i = 0; i < *numNodesActual; ++i) {
		str name = table_name[chosenNodes[i]];
		printf("CS %s\n", name);
	}

	// statistics
	for (i = 0; i < tableCount; ++i) {
		csCount += 1;
		sumSubjects += table_freq[i];
		for (j = 0; j < *numNodesActual; ++j) {
			if (chosenNodes[j] == i) sumChosenSubjects += table_freq[i];
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, *numNodesActual, csCount, (100.00 * *numNodesActual) / csCount);
	return chosenNodes;
}

int* retrieval(int root, int numNodesMax, int* numNodesActual, long int* table_id, str* table_name, long int* table_freq, int tableCount, long int* adjacency_from, long int* adjacency_to, long int* adjacency_freq, int adjacencyCount) {
	if (SUBSCHEMA_HEURISTIC == 5) {
		(void) numNodesMax;
		(void) root;
		return retrievalOverview(numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
	} else if (SUBSCHEMA_HEURISTIC == 4) {
		return retrieval4(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacency_freq, adjacencyCount);
	} else if (SUBSCHEMA_HEURISTIC == 3) {
		(void) adjacency_freq;
		return retrieval3(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
	} else if (SUBSCHEMA_HEURISTIC == 2) {
		(void) adjacency_freq;
		return retrieval2(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
	}
	// SUBSCHEMA_HEURISTIC == 1 or other value
	(void) adjacency_freq;
	return retrieval1(root, numNodesMax, numNodesActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacencyCount);
};
