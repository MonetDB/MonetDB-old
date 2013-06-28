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
char** initAdjacencyMatrix(int csCount) {
	char	**matrix = NULL; // matrix[from][to]
	int	i, j;

	matrix = (char **) malloc(sizeof(char *) * csCount);
	if (!matrix) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < csCount; ++i) {
		matrix[i] = (char *) malloc(sizeof(char *) * csCount);
		if (!matrix) fprintf(stderr, "ERROR: Couldn't realloc memory!\n");

		for (j = 0; j < csCount; ++j) {
			matrix[i][j] = 0;
		}
	}

	return matrix;
}

static
void createAdjacencyMatrix(char** matrix, CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet) {
	int	i, j, k;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore

		for (j = 0; j < freqCSset->items[i].numProp; ++j) { // propNo in CS order
			// check foreign key frequency
			int sum = 0;
			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) {
				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == freqCSset->items[i].lstProp[j]) {
					sum += csRelBetweenMergeFreqSet[i].lstCnt[k];
				}
			}

			for (k = 0; k < csRelBetweenMergeFreqSet[i].numRef; ++k) { // propNo in CSrel
				if (csRelBetweenMergeFreqSet[i].lstPropId[k] == freqCSset->items[i].lstProp[j]) {
					int to = csRelBetweenMergeFreqSet[i].lstRefFreqIdx[k];
					if (i == to) continue; // ignore self references
					if ((int) (100.0 * csRelBetweenMergeFreqSet[i].lstCnt[k] / sum + 0.5) < FK_FREQ_THRESHOLD) continue; // foreign key is not frequent enough
					matrix[i][to] = 1;
				}
			}	
		}
	}
}

static
NodeStat* initNodeStats(CSset* freqCSset) {
	NodeStat*	nodeStats = NULL;
	int		i;

	nodeStats = (NodeStat *) malloc(sizeof(NodeStat) * freqCSset->numCSadded);
	if (!nodeStats) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		nodeStats[i].origWeight = freqCSset->items[i].support;
		nodeStats[i].weight = freqCSset->items[i].support; // weight = origWeight
		nodeStats[i].steps = -1;
		nodeStats[i].predecessor = -1;
	}

	return nodeStats;
}

static
NodeStat* initNodeStats23(CSset* freqCSset) {
	NodeStat*	nodeStats = NULL;
	int		i;

	nodeStats = (NodeStat *) malloc(sizeof(NodeStat) * freqCSset->numCSadded);
	if (!nodeStats) fprintf(stderr, "ERROR: Couldn't malloc memory!\n");

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		nodeStats[i].origWeight = freqCSset->items[i].support;
		nodeStats[i].weight = 0;
		nodeStats[i].steps = -1; // not used
		nodeStats[i].predecessor = 0; // not used
	}

	return nodeStats;
}

static
void bfs1(int root, CSset* freqCSset, char** adjacencyMatrix, int* queue, int* visited, int* isInQueue, int* queuePosition, int* queueLength, NodeStat* nodeStats) {
	int	i;

	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (adjacencyMatrix[root][i]) {
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
				queue[((*queueLength + *queuePosition) % freqCSset->numCSadded)] = i;
				*queueLength += 1;
				isInQueue[i] = 1;
			}

		}
	}

	if (*queueLength > 0) {
		visited[queue[(*queuePosition % freqCSset->numCSadded)]] = 1;
		isInQueue[queue[(*queuePosition % freqCSset->numCSadded)]] = 0;
		*queuePosition += 1;
		*queueLength -= 1;
		bfs1(queue[((*queuePosition + freqCSset->numCSadded - 1) % freqCSset->numCSadded)], freqCSset, adjacencyMatrix, queue, visited, isInQueue, queuePosition, queueLength, nodeStats);
	}
}

static
void addNode1(char** adjacencyMatrix, NodeStat* nodeStats, CSset* freqCSset, int root, char initial) {
	int	queue[freqCSset->numCSadded]; // cyclic array
	int	visited[freqCSset->numCSadded];
	int	isInQueue[freqCSset->numCSadded];
	int	queuePosition; // next element in queue to view at
	int	queueLength;
	int	pathId, pathIdTmp;
	int	i;

	// init
	for (i = 0; i < freqCSset->numCSadded; ++i) {
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

	bfs1(root, freqCSset, adjacencyMatrix, queue, visited, isInQueue, &queuePosition, &queueLength, nodeStats);
}

int* retrieval1(int root, int numNodesMax, int* numNodesActual, CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet) {
	char		**adjacencyMatrix = NULL;
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

	adjacencyMatrix = initAdjacencyMatrix(freqCSset->numCSadded);
	createAdjacencyMatrix(adjacencyMatrix, freqCSset, csRelBetweenMergeFreqSet);
	nodeStats = initNodeStats(freqCSset);
	numNodes = 1;

	// add root node
	addNode1(adjacencyMatrix, nodeStats, freqCSset, root, 1);

	// add nodes
	while (numNodes < numNodesMax) {
		// get top node (highest fraction (weight/steps))
		int top = -1;
		for (i = 0; i < freqCSset->numCSadded; ++i) {
			int topWeight, testWeight;
			if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
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
		addNode1(adjacencyMatrix, nodeStats, freqCSset, top, 0); // add node(s)
	}

	// store list of chosen nodes
	printf("SUBSCHEMA:\n");
	j = 0; // counter for chosenNodes[]
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		if (nodeStats[i].steps != 0) continue; // non-chosen
		chosenNodes[j++] = i;
		printf("CS "BUNFMT"\n", freqCSset->items[i].csId);
	}

	// statistics
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		csCount += 1;
		sumSubjects += freqCSset->items[i].support;
		if (nodeStats[i].steps == 0) {
			sumChosenSubjects += freqCSset->items[i].support;
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	// free
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		free(adjacencyMatrix[i]);
	}
	free(adjacencyMatrix);
	free(nodeStats);

	*numNodesActual = numNodes;
	return chosenNodes;
}

static
void assignWeightToChildren2(char** adjacencyMatrix, NodeStat* nodeStats, CSset* freqCSset, int root) {
	int		i, j;

	// mark root as a "chosen node"
	nodeStats[root].steps = 0;
	nodeStats[root].weight = 0;

	// set summed weight for children
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (adjacencyMatrix[root][i]) {
			if (nodeStats[i].steps == 0) continue; // already in list
			nodeStats[i].weight = 0;
			for (j = 0; j < freqCSset->numCSadded; ++j) {
				if (adjacencyMatrix[i][j]) {
					if (nodeStats[j].steps == 0) continue; // already in list
					nodeStats[i].weight += nodeStats[j].origWeight;
				}
			}
			nodeStats[i].weight += nodeStats[i].origWeight;
		}
	}
}

int* retrieval2(int root, int numNodesMax, int* numNodesActual, CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet) {
	char		**adjacencyMatrix = NULL;
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

	adjacencyMatrix = initAdjacencyMatrix(freqCSset->numCSadded);
	createAdjacencyMatrix(adjacencyMatrix, freqCSset, csRelBetweenMergeFreqSet);
	nodeStats = initNodeStats23(freqCSset);
	numNodes = 1;

	// add root node
	numNodes = 1;
	assignWeightToChildren2(adjacencyMatrix, nodeStats, freqCSset, root);

	// add nodes
	for (i = numNodes; i < numNodesMax; ++i) {
		// get top node (highest weight)
		int top = -1;
		for (j = 0; j < freqCSset->numCSadded; ++j) {
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
		assignWeightToChildren2(adjacencyMatrix, nodeStats, freqCSset, top);
	}

	// store list of chosen nodes
	printf("SUBSCHEMA:\n");
	j = 0; // counter for chosenNodes[]
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		if (nodeStats[i].steps != 0) continue; // non-chosen
		chosenNodes[j++] = i;
		printf("CS "BUNFMT"\n", freqCSset->items[i].csId);
	}

	// statistics
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		csCount += 1;
		sumSubjects += freqCSset->items[i].support;
		if (nodeStats[i].steps == 0) {
			sumChosenSubjects += freqCSset->items[i].support;
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	// free
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		free(adjacencyMatrix[i]);
	}
	free(adjacencyMatrix);
	free(nodeStats);

	*numNodesActual = numNodes;
	return chosenNodes;
}

static
void assignWeightToChildren3(char** adjacencyMatrix, NodeStat* nodeStats, CSset* freqCSset, int root) {
	int		i, j, k;
	char		visited[freqCSset->numCSadded];

	// mark root as a "chosen node"
	nodeStats[root].steps = 0;
	nodeStats[root].weight = 0;

	// set summed weight for children
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (adjacencyMatrix[root][i]) {
			if (nodeStats[i].steps == 0) continue; // already in list
			nodeStats[i].weight = 0;

			for (j = 0; j < freqCSset->numCSadded; ++j) {
				visited[j] = 0;
			}
			visited[i] = 1;

			for (j = 0; j < freqCSset->numCSadded; ++j) {
				if (adjacencyMatrix[i][j]) {
					if (nodeStats[j].steps == 0) continue; // already in list
					for (k = 0; k < freqCSset->numCSadded; ++k) {
						if (adjacencyMatrix[j][k]) {
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

int* retrieval3(int root, int numNodesMax, int* numNodesActual, CSset* freqCSset, CSmergeRel* csRelBetweenMergeFreqSet) {
	char		**adjacencyMatrix = NULL;
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

	adjacencyMatrix = initAdjacencyMatrix(freqCSset->numCSadded);
	createAdjacencyMatrix(adjacencyMatrix, freqCSset, csRelBetweenMergeFreqSet);
	nodeStats = initNodeStats23(freqCSset);
	numNodes = 1;

	// add root node
	numNodes = 1;
	assignWeightToChildren3(adjacencyMatrix, nodeStats, freqCSset, root);

	// add nodes
	for (i = numNodes; i < numNodesMax; ++i) {
		// get top node (highest weight)
		int top = -1;
		for (j = 0; j < freqCSset->numCSadded; ++j) {
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
		assignWeightToChildren3(adjacencyMatrix, nodeStats, freqCSset, top);
	}

	// store list of chosen nodes
	printf("SUBSCHEMA:\n");
	j = 0; // counter for chosenNodes[]
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		if (nodeStats[i].steps != 0) continue; // non-chosen
		chosenNodes[j++] = i;
		printf("CS "BUNFMT"\n", freqCSset->items[i].csId);
	}

	// statistics
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		if (freqCSset->items[i].parentFreqIdx != -1) continue; // ignore
		csCount += 1;
		sumSubjects += freqCSset->items[i].support;
		if (nodeStats[i].steps == 0) {
			sumChosenSubjects += freqCSset->items[i].support;
		}
	}
	printf("COVERAGE:\n");
	printf("%d out of %d (%f %%) using %d out of %d tables (%f %%)\n", sumChosenSubjects, sumSubjects, (100.00 * sumChosenSubjects) / sumSubjects, numNodes, csCount, (100.00 * numNodes) / csCount);

	// free
	for (i = 0; i < freqCSset->numCSadded; ++i) {
		free(adjacencyMatrix[i]);
	}
	free(adjacencyMatrix);
	free(nodeStats);

	*numNodesActual = numNodes;
	return chosenNodes;
}
