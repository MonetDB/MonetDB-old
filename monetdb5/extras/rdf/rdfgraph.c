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

/* This contains graph algorithms for the graph formed by CS's and their relationships */

#include "monetdb_config.h"
#include "mal_exception.h"
#include "url.h"
#include "tokenizer.h"
#include <gdk.h>
#include <rdf.h>
#include <rdfschema.h>
#include <rdfgraph.h>



static
void visit(int curV, CSrel *csrelSet, char *visited, int* d, int level, int* maxd, int* maxV, char* globalVisited){
	int j; 
	int nextV; 

	visited[curV] = 1; 
	globalVisited[curV] = 1; 
	d[curV] = level; 
	
	//Update and store the value with max distance
	if (d[curV] > *maxd){
		*maxd = d[curV];
		*maxV = curV; 
	}
	
	//We use this condition for preventing the time consumption in
	//finding the graph diameter. Because, even though the graph 
	//diameter may be large, we will not run more than 10 iterations
	//for detecting the dimension table.
	//
	if (*maxd > MAX_ITERATION_NO){		
		return; 
	}

	if (csrelSet[curV].numRef != 0){
		for  (j = 0; j < csrelSet[curV].numRef; j++){
			nextV = csrelSet[curV].lstRefFreqIdx[j];

			if (!visited[nextV]) visit(nextV, csrelSet, visited, d, level + 1, maxd, maxV, globalVisited); 
		}
	}
}

static
void bfs(int numV, int start, CSrel *csrelSet, int* maxd, int* maxV, char* globalVisited){
	
	char* visited; 
	int i;
	int* d; 	//distances

	visited = (char*)GDKmalloc(sizeof(char) * numV); 
	d = (int*)GDKmalloc(sizeof(int) * numV); 
	
	for (i = 0; i < numV; i++){
		visited[i] = 0;
	}
	
	*maxd = -1; 

	visit(start,csrelSet,visited, d, 0, maxd, maxV, globalVisited); 

	GDKfree(visited); 
	GDKfree(d); 
}


//This function is implemented according to http://link.springer.com/chapter/10.1007%2F11764298_9
//k: Number of vertices
//
int getDiameter(int k, int numV,CSrel *csrelSet){
	int i; 
	int maxd = -1;
	int maxV = -1;
	int startV = -1;

	int unCheckV = 0; 
	int nConnectGraph = 0;

	char* globalVisited = NULL;	//To handle the case when the graph is not connected
	int globalMaxd = 0; 

	//init
	globalVisited = (char*) GDKmalloc(sizeof(char) * numV); 
	for (i = 0; i < numV; i++){
		globalVisited[i] = 0;
	}
	
	unCheckV = 0;
	//Go through each connected graph
	for (unCheckV = 0; unCheckV < numV; unCheckV++){
		if (globalVisited[unCheckV] || csrelSet[unCheckV].numRef == 0) continue;
		
		//For each connected graph, run k times	
		nConnectGraph++;
		printf("Start with connected graph %d from node %d\n", nConnectGraph,unCheckV);

		for (i = 0; i < k; i++){
			if (i == 0) startV = unCheckV;
			else{
				startV = maxV; 	
			}
			
			bfs(numV,startV, csrelSet, &maxd, &maxV, globalVisited);
			//printf("Max distance after the %d bfs is %d (from %d to %d)\n ", i, maxd, startV, maxV);
			if (maxd > globalMaxd) globalMaxd = maxd;
		}

		if (globalMaxd > MAX_ITERATION_NO)
			break; 
	}		


	GDKfree(globalVisited); 

	printf("Diameter is %d\n",globalMaxd);
	return globalMaxd;	
}

int getDiameterExact(int numV,CSrel *csrelSet){
	int i; 
	int maxd = -1;
	int maxV = -1;
	int startV = -1;

	int unCheckV = 0; 

	char* globalVisited = NULL;	//To handle the case when the graph is not connected
	int globalMaxd = 0; 

	//init
	globalVisited = (char*) GDKmalloc(sizeof(char) * numV); 
	for (i = 0; i < numV; i++){
		globalVisited[i] = 0;
	}
	
	//Go through each connected graph
	for (unCheckV = 0; unCheckV < numV; unCheckV++){
		if (csrelSet[unCheckV].numRef == 0) continue; 
		if (unCheckV % 1000 == 0) printf("Updated globalMaxd = %d\n",globalMaxd);
		startV = unCheckV;
		bfs(numV,startV, csrelSet, &maxd, &maxV, globalVisited);
		//printf("Max distance after the %d bfs is %d (from %d to %d)\n ", i, maxd, startV, maxV);
		if (maxd > globalMaxd){ 
			globalMaxd = maxd;
			//printf("New max %d from startV %d\n",globalMaxd,startV);
		}

		if (globalMaxd > MAX_ITERATION_NO) break; 
	}		


	GDKfree(globalVisited); 

	printf("Diameter is %d\n",globalMaxd);
	return globalMaxd;	
}
