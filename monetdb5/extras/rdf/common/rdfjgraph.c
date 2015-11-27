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
#include "mal_exception.h"
#include <gdk.h>
#include <rdfjgraph.h>

static 
void freeJGnode(jgnode *node){
	jgedge *nxtedge;
	jgedge *tmpedge; 

	nxtedge = node->first; 
	while (nxtedge != NULL){
		tmpedge = nxtedge;	
		nxtedge = nxtedge->next; 	
		free(tmpedge); 
	}

	/* The data is sql_rel
	 *
	if (node->data != NULL) 
		free(node->data); 
	*/
	free(node); 
}
jgraph* initJGraph(void){
	jgraph *jg = NULL; 
	jg = (jgraph*)malloc(sizeof(jgraph)); 
	jg->nNode = 0; 
	jg->nEdge = 0; 
	jg->nAllocation = MAX_JGRAPH_NODENUMBER;
	jg->lstnodes = (jgnode**) malloc(sizeof (jgnode*) * MAX_JGRAPH_NODENUMBER);
	return jg; 
}

void addJGnode(int *vid, jgraph *jg, void *data, int subjgId, oid _soid, oid _poid, char *prop, JNodeT t){

	jgnode *node = (jgnode *) malloc(sizeof(jgnode)); 
	node->vid = jg->nNode; 
	node->nEdge = 0; 
	node->first = NULL; 	
	node->sccIdx = -1; 	//for SCC
	node->lowlink = -1; 	//for SCC
	node->data = data; 
	node->subjgId = subjgId; 
	node->patternId = -1; 
	node->ijpatternId = -1;
	node->soid = _soid; 
	node->poid = _poid; 
	if (prop) node->prop = GDKstrdup(prop);
	else  node->prop = NULL; 

	node->type = t;
	
	jg->lstnodes[jg->nNode] = node; 
	*vid = jg->nNode; 	//return node id; 
	jg->nNode++;
}

void setNodeType(jgnode *node, JNodeT t){
	node->type = t; 
}


static 
int existEdge(jgraph *jg, int from, int to){
	jgedge *nxtedge;
	jgedge *tmpedge; 
	int exist = 0; 
	
	jgnode *node = jg->lstnodes[from]; 

	nxtedge = node->first; 
	while (nxtedge != NULL){
		tmpedge = nxtedge;
		assert(tmpedge->from == node->vid); 
		if (tmpedge->to == to){
			exist = 1; 
			break; 
		}
		nxtedge = nxtedge->next; 	
	}

	return exist; 
}

/*
 * We DO NOT check for the duplication when adding edge
 * UPDATE: ADD checking for duplication
 * */
void add_directedJGedge(int from, int to, operator_type op, jgraph *jg, void *data, JP jp, int rel_id, int p_rel_id, int need_add_exps){
	jgnode *fromnode;
	jgedge *edge;
	
	if (existEdge(jg, from, to)){ 
		printf("The edge[%d,%d] is duplicated\n", from,to);
		return; 
	}

	edge = (jgedge *) malloc(sizeof(jgedge));
	edge->from = from; 
	edge->to = to; 
	edge->op = op; 
	edge->next = NULL; 
	edge->data = data; 	
	edge->jp = jp; 
	edge->gl_edge_id = jg->nEdge; 
	edge->r_id = rel_id; 
	edge->p_r_id = p_rel_id; 
	edge->is_processed = 0;
	edge->need_add_exps = need_add_exps;
	
	fromnode = jg->lstnodes[from]; 

	assert(fromnode->vid == edge->from); 
	if (fromnode->nEdge == 0){ //The first edge
		printf("First edge of node %d from %d to %d\n", fromnode->vid, edge->from, edge->to); 
		fromnode->first = edge;
		fromnode->last = edge;
	} 
	else{	
		fromnode->last->next = edge;  
		fromnode->last = edge; 
		printf("Add edge of node %d from %d to %d\n", fromnode->vid, edge->from, edge->to); 
	}
	fromnode->nEdge++;
	jg->nEdge++;
}

jgedge* get_edge_jp(jgraph *jg, int from, int to){
	jgedge *edge = NULL, *nxtedge = NULL, *tmpedge = NULL;
	jgnode *node = jg->lstnodes[from];
	
	nxtedge = node->first; 
	while (nxtedge != NULL){
		tmpedge = nxtedge;
		assert(tmpedge->from == node->vid); 
		if (tmpedge->to == to){
			edge = tmpedge;
			break;
		}
		nxtedge = nxtedge->next;
	}

	return edge; 
}

static 
void update_directededge_jp(jgraph *jg, int from, int to, JP jp){
	jgedge *nxtedge;
	jgedge *tmpedge; 
	
	jgnode *node = jg->lstnodes[from]; 

	nxtedge = node->first; 
	while (nxtedge != NULL){
		tmpedge = nxtedge;
		assert(tmpedge->from == node->vid); 
		if (tmpedge->to == to){
			JP oldjp;
			JP newjp;
			oldjp = tmpedge->jp; 
			newjp = oldjp; //if there is no update
			if ((oldjp == JP_O) && (jp == JP_S))
				newjp = JP_SO; 
			if ((oldjp == JP_S) && (jp == JP_O))	
				newjp = JP_SO; 
			

			tmpedge->jp = newjp; 
			break; 
		}
		nxtedge = nxtedge->next; 	
	}
}

void update_undirectededge_jp(jgraph *jg, int from, int to, JP jp){
	update_directededge_jp(jg, from, to, jp); 
	update_directededge_jp(jg, to, from, jp);
}

void add_undirectedJGedge(int from, int to, operator_type op, jgraph *jg, void *data, JP jp, int rel_id, int p_rel_id, int need_add_exps){

		add_directedJGedge(from, to, op, jg, data, jp, rel_id, p_rel_id, need_add_exps); 
		add_directedJGedge(to, from, op, jg, data, jp, rel_id, p_rel_id, need_add_exps); 	
		
}

void freeJGraph(jgraph *jg){
	int i; 
	for (i = 0; i < jg->nNode; i++){
		freeJGnode(jg->lstnodes[i]); 
	}
	free(jg->lstnodes); 
	free(jg); 
}

void printJGraph(jgraph *jg){
	int i; 
	jgnode *tmpnode; 
	jgedge *tmpedge; 
	printf("---- Join Graph -----\n"); 
	for (i = 0; i  < jg->nNode; i++){
		tmpnode = jg->lstnodes[i]; 
		printf("Node %d: ", i); 
		tmpedge = tmpnode->first; 
		while (tmpedge != NULL){
			assert(tmpedge->from == tmpnode->vid); 
			printf(" %d", tmpedge->to); 
			tmpedge = tmpedge->next; 
		}
		printf("\n"); 
	}
	printf("---------------------\n"); 
}

void buildExampleJGraph(void){
	int i, tmpid; 
	jgraph *jg = initJGraph(); 

	for (i = 0; i < 5; i++){
		addJGnode(&tmpid, jg, NULL, 0, BUN_NONE, BUN_NONE, NULL,  JN_REQUIRED); 
	}
	add_undirectedJGedge(0, 1, op_join, jg, NULL, JP_NAV, -1, -1, 0); 
	add_undirectedJGedge(0, 3, op_join, jg, NULL, JP_NAV, -1, -1, 0); 
	add_undirectedJGedge(0, 4, op_join, jg, NULL, JP_NAV, -1, -1, 0); 

	add_undirectedJGedge(1, 3, op_join, jg, NULL, JP_NAV, -1, -1, 0); 
	add_undirectedJGedge(1, 2, op_join, jg, NULL, JP_NAV, -1, -1, 0); 

	add_undirectedJGedge(2, 3, op_join, jg, NULL, JP_NAV, -1, -1, 0); 
	add_undirectedJGedge(2, 4, op_join, jg, NULL, JP_NAV, -1, -1, 0); 
	printJGraph(jg); 
	freeJGraph(jg); 
}

static
void strongconnect(jgnode *v, int *curIdx, stackT *s, jgraph *jg){

	jgedge *tmpedge = NULL; 
	int 	tmpId; 
	v->sccIdx = *curIdx; 	
	v->lowlink = *curIdx;
	*curIdx = *curIdx + 1; 

	stPush(s, (stElementT) v->vid); 

	//Go through each edge
	tmpedge = v->first; 
	while (tmpedge != NULL){
		jgnode *toV = jg->lstnodes[tmpedge->to];

		if (toV->sccIdx == -1){
			strongconnect(toV, curIdx, s, jg); 
			v->lowlink = (v->lowlink > toV->lowlink)?toV->lowlink:v->lowlink; //min(v->lowlink, toV->lowlink)
		}
		else if (isInStack(s, toV->vid)){	//DUC: May be, put a attribute to each node to specify 
							// whether the node is in stack or not (update when push/pop)
			v->lowlink = (v->lowlink > toV->sccIdx)?toV->sccIdx:v->lowlink; //min(v->lowlink, toV->index)
		}
		tmpedge = tmpedge->next; 	
	}

	if (v->lowlink == v->sccIdx){
		do {
			tmpId = stPop(s); 		
		}
		while (tmpId != v->vid); 
	
	}

}
/*
 * Detect strongly connected components
 * Output: The sccIdx of each node is updated 
 * by the Idx of scc it belongs to
 * Implement according to Tarjan algorithm
 * http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
 * */
void detectSCC(jgraph *jg){
	int i; 
	int index = 0; 
	stackT *s; 
	
	s = initStack(MAX_JGRAPH_NODENUMBER);

	for (i = 0; i < jg->nNode; i++){
		if (jg->lstnodes[i]->sccIdx == -1){
			strongconnect(jg->lstnodes[i], &index, s, jg); 	
		}
	}
	
	stFree(s); 
}
