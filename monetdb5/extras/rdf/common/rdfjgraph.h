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

#ifndef _RDFJOINGRAPH_H_
#define _RDFJOINGRAPH_H_

#include <gdk.h>
#include <sql_relation.h>
#include <rdfstack.h>

#ifdef WIN32
#ifndef LIBRDF
#define rdf_export extern __declspec(dllimport)
#else
#define rdf_export extern __declspec(dllexport)
#endif
#else
#define rdf_export extern
#endif

#define MAX_JGRAPH_NODENUMBER	20	//The join graph should have less than 20 nodes
#define UNDIRECTED_JGRAPH	1	//If the graph is undirected graph

typedef enum JoinPredicate {			//Join predicate
	JP_NAV,				//Not availabel
	JP_S, 				//Join on S
	JP_P, 					
	JP_O, 
	JP_SO,				//Join on both S and O
	JP_CP				//Cross Product
} JP; 

typedef enum JNodeType {
	JN_REQUIRED, 
	JN_OPTIONAL			
} JNodeT; 			

typedef struct jgedge {
	int from; 	
	int to; 
	operator_type op;		
	struct jgedge *next; 
	void *data; 
	JP jp; 
	int gl_edge_id; 
	int r_id; 	//rel id. For keeping the information of 
			//join where this edge was created
	int p_r_id; 	//parent child join id
	int is_processed; 	//Keep state of the edge --> edge is already processed 
				//and transformed 
	int need_add_exps; //All exps from rel->exps of this edge need to be added
			//into the join predicate between patterns

} jgedge; 

typedef struct jgnode {
	int vid; 			//vertex id
	int nEdge; 			//number of edges
	jgedge *first; 			//First edge
	jgedge *last; 			//Last edge
	int sccIdx;			//Index of strongly connected component this vertex
					//belongs to
	int lowlink; 			//For detecting scc
	void *data; 			//Data stored in each node. 
					//Currently, it should be sql_rel of op_select or op_basetable
	int subjgId;
	int patternId; 			//For star pattern
	int ijpatternId; 		//Id of inner join subgraph for nodes connected 
					//only by inner join in the star patter
	oid soid; 			//Oid of the subject if it is already known
	oid poid; 			//Oid of the predicate if it is already known (usually Yes)
	str prop; 			//Store the original name of the prop
	JNodeT type;			//This node can be optional, or required in the pattern
} jgnode; 

typedef struct jgraph{
	int nNode;	 	//Number of vertex
	int nEdge; 
	jgnode **lstnodes; 
	int nAllocation; 	//Number of allocation. 
} jgraph; 

rdf_export
jgraph *initJGraph(void);		//Init join graph

rdf_export
void freeJGraph(jgraph *jg);		//Free join graph

rdf_export
void addJGnode(int *vid, jgraph *jg, void *data, int subjgId, oid soid, oid poid, char *prop, JNodeT type); 

rdf_export 
void add_undirectedJGedge(int from, int to, operator_type op, jgraph *jg, void *data, JP jp, int rel_id, int p_rel_id, int need_add_exps);

rdf_export
void add_directedJGedge(int from, int to, operator_type op, jgraph *jg, void *data, JP jp, int rel_id, int p_rel_id, int need_add_exps);

rdf_export 
void update_undirectededge_jp(jgraph *jg, int from, int to, JP jp);

rdf_export
void setNodeType(jgnode *node, JNodeT type); 

rdf_export
jgedge* get_edge_jp(jgraph *jg, int from, int to);

rdf_export
void detectSCC(jgraph *jg);

rdf_export
void printJGraph(jgraph *jg); 

rdf_export
void buildExampleJGraph(void); 


#endif /* _RDFJOINGRAPH_H_ */
