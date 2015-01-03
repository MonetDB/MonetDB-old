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
 * Copyright August 2008-2014 MonetDB B.V.
 * All Rights Reserved.
 */

#include "monetdb_config.h"
#include <sql_rdf_jgraph.h>
#include <rel_dump.h>
#include <rel_select.h>

/*
static
void addRelationsToJG(mvc *sql, sql_rel *rel, int depth, jgraph *jg){
	
	char *r = NULL;

	if (!rel)
		return;

	if (rel_is_ref(rel)) {
		(void); 
	}

	switch (rel->op) {
	case op_basetable: {
	} 	break;
	case op_table:
		break;
	case op_ddl:
		break;
	case op_join: 
	case op_left: 
	case op_right: 
	case op_full: 
	case op_apply: 
	case op_semi: 
	case op_anti: 
	case op_union: 
	case op_inter: 
	case op_except: 
		r = "join";
		if (rel->op == op_left)
			r = "left outer join";
		else if (rel->op == op_right)
			r = "right outer join";
		else if (rel->op == op_full)
			r = "full outer join";
		else if (rel->op == op_apply) {
			r = "apply";
			if (rel->flag == APPLY_JOIN)
				r = "apply join";
			else if (rel->flag == APPLY_LOJ)
				r = "apply left outer join";
			else if (rel->flag == APPLY_EXISTS)
				r = "apply exists";
			else if (rel->flag == APPLY_NOTEXISTS)
				r = "apply not exists";
		}
		else if (rel->op == op_semi)
			r = "semijoin";
		else if (rel->op == op_anti)
			r = "antijoin";
		else if (rel->op == op_union)
			r = "union";
		else if (rel->op == op_inter)
			r = "intersect";
		else if (rel->op == op_except)
			r = "except";
		else if (!rel->exps && rel->op == op_join)
			r = "crossproduct";
		print_indent(sql, fout, depth);
		if (need_distinct(rel))
			mnstr_printf(fout, "distinct ");
		mnstr_printf(fout, "%s (", r);
		if (rel_is_ref(rel->l)) {
			int nr = find_ref(refs, rel->l);
			print_indent(sql, fout, depth+1);
			mnstr_printf(fout, "& REF %d ", nr);
		} else
			rel_print_(sql, fout, rel->l, depth+1, refs);
		mnstr_printf(fout, ",");
		if (rel_is_ref(rel->r)) {
			int nr = find_ref(refs, rel->r);
			print_indent(sql, fout, depth+1);
			mnstr_printf(fout, "& REF %d  ", nr);
		} else
			rel_print_(sql, fout, rel->r, depth+1, refs);
		print_indent(sql, fout, depth);
		mnstr_printf(fout, ")");
		exps_print(sql, fout, rel->exps, depth, 1, 0);
		break;
	case op_project:
	case op_select: 
	case op_groupby: 
	case op_topn: 
	case op_sample: 
		r = "project";
		if (rel->op == op_select)
			r = "select";
		if (rel->op == op_groupby)
			r = "group by";
		if (rel->op == op_topn)
			r = "top N";
		if (rel->op == op_sample)
			r = "sample";
		print_indent(sql, fout, depth);
		if (rel->l) {
			if (need_distinct(rel))
				mnstr_printf(fout, "distinct ");
			mnstr_printf(fout, "%s (", r);
			if (rel_is_ref(rel->l)) {
				int nr = find_ref(refs, rel->l);
				print_indent(sql, fout, depth+1);
				mnstr_printf(fout, "& REF %d ", nr);
			} else
				rel_print_(sql, fout, rel->l, depth+1, refs);
			print_indent(sql, fout, depth);
			mnstr_printf(fout, ")");
		}
		if (rel->op == op_groupby)  // group by columns 
			exps_print(sql, fout, rel->r, depth, 1, 0);
		exps_print(sql, fout, rel->exps, depth, 1, 0);
		if (rel->r && rel->op == op_project) // order by columns 
			exps_print(sql, fout, rel->r, depth, 1, 0);
		break;
	default:
		assert(0);
	}
}
*/



static 
int isBasicPattern(sql_rel *r){
	if (r->op != op_select && r->op != op_basetable){	//r->op != op_table is removed
		return 0; 
	}

	if (r->op == op_select){
		assert (r->l);
		if (((sql_rel *)r->l)->op == op_basetable){
			return 1; 
		}
		return 0; 
	}

	return 1; 

}

static 
void exps_print_ext(mvc *sql, sql_rel *rel, int depth, char *prefix){
	
	size_t pos;
	size_t nl = 0;
	size_t len = 0, lastpos = 0;

	stream *fd = sql->scanner.ws;
	stream *s;
	buffer *b = buffer_create(2000); /* hopefully enough */
	if (!b)
		return; /* signal somehow? */
	s = buffer_wastream(b, "SQL Plan");
	if (!s) {
		buffer_destroy(b);
		return; /* signal somehow? */
	}

	exps_print(sql, s, rel->exps, depth, 1, 0);
	
	mnstr_printf(s, "\n");

	/* count the number of lines in the output, skip the leading \n */
	for (pos = 1; pos < b->pos; pos++) {
		if (b->buf[pos] == '\n') {
			nl++;
			if (len < pos - lastpos)
				len = pos - lastpos;
			lastpos = pos + 1;
		}
	}
	b->buf[b->pos - 1] = '\0';  /* should always end with a \n, can overwrite */

	mnstr_printf(fd, "%s \n", b->buf + 1 /* omit starting \n */);
	printf("%s %s\n", prefix, b->buf + 1 /* omit starting \n */);
	mnstr_close(s);
	mnstr_destroy(s);
	buffer_destroy(b);
}

static
void printRel_JGraph(jgraph *jg, mvc *sql){
	int i; 
	jgnode *tmpnode; 
	jgedge *tmpedge; 
	char tmp[50];
	printf("---- Join Graph -----\n"); 
	for (i = 0; i  < jg->nNode; i++){
		tmpnode = jg->lstnodes[i]; 
		//printf("Node %d: ", i); 
		sprintf(tmp, "Node %d: [Pattern: %d] ", i, tmpnode->patternId); 
		//mnstr_printf(sql->scanner.ws, "Node %d: ", i); 
		exps_print_ext(sql, (sql_rel *) tmpnode->data, 0, tmp); 
		tmpedge = tmpnode->first; 
		while (tmpedge != NULL){
			assert(tmpedge->from == tmpnode->vid); 
			printf(" %d", tmpedge->to); 
			tmpedge = tmpedge->next; 
		}
		printf("\n"); 
		//mnstr_printf(sql->scanner.ws, "\n"); 
	}
	printf("---------------------\n"); 
}

/*
 * Algorithm for adding sql rels to Join Graph
 *
 * 1. We consider joins that are directed connected to each other
 * are belonging to one subgraph of join. 
 * E.g., For join1->join2->project->join3->join4->join5, join1, join2
 * belong to one subgraph, join3,4,5 belong to one subgraph
 *
 * 2. We go from the top sql_rel
 *
 * */
static
void addRelationsToJG(mvc *c, sql_rel *rel, int depth, jgraph *jg, int new_subjg, int *subjgId){
	int tmpvid =-1; 

	switch (rel->op) {
		case op_left:
		case op_right:
		case op_join:
			if (rel->op == op_left || rel->op == op_right) printf("[Outter join]\n");
			else printf("[join]\n"); 

			printf("--- Between %s and %s ---\n", op2string(((sql_rel *)rel->l)->op), op2string(((sql_rel *)rel->r)->op) );		
			
			if (new_subjg){ 	//The new subgraph flag is set
				*subjgId = *subjgId + 1; 
			}

			addRelationsToJG(c, rel->l, depth+1, jg, 0, subjgId);
			addRelationsToJG(c, rel->r, depth+1, jg, 0, subjgId);

			break; 
		case op_select: 
			 printf("[select]\n");
			if (isBasicPattern(rel)){
				printf("Found a basic pattern\n");
				rel_print(c, rel, 0);
				//if (1) exps_print_ext(c, rel, 0); 
				addJGnode(&tmpvid, jg, (sql_rel *) rel, *subjgId); 
			}
			else{	//This is the connect to a new join sg
				addRelationsToJG(c, rel->l, depth+1, jg, 1, subjgId);
			}
			break; 
		case op_basetable:
			printf("[Base table]\n");		
			addJGnode(&tmpvid, jg, (sql_rel *) rel, *subjgId);
			rel_print(c, rel, 0);
			break;
		default:
			printf("[%s]\n", op2string(rel->op)); 
			if (rel->l) 
				addRelationsToJG(c, rel->l, depth+1, jg, 1, subjgId); 
			if (rel->r)
				addRelationsToJG(c, rel->r, depth+1, jg, 1, subjgId); 
			break; 
			
	}

}

/*
 * TODO: Should we use the function rel_name() from rel_select.c 
 * Consider removing this function
 * */
static
char *get_relname_from_basetable(sql_rel *rel){

	sql_exp *tmpexp;
	char *rname = NULL; 
	list *tmpexps; 

	assert(rel->op == op_basetable); 
	tmpexps = rel->exps;
	if (tmpexps){
		node *en; 
		
		rname = ((sql_exp *) tmpexps->h->data)->rname; 
		//For verifying that there is 
		//only one relation name
		for (en = tmpexps->h; en; en = en->next){
			tmpexp = (sql_exp *) en->data; 
			assert(tmpexp->type == e_column);
			printf("[Table] %s -> [Column] %s", tmpexp->rname, tmpexp->name);
			assert(strcmp(rname, tmpexp->rname) == 0); 
		}
	}
	
	printf("rname %s vs rname %s from rel_name fucntion", rname, rel_name(rel));
	return rname; 

}



/*
 * Get the name of the relation of each JG node
 * */
/*
static 
char *getNodeRel(jgnode *node){
	sql_rel *tmprel = (sql_rel *) node->data;
	assert(tmprel != NULL)
	switch (tmprel->op){
		case op_basetable: 

			break; 
		case op_select:
			break;
		default: 
			assert(0); 
	}

}
*/

static 
nMap *create_nMap(int maxnum){
	nMap *nm; 
	nm = (nMap*) malloc(sizeof(nMap));
	nm->lmap = BATnew(TYPE_void, TYPE_str, maxnum, TRANSIENT); 
	nm->rmap = BATnew(TYPE_void, TYPE_int, maxnum, TRANSIENT); 
	
	return nm; 
}
static
void free_nMap(nMap *nm){
	
	BBPunfix(nm->lmap->batCacheid); 
	BBPunfix(nm->rmap->batCacheid);
	free(nm); 
}
static 
void add_to_nMap(nMap *nm, str s, int *id){
	BUN bun; 

	bun = BUNfnd(nm->lmap,(ptr) (str)s);
	if (bun == BUN_NONE){
		BUNappend(nm->lmap, s, TRUE); 
		BUNappend(nm->rmap, id, TRUE); 
		printf("Add rname %s | %d to nmap\n", s, *id); 
	}
	else{
	
		printf("This should not happen\n");
		assert(0); 
	}
}

static 
int rname_to_nodeId(nMap *nm, str rname){
	int *id; 
	BUN bun; 
	bun = BUNfnd(nm->lmap,rname);

	if (bun == BUN_NONE){
		printf("Rel %s is not found in nmap \n", rname); 
		return -1; 
	}
	else{
		id = (int *) Tloc(nm->rmap, bun);  		
	}
	
	return *id; 
}


static 
void add_relNames_to_nmap(jgraph *jg, nMap *nm){
			
	jgnode *tmpnode; 
	//jgedge *tmpedge; 
	sql_rel *tmprel; 
	int i; 
	for (i = 0; i  < jg->nNode; i++){
		tmpnode = jg->lstnodes[i]; 
		tmprel = (sql_rel *) tmpnode->data; 
		
		if (tmprel->op == op_basetable){
			str s = get_relname_from_basetable(tmprel); 
			printf("[Node %d --> Table] %s\n", i, s);
			add_to_nMap(nm, s, &i); 

		}
		else if (tmprel->op == op_select){
			str s; 
			assert(((sql_rel *)tmprel->l)->op == op_basetable); //Only handle the case 
									    //when selecting from base_table	
			s = get_relname_from_basetable((sql_rel *)tmprel->l); 	
			printf("[Node %d -->Table from select] %s\n",i, s); 
			add_to_nMap(nm, s, &i);
		}
	}

	//Test the rname_to_id
	for (i = (jg->nNode - 1); i  >= 0; i--){
		tmpnode = jg->lstnodes[i];
		tmprel = (sql_rel *) tmpnode->data;
		if (tmprel->op == op_basetable){
			str s = get_relname_from_basetable(tmprel);
			printf("Get nodeId for %s from nmap: %d\n", s, rname_to_nodeId(nm, s)); 
		}
		else if (tmprel->op == op_select){
			str s = get_relname_from_basetable((sql_rel *)tmprel->l);
			printf("Get nodeId for %s from nmap: %d\n", s, rname_to_nodeId(nm, s));
		}

	}
}

static
void get_jp(str pred1, str pred2, JP *jp){
	
	if (strcmp(pred1, "s")==0 && strcmp(pred2, "s")==0){
			*jp = JP_S; 
	
	} else if (strcmp(pred1, "o")==0 && strcmp(pred2, "o")==0){
			*jp = JP_O; 
	} else
		*jp = JP_NAV; 
	
}

/*
 * Input: sql_rel with op == op_join, op_left or op_right
 * */
static
void _add_join_edges(jgraph *jg, sql_rel *rel, nMap *nm, char **isConnect){

	sql_exp *tmpexp;
	list *tmpexps; 

	assert(rel->op == op_join || rel->op == op_left || rel->op == op_right); 
	tmpexps = rel->exps;
	if (tmpexps){
		node *en; 

		for (en = tmpexps->h; en; en = en->next){
			sql_exp *l; 
			sql_exp *r; 
			tmpexp = (sql_exp *) en->data;
			if(tmpexp->type == e_cmp) {
				int to, from; 

				l = tmpexp->l; 
				r = tmpexp->r; 
				assert(l->type == e_column);
				assert(r->type == e_column); 
				printf("Join: [Table]%s.[Column]%s == [Table]%s.[Column]%s \n", l->rname, l->name, r->rname, r->name);
				from = rname_to_nodeId(nm, l->rname); 
				to = rname_to_nodeId(nm, r->rname); 
				printf("Node %d to Node %d\n", from, to); 
				if (isConnect[from][to] == 0){
					JP tmpjp = JP_NAV; 
					get_jp(l->name, r->name, &tmpjp); 
					printf("Join predicate = %d\n", tmpjp); 
					addJGedge(from, to, rel->op, jg, rel, tmpjp);
					isConnect[from][to] = 1;  
				}
				else{
					JP tmpjp = JP_NAV;
					printf("This edge is created \n"); 
					get_jp(l->name, r->name, &tmpjp);
					if (1) update_edge_jp(jg, from, to, tmpjp); 
					printf("Updated join predicate = %d\n", tmpjp); 
				}
			} else if (tmpexp->type == e_atom){
				//Only handle the case of 1
				//TODO: Handle other cases
				int tmpCond; 
				int from, to; 
				assert(tmpexp->l); 
				assert(atom_type(tmpexp->l)->type->localtype != TYPE_ptr);
				tmpCond = (int) atom_get_int(tmpexp->l); 
				//printf("Atom value %d \n",tmpCond);
				if (tmpCond == 1){
					printf("Join (condition 1) between %s and %s\n", rel_name((sql_rel*) rel->l), rel_name((sql_rel *)rel->r)); 
					from = rname_to_nodeId(nm,  rel_name((sql_rel*) rel->l));
					to = rname_to_nodeId(nm,  rel_name((sql_rel*) rel->r));	
					addJGedge(from, to, rel->op, jg, rel, JP_NAV);
				}
				continue; 
			} else {
				printf("This tmpexp->type == %d has not been handled yet\n", tmpexp->type); 
				assert(0); 
			}

		}
		printf("\n\n\n"); 
	}

}


static
void addJoinEdgesToJG(mvc *c, sql_rel *rel, int depth, jgraph *jg, int new_subjg, int *subjgId, nMap *nm, char **isConnect){
	//int tmpvid =-1; 

	switch (rel->op) {
		case op_left:
		case op_right:
		case op_join:
			if (new_subjg){ 	//The new subgraph flag is set
				*subjgId = *subjgId + 1; 
			}
			addJoinEdgesToJG(c, rel->l, depth+1, jg, 0, subjgId, nm, isConnect);
			addJoinEdgesToJG(c, rel->r, depth+1, jg, 0, subjgId, nm, isConnect);
			// Get the node Ids from 			
			_add_join_edges(jg, rel, nm, isConnect); 
			break; 
		case op_select: 
			if (isBasicPattern(rel)){
				//printf("Found a basic pattern\n");
				rel_print(c, rel, 0);
				//if (1) exps_print_ext(c, rel, 0); 
			}
			else{	//This is the connect to a new join sg

				//if is_join(((sql_rel *)rel->l)->op) printf("Join graph will be connected from here\n"); 
				addJoinEdgesToJG(c, rel->l, depth+1, jg, 1, subjgId, nm, isConnect);
			}
			break; 
		case op_basetable:
			break;
		default:		//op_project, topn,...
			if (rel->l){
				//if is_join(((sql_rel *)rel->l)->op) printf("Join graph will be connected from here\n"); 
				addJoinEdgesToJG(c, rel->l, depth+1, jg, 1, subjgId, nm, isConnect); 
			}
			if (rel->r){
				//if is_join(((sql_rel *)rel->l)->op) printf("Join graph will be connected from here\n"); 
				addJoinEdgesToJG(c, rel->r, depth+1, jg, 1, subjgId, nm, isConnect); 
			}
			break; 
			
	}

}

static 
char** createMatrix(int num, char initValue){
	int i, j; 
	char **m; 

	m = (char **)malloc(sizeof(char*) * num);
	for (i = 0; i < num; i++){
		m[i] = (char *)malloc(sizeof(char) * num);
		for (j = 0; j < num; j++){
			m[i][j] = initValue; 
		}
	}

	return m; 
}
static 
void freeMatrix(int num, char **m){
	int i; 
	for (i = 0; i < num; i++){
		free(m[i]);
	}
	free(m); 
}

static 
void _detect_star_pattern(jgraph *jg, jgnode *node, int pId){
	//Go through all edges of the node
	//If the edge has join predicate JP_S
	//and the to_node does not belong to
	//any pattern, add to the current pattern
	jgedge *tmpedge; 
	tmpedge = node->first; 
	while (tmpedge != NULL){
		if (tmpedge->jp == JP_S){
			jgnode *tonode = jg->lstnodes[tmpedge->to]; 
			if (tonode->patternId == -1){
				tonode->patternId = pId; 
				_detect_star_pattern(jg, tonode, pId); 
			} 
		}
		tmpedge = tmpedge->next; 
	}
}
static
void detect_star_pattern(jgraph *jg){
	
	int i; 
	int pId = -1; 
	int num = jg->nNode;

	for (i = 0; i < num; i++){
		jgnode *node = jg->lstnodes[i]; 
		if (node->patternId == -1){
			pId++;
			node->patternId = pId; 
			_detect_star_pattern(jg, node, pId); 	
		}
	}
}

void buildJoinGraph(mvc *c, sql_rel *r, int depth){
	//rel_print(c, r, 0);
	//Detect join between subject of triple table
	//
	// Go from the first sql_rel to the left and right of 
	// the sql_rel
	// If sql_rel has the op is op_join, check 
	// the exps (see rel_dump.c:264) in order to 
	// check from which table and condition 
	
	jgraph *jg; 
	nMap *nm = NULL; 
	int subjgId = -1; 
	int subjgId2 = -1;
	char **isConnect; //Matrix storing state whether two nodes are conneccted	
			  //In case of large sparse graph, this should not be used.

	(void) c; 
	(void) r; 
	(void) depth; 
	jg = initJGraph(); 
	
	addRelationsToJG(c, r, depth, jg, 0, &subjgId); 
	
	nm = create_nMap(MAX_JGRAPH_NODENUMBER); 
	add_relNames_to_nmap(jg, nm); 

	isConnect = createMatrix(jg->nNode, 0); 

	addJoinEdgesToJG(c, r, depth, jg, 0, &subjgId2, nm, isConnect);

	detect_star_pattern(jg); 

	printRel_JGraph(jg, c); 
	

	free_nMap(nm); 
	freeMatrix(jg->nNode, isConnect); 
	//printJGraph(jg); 

	freeJGraph(jg); 
	
	/*
	switch (r->op) {
		case op_join:
			if (((sql_rel *)r->l)->op == op_select && ((sql_rel *)r->r)->op == op_select){
				printf("POSSIBLE STAR PATTERN\n");		
			}
			else{
				if (((sql_rel *)r->l)->op != op_select) 
					buildJoinGraph(c, r->l, depth + 1);
				if (((sql_rel *)r->r)->op != op_select)
					buildJoinGraph(c, r->r, depth + 1);
			}
		case op_basetable:
			break; 	//Temporarily do nothign
		default:
			if (r->l) 
				buildJoinGraph(c, r->l, depth+1); 
			if (r->r)
				buildJoinGraph(c, r->r, depth+1); 
			
	}
	*/
}

