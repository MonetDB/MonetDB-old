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
#include <rdf.h>
#include <rel_dump.h>
#include <rel_select.h>
#include <rdflabels.h>
#include <rel_exp.h>
#include <sql_rdf.h>
#include <rdfschema.h>
#include <sql_rdf_rel.h>
#include <rel_rdfscan.h>

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

/*
 * Get the table from rdf schema
 * */

static void get_predicate_from_exps(mvc *c, list *tmpexps, char **prop, char **subj, int get_subject_only);

static
sql_table* get_rdf_table(mvc *c, char *tblname){
	sql_table *tbl = NULL; 
	str schema = "rdf"; 
	sql_schema *sch = NULL; 

	sch = mvc_bind_schema(c, schema); 

	assert(sch != NULL); 

	tbl = mvc_bind_table(c, sch, tblname); 

	assert (tbl != NULL); 

	return tbl; 

}


/*
 * Get the column of a table from rdf schema
 * */

static
sql_column* get_rdf_column(mvc *c, char *tblname, char *cname){
	sql_table *tbl = NULL; 
	str schema = "rdf"; 
	sql_schema *sch = NULL;
	sql_column *col = NULL;

	sch = mvc_bind_schema(c, schema); 

	assert(sch != NULL); 

	tbl = mvc_bind_table(c, sch, tblname); 

	assert (tbl != NULL); 

	col =  mvc_bind_column(c, tbl, cname);

	assert (col != NULL); 

	return col; 

}

#if HANDLING_EXCEPTION

static
sql_table *create_dummy_table(mvc *c, str tblname, list *proj_exps, int nump){
	
	sql_table *tbl = NULL;
	str schema = "rdf"; 
	sql_schema *sch = NULL; 

	node *en;
	//sql_subtype tpe_oid;		

	sch = mvc_bind_schema(c, schema); 
	assert(sch != NULL); 

	if ((tbl = mvc_bind_table(c, sch, tblname)) == NULL){
		printf("The dummy table does not exist --> Create new one\n"); 
		tbl = mvc_create_table(c, sch, tblname, tt_table, 0, SQL_PERSIST, 0, 3);
		//tbl = mvc_create_table(c, sch, tblname, tt_view, 0, SQL_PERSIST, 0, 3);
		//tbl = mvc_create_table(c, sch, tblname, tt_table, 0, SQL_LOCAL_TEMP, 0, 3);
	} else {
		drop_table(c, schema, tblname, 0);
		tbl = mvc_create_table(c, sch, tblname, tt_table, 0, SQL_PERSIST, 0, 3);
	}
		
	//create columns
	if (proj_exps != NULL){	//Create the column based on the an existing expression
		for (en = proj_exps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data; 
			sql_subtype *tpe; 
			char colname[100]; 

			assert(tmpexp->type == e_column); 
			tpe = exp_subtype(tmpexp); 
			sprintf(colname, "dummy_%s_%s", tmpexp->rname, tmpexp->name);
			if (mvc_bind_column(c, tbl, colname) == NULL){
				mvc_create_column(c, tbl, colname, tpe);
			}
		}
	} else {	//No existing expression
		int i;
		for (i = 0; i < nump; i++){
			char colname_s[50], colname_o[50]; 
			sql_subtype tpe; 
			sql_find_subtype(&tpe, "oid", 31, 0);
			sprintf(colname_s, "dummy_col_%d_s", i); 
			sprintf(colname_o, "dummy_col_%d_o", i); 		
			if (mvc_bind_column(c, tbl, colname_s) == NULL){
				mvc_create_column(c, tbl, colname_s, &tpe);
			}
			if (mvc_bind_column(c, tbl, colname_o) == NULL){
				mvc_create_column(c, tbl, colname_o, &tpe);
			}
		}
	}
	
	return tbl; 
}

#endif

static 
str create_abstract_table(mvc *c){
	sql_table *tbl = NULL;
	str schema = "rdf"; 
	sql_schema *sch = NULL; 

	if ((sch = mvc_bind_schema(c, schema)) == NULL){
	 	throw(SQL, "sql_rdf_jgraph", "3F000!schema missing");
	}

	if ((tbl = mvc_bind_table(c, sch, tbl_abstract_name)) == NULL){
		sql_subtype tpe; 
		sql_find_subtype(&tpe, "oid", 31, 0);

		tbl = mvc_create_table(c, sch, tbl_abstract_name, tt_table, 0, SQL_PERSIST, 0, 3);
		mvc_create_column(c, tbl, "id", &tpe);
		assert(tbl != NULL); 
	}

	return MAL_SUCCEED; 
}

static
str add_abstract_column(mvc *c, str cname){
	
	sql_table *tbl = NULL;
	sql_schema *sch = NULL; 
	str schema = "rdf"; 
	sql_subtype tpe; 
	sql_column *col = NULL; 

	(void) col; 

	sql_find_subtype(&tpe, default_abstbl_col_type, 100, 0);

	if ((sch = mvc_bind_schema(c, schema)) == NULL){
	 	throw(SQL, "sql_rdf_jgraph", "3F000!schema missing");
	}

	if ((tbl = mvc_bind_table(c, sch, tbl_abstract_name)) == NULL){
	 	throw(SQL, "sql_rdf_jgraph", "tblabstract has not been created\n");
	}
	
	if ((col =  mvc_bind_column(c, tbl, cname)) == NULL){
		mvc_create_column(c, tbl, cname, &tpe); 
	}
	else
		printf("Column %s has already created in abstract table\n", cname); 

	return MAL_SUCCEED; 
}


static 
int is_basic_pattern(sql_rel *r){
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
void exps_print_ext(mvc *sql, list *exps, int depth, char *prefix){
	(void) prefix; 
	mnstr_printf(THRdata[0], "%s ", prefix);
	exps_print(sql, THRdata[0], exps, depth, 1, 0);
	mnstr_printf(THRdata[0], "\n");
}
	

/*
static 
list *single_exp_list(sql_allocator *sa, sql_exp *e){
	list *lst = NULL; 
	lst = new_exp_list(sa);

	append(lst, e); 

	return lst; 
}
*/

/*
 * Return sys.isnull(e)
 * Right now, it is more like e = null
 * */
static 
sql_exp* exp_isnull(mvc *sql, sql_exp *e){
	sql_exp *isnull_exp = rel_unop_(sql, e, NULL, "isnull", card_value);
	return isnull_exp; 
}

static 
sql_exp* exp_isnotnull(mvc *sql, sql_exp *e){
	sql_exp *isnull_exp = rel_unop_(sql, e, NULL, "isnull", card_value);
	sql_exp *not_exp = exp_atom_bool(sql->sa, 0);
	
	sql_exp *isnotnull_exp = exp_compare(sql->sa, isnull_exp, not_exp, cmp_equal);

	return isnotnull_exp; 
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
		sprintf(tmp, "Node %d: [Pattern: %d] [Type: %d] ", i, tmpnode->patternId, tmpnode->type); 
		//mnstr_printf(sql->scanner.ws, "Node %d: ", i); 
		exps_print_ext(sql, ((sql_rel *) tmpnode->data)->exps, 0, tmp); 
		tmpedge = tmpnode->first; 
		while (tmpedge != NULL){
			assert(tmpedge->from == tmpnode->vid); 
			printf(" %d", tmpedge->to); 
			if ( (jg->lstnodes[tmpedge->to])->patternId != tmpnode->patternId)
				printf(" [connect pattern, r = %d, p_r = %d]", tmpedge->r_id, tmpedge->p_r_id); 
			else
				printf(" [r = %d, p_r = %d]", tmpedge->r_id, tmpedge->p_r_id);
			tmpedge = tmpedge->next; 
		}
		printf("\n"); 
		//mnstr_printf(sql->scanner.ws, "\n"); 
	}
	printf("---------------------\n"); 
}


/*
 * Get cross star pattern edges
 * */

static
jgedge** get_cross_sp_edges(jgraph *jg, int *num_cross_edges){
	int i; 
	int num = 0; 
	jgedge **lstedges = (jgedge **) malloc(sizeof(jgedge*) * (jg->nEdge)); //May be redundant
	for (i = 0; i < jg->nEdge; i++){
		lstedges[i] = NULL; 
	}
	for (i = 0; i  < jg->nNode; i++){
		jgnode *tmpnode = jg->lstnodes[i]; 
		jgedge *tmpedge = tmpnode->first; 
		while (tmpedge != NULL){
			if ( (jg->lstnodes[tmpedge->to])->patternId != tmpnode->patternId){ //cross pattern
					
				if (tmpedge->op == op_join){	//For inner join, only keep the edge from low node id -> high node id
					if (tmpedge->from < tmpedge->to){
						lstedges[num] = tmpedge;
						num++;
					}
				}
				else{
					lstedges[num] = tmpedge;
					num++;
				}
			}
			tmpedge = tmpedge->next; 
		}
	}

	*num_cross_edges = num; 
	return lstedges;
}

//Compute the order of applying the crossedges for 
//connecting star pattern. 
//Each cross edge is actually a join. 
//We based on the parent_rel_id of each edge
//in order to find this order.
//E.g., the cross edges may have rel_id and parent_rel_id as
//[1,0]  [0,-1]  [5,4]  [5,4]
//Then the apply order is 
//[5,4]  [5,4] [1,0] [0,-1]
//Note: The edges [5,4]  [5,4] are usually the join (NOT outer join)
//so that we only to apply one of them
static
int* get_crossedge_apply_orders(jgraph *jg, jgedge **lstEdges, int num){
	int* orders = NULL;
	int i, j, tmp; 

	orders = (int *) malloc(sizeof(int) * num); 

	for (i = 0; i < num; i++){
		orders[i] = i; 
	}

	for (i = 0; i < num; i++){
		jgedge *e1 = lstEdges[orders[i]]; 
		for (j = i+1; j < num; j++){
			jgedge *e2 = lstEdges[orders[j]];		
			if (e1->p_r_id < e2->p_r_id){
				tmp = orders[i];
				orders[i] = orders[j];
				orders[j] = tmp; 
			}
		}
	}
	
	printf("Orders of applying cross edges\n");
	for (i = 0; i < num; i++){
		int from = lstEdges[orders[i]]->from; 
		int to = lstEdges[orders[i]]->to;
		jgnode *fromnode = jg->lstnodes[from];
		jgnode *tonode = jg->lstnodes[to];
		printf("Cross edge [%d, %d][P%d -> P%d] [r = %d, p = %d][Exp_Need = %d]\n", from, to, fromnode->patternId, tonode->patternId, lstEdges[orders[i]]->r_id, lstEdges[orders[i]]->p_r_id, lstEdges[orders[i]]->need_add_exps);
	}

	return orders; 
}

static
void _add_jg_node(mvc *c, jgraph *jg, sql_rel *rel, int subjgId, JNodeT t){
	int tmpvid = -1;
	str subj = NULL; 
	str prop = NULL; 
	oid soid = BUN_NONE; 
	oid poid = BUN_NONE; 

	//Set subject oid if it is there
	if (rel->op==op_select){
		//Check whether there is constrainst on s_oid
		list *exps = rel->exps; 

		if (exps){
			get_predicate_from_exps(c, exps, &prop , &subj, 0);
		}
		if (subj){
			SQLrdfstrtoid(&soid, &subj);
			GDKfree(subj); 
			assert (soid != BUN_NONE); 
		}
		if (prop){
			
			//Get propId, assuming the tokenizer is open already 
			//Note that, the prop oid is the original one (before
			//running structural recognition process) so that 
			//we can directly get its oid from TKNR
			TKNRstringToOid(&poid, &prop); 
			assert (poid != BUN_NONE); 
		}
	}

	addJGnode(&tmpvid, jg, rel, subjgId, soid, poid, prop, t);

	if (prop) GDKfree(prop); 
}

/* Example
 * s10_t0.p, s10_t0.s, s10_t0.o, s10_t0.%TID% NOT NULL, s10_t1.p, s10_t1.s, s10_t1.o, s10_t1.%TID% NOT NULL, s10_t2.p, s10_t2.s, s10_t2.o, s10_t2.%TID% NOT NULL, s10_t3.p, s10_t3.s, s10_t3.o, s10_t3.%TID% NOT NULL, s10_t4.p, s10_t4.s, s10_t4.o, s10_t4.%TID% NOT NULL, sys.rdf_idtostr(s10_t0.o) as L1.L1 ]
 *
 * We remove .p and .%TID%
 * */


static 
list* remove_p_from_proj_exps(mvc *c, list *exps){
	
	node *en;
	sql_allocator *sa = c->sa;
	list *newexps = NULL; 
	newexps = new_exp_list(sa);
	for (en = exps->h; en; en = en->next){
		sql_exp *e = (sql_exp *) en->data; 

		if ((e->type == e_column && strcmp(e->name, "p") == 0) || 
		    (e->type == e_column && strcmp(e->name, "%TID%") == 0)){ //e.g., sys.rdf_idtostr(s10_t0.s) as L.product, sys.rdf_idtostr(s10_t0.o) as L.label
			continue; 
		} else {
			sql_exp *newexp = exp_copy(sa, e);
			//append this exp to list
			append(newexps, newexp);
		}

	}

	return newexps; 
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
void addRelationsToJG(mvc *c, sql_rel *parent, sql_rel *rel, int depth, jgraph *jg, int new_subjg, int *subjgId, int *level, int tmp_level, sql_rel **node_root, int *hasOuter){

	switch (rel->op) {
		case op_right:
			assert(0);	//This case is not handled yet
			break;
		case op_left:
		case op_join:
			if (rel->op == op_left || rel->op == op_right){ 
				printf("[Outter join]\n");
				*hasOuter = 1;
			}
			else printf("[join]\n"); 

			printf("--- Between %s and %s ---\n", op2string(((sql_rel *)rel->l)->op), op2string(((sql_rel *)rel->r)->op) );		
			
			if (new_subjg){ 	//The new subgraph flag is set
				*subjgId = *subjgId + 1; 
			}

			addRelationsToJG(c, rel, rel->l, depth+1, jg, 0, subjgId, level, tmp_level + 1, node_root, hasOuter);
			addRelationsToJG(c, rel, rel->r, depth+1, jg, 0, subjgId, level, tmp_level + 1, node_root, hasOuter);

			break; 
		case op_select: 
			 printf("[select]\n");
			if (is_basic_pattern(rel)){
				printf("Found a basic pattern\n");
				if (*level == 0){ 
					*level = tmp_level; 
					 *node_root = parent; 
				}
				_add_jg_node(c, jg, (sql_rel *) rel, *subjgId, JN_REQUIRED);
			}
			else{	//This is the connect to a new join sg
				addRelationsToJG(c, rel, rel->l, depth+1, jg, 1, subjgId, level, tmp_level + 1, node_root, hasOuter);
			}
			break; 
		case op_basetable:
			printf("[Base table]\n");		
			if (*level == 0){
				*level = tmp_level;
				*node_root = parent;
			}
			_add_jg_node(c, jg, (sql_rel *) rel, *subjgId, JN_REQUIRED);
			break;
		case op_project: 
			printf("[%s]\n", op2string(rel->op)); 
			//Update project expression in order to remove p from expressions
			rel->exps = remove_p_from_proj_exps(c, rel->exps); 
			
			if (rel->l) 
				addRelationsToJG(c, rel, rel->l, depth+1, jg, 1, subjgId, level, tmp_level + 1, node_root, hasOuter); 
			if (rel->r)
				addRelationsToJG(c, rel, rel->r, depth+1, jg, 1, subjgId, level, tmp_level + 1, node_root, hasOuter); 
			break; 
		case op_union:
			printf("[union] ==> Handling differently\n"); 
			assert (rel->l && rel->r); 
			buildJoinGraph(c, rel->l, depth + 1);
			buildJoinGraph(c, rel->r, depth + 1); 
			break; 
		default:
			printf("[%s]\n", op2string(rel->op)); 
			if (rel->l) 
				addRelationsToJG(c, rel, rel->l, depth+1, jg, 1, subjgId, level, tmp_level + 1, node_root, hasOuter); 
			if (rel->r)
				addRelationsToJG(c, rel, rel->r, depth+1, jg, 1, subjgId, level, tmp_level + 1, node_root, hasOuter); 
			break; 
			
	}

}


static
void handling_Union(mvc *c, sql_rel *rel, int depth, int *hasUnion){

	switch (rel->op) {
		case op_right:
			assert(0);	//This case is not handled yet
			break;
		case op_left:
		case op_join:
			handling_Union(c, rel->l, depth+1, hasUnion);
			handling_Union(c, rel->r, depth+1, hasUnion);

			break; 
		case op_select: 
			if (!is_basic_pattern(rel)){
				handling_Union(c, rel->l, depth+1, hasUnion);
			}
			break; 
		case op_basetable:
			break;
		case op_project: 
			if (rel->l) 
				handling_Union(c, rel->l, depth+1, hasUnion); 
			if (rel->r)
				handling_Union(c, rel->r, depth+1, hasUnion); 
			break; 
		case op_union:
			assert (rel->l && rel->r); 
			*hasUnion = 1; 
			buildJoinGraph(c, rel->l, depth + 1);
			buildJoinGraph(c, rel->r, depth + 1); 
			break; 
		default:
			if (rel->l) 
				handling_Union(c, rel->l, depth+1, hasUnion); 
			if (rel->r)
				handling_Union(c, rel->r, depth+1, hasUnion); 
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

static
int have_same_subj(jgraph *jg, int from, int to){

	jgnode *fromnode, *tonode; 
	
	fromnode = jg->lstnodes[from];
	tonode = jg->lstnodes[to]; 
	
	if ((fromnode->soid != BUN_NONE) && (fromnode->soid == tonode->soid)) return 1; 

	else return 0; 
}

static
void connect_same_subj_node(jgraph *jg){
	int nnode = jg->nNode; 
	int from = 0, to = 0; 

	for (from = 0; from < nnode; from++){
		for (to = (from + 1); to < nnode; to++){
			if (have_same_subj(jg, from, to) == 1){
				JP tmpjp = JP_S;
				add_undirectedJGedge(from, to, op_join, jg, NULL, tmpjp, -1, -1, 0);
			}
		}
	}
}

/*
 * Input: sql_rel with op == op_join, op_left or op_right
 * */
static
void _add_join_edges(jgraph *jg, sql_rel *rel, nMap *nm, char **isConnect, int rel_id, int p_rel_id){

	sql_exp *tmpexp;
	list *tmpexps; 
	int need_add_exps = 0; 

	assert(rel->op == op_join || rel->op == op_left || rel->op == op_right); 
	tmpexps = rel->exps;
	
	if (rel->op == op_left){
		printf("Add Join Edge via LEFT JOIN: \n");
	}

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

				//About the case "oid[s14_t4.o] < sys.sql_add(s14_t3.o, oid "120@0")"
				//e.g., q5 bsbm, just ignore
				if (l->type != e_column && r->type != e_column){
					JP tmpjp = JP_NAV;
					sql_exp *tmp = NULL; 
					list *lst = NULL;
					node *tmpen = NULL; 

					(void) tmpjp;
					assert(l->type == e_convert); 
					assert(r->type == e_func);

				        lst = r->l;
					tmpen = lst->h; 
				        
					//There can be two parameters in the list, however, we only need to check 
					//the first one --> s14_t3.o in the example
					tmp = (sql_exp *) tmpen->data;
					
					assert(tmp->type == e_column); 

					from = rname_to_nodeId(nm, ((sql_exp *)l->l)->rname); 
					
					to = rname_to_nodeId(nm, tmp->rname); 

					assert(rel->op == op_join); 
					
					printf("Add edge between patterns from node %d to node %d\n", from, to); 
					if (need_add_exps == 0){
						need_add_exps = 1; 
						add_undirectedJGedge(from, to, rel->op, jg, rel, tmpjp, rel_id, p_rel_id, 1);
					} else {
						add_undirectedJGedge(from, to, rel->op, jg, rel, tmpjp, rel_id, p_rel_id, 0);
					}

					continue; 

				}
				

				//For normal case 
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
					if (rel->op == op_join) add_undirectedJGedge(from, to, rel->op, jg, rel, tmpjp, rel_id, p_rel_id, 0);
					if (rel->op == op_left){ 
						add_directedJGedge(from, to, op_left, jg, rel, tmpjp, rel_id, p_rel_id, 0);
					}
					if (rel->op == op_right){ 
						add_directedJGedge(from, to, op_right, jg, rel, tmpjp, rel_id, p_rel_id, 0);
					}
					isConnect[from][to] = 1;  
				}
				else{
					JP tmpjp = JP_NAV;
					printf("This edge is created \n"); 
					get_jp(l->name, r->name, &tmpjp);
					if (1) update_undirectededge_jp(jg, from, to, tmpjp); 
					printf("Updated join predicate = %d\n", tmpjp); 
				}
			} else if (tmpexp->type == e_atom){
				//Only handle the case of 1
				//TODO: Handle other cases
				int tmpCond; 
				int from, to; 
				JP tmpjg = JP_NAV; 
				assert(tmpexp->l); 
				assert(atom_type(tmpexp->l)->type->localtype != TYPE_ptr);
				tmpCond = (int) atom_get_int(tmpexp->l); 
				//printf("Atom value %d \n",tmpCond);
				if (tmpCond == 1){
					printf("Join (condition 1) between %s and %s\n", rel_name((sql_rel*) rel->l), rel_name((sql_rel *)rel->r)); 
					from = rname_to_nodeId(nm,  rel_name((sql_rel*) rel->l));
					to = rname_to_nodeId(nm,  rel_name((sql_rel*) rel->r));	
					if (have_same_subj(jg, from, to) == 1){
						tmpjg = JP_S; 
					}
					if (rel->op == op_join) add_undirectedJGedge(from, to, rel->op, jg, rel, tmpjg, rel_id, p_rel_id, 0);
					else if (rel->op == op_left)	add_directedJGedge(from, to, op_left, jg, rel, tmpjg, rel_id, p_rel_id, 0);
					else assert(0);	//Other case is not handled yet
					
					printf("From: %d To %d\n", from, to); 
				}
				continue; 
			} else {
				printf("This tmpexp->type == %d has not been handled yet\n", tmpexp->type); 
				assert(0); 
			}

		}
		printf("\n\n\n"); 
	}
	else{
		str relname1; 
		str relname2;
		int from, to;
		JP tmpjp = JP_S;

		relname1 = rel_name((sql_rel*) rel->l);
		relname2 = rel_name((sql_rel*) rel->r);

		printf("CROSS PRODUCT HERE between %s and %s\n", relname1, relname2); 

		assert (rel->op == op_join); 

		from = rname_to_nodeId(nm, relname1); 
		to = rname_to_nodeId(nm, relname2); 

		if (have_same_subj(jg, from, to) == 1){
			printf("Connect to nodes having known subjects\n"); 
			add_undirectedJGedge(from, to, rel->op, jg, rel, tmpjp, rel_id, p_rel_id, 0);
		}
	}

}


static
void addJoinEdgesToJG(mvc *c, sql_rel *parent, sql_rel *rel, int depth, jgraph *jg, int new_subjg, int *subjgId, nMap *nm, char **isConnect, int *last_rel_join_id, int p_rel_id, int *level, int tmp_level, sql_rel **edge_root){
	int tmp_r_id; 

	switch (rel->op) {
		case op_right:
			assert(0);	//This case is not handled yet
			break;
		case op_left:
		case op_join:
			*last_rel_join_id = (*last_rel_join_id) + 1; 
			tmp_r_id = *last_rel_join_id;
			if (new_subjg){ 	//The new subgraph flag is set
				*subjgId = *subjgId + 1; 
			}

			if (*level == 0){
				*level = tmp_level; 	//Store the depth of the relational plan where the first join edge is found
				*edge_root = parent; 
			}

			addJoinEdgesToJG(c, rel, rel->l, depth+1, jg, 0, subjgId, nm, isConnect, last_rel_join_id, tmp_r_id, level, tmp_level+1, edge_root);
			addJoinEdgesToJG(c, rel, rel->r, depth+1, jg, 0, subjgId, nm, isConnect, last_rel_join_id, tmp_r_id, level, tmp_level+1, edge_root);

			// Get the node Ids from 			
			_add_join_edges(jg, rel, nm, isConnect, tmp_r_id, p_rel_id); 

			break; 
		case op_select: 
			if (is_basic_pattern(rel)){
				//printf("Found a basic pattern\n");
			}
			else{	//This is the connect to a new join sg
				//if is_join(((sql_rel *)rel->l)->op) printf("Join graph will be connected from here\n"); 
				addJoinEdgesToJG(c, rel, rel->l, depth+1, jg, 1, subjgId, nm, isConnect, last_rel_join_id, (*last_rel_join_id), level, tmp_level+1, edge_root);
			}
			break; 
		case op_basetable:
			break;
		default:		//op_project, topn,...
			if (rel->l){
				//if is_join(((sql_rel *)rel->l)->op) printf("Join graph will be connected from here\n"); 
				addJoinEdgesToJG(c, rel, rel->l, depth+1, jg, 1, subjgId, nm, isConnect, last_rel_join_id, (*last_rel_join_id), level, tmp_level + 1, edge_root); 
			}
			if (rel->r){
				//if is_join(((sql_rel *)rel->l)->op) printf("Join graph will be connected from here\n"); 
				addJoinEdgesToJG(c, rel, rel->r, depth+1, jg, 1, subjgId, nm, isConnect, last_rel_join_id, (*last_rel_join_id), level, tmp_level + 1, edge_root); 
			}
			break; 
			
	}

}

static
void connect_rel_with_sprel(sql_rel *rel, sql_rel *firstsp, int e_level, int n_level, sql_rel *node_root, sql_rel *edge_root){
	(void) rel; 

	if (e_level == 0){ //No join edge
		node_root->l = firstsp; 
	}
	else{
		if (e_level < n_level){
			edge_root->l = firstsp; 
		}
		else{
			node_root->l = firstsp;
		}
	}

	/* OLD CODE --> to be removed
	if (rel->l){
		if (((sql_rel*)rel->l)->op == op_join ||
		    ((sql_rel*)rel->l)->op == op_left ||			
		    ((sql_rel*)rel->l)->op == op_right ){
			rel->l = firstsp; 		
		}
		else{
			connect_rel_with_sprel(rel->l, firstsp, level + 1, connect_level);
		}
	}
	

	if (rel->r){
		if (((sql_rel*)rel->r)->op == op_join ||
		    ((sql_rel*)rel->r)->op == op_left ||			
		    ((sql_rel*)rel->r)->op == op_right ){
			rel->r = firstsp; 		
		}
		else{
			connect_rel_with_sprel(rel->r, firstsp, level + 1, connect_level);
		}
	}
	*/
}

/**
 * Use the combined edge betwwen sp to connect them.
 * Each combined edge is an join. 
 * Left is the
 */
static 
void connect_groups(int numsp, sql_rel **lstRels, sql_rel **lstEdgeRels){
	
	int i; 
	
	assert (numsp > 1); 

	for (i = 0; i < (numsp -2); i++){
		lstEdgeRels[i]->l = lstRels[i];
		lstEdgeRels[i]->r = lstEdgeRels[i+1];
	}

	lstEdgeRels[numsp -2]->l = lstRels[numsp - 2];
	lstEdgeRels[numsp -2]->r = lstRels[numsp - 1];
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
void _detect_star_pattern(jgraph *jg, jgnode *node, int pId, int _optm){
	//Go through all edges of the node
	//If the edge has join predicate JP_S
	//and the to_node does not belong to
	//any pattern, add to the current pattern
	
	jgedge *tmpedge; 
	tmpedge = node->first; 
	while (tmpedge != NULL){
		if (tmpedge->jp == JP_S){
			int optm = _optm; 
			jgnode *tonode = jg->lstnodes[tmpedge->to]; 

			/*
			if (tmpedge->op == op_left){ //left outer join
				optm = 1;
				tonode_ijpatternId++;
			}
			else if (tmpedge->op == op_right){
				assert(0); //Have not handle this case
			}
			*/

			if (tonode->patternId == -1){
				tonode->patternId = pId; 

				if (optm == 1) setNodeType(tonode, JN_OPTIONAL);

				_detect_star_pattern(jg, tonode, pId, optm); 
			}

		}
		tmpedge = tmpedge->next; 
	}
}

/*
 * Init property list (columns) extracted from
 * a star pattern
 * */
static 
spProps *init_sp_props(mvc *c, int num){
	int i; 
	spProps* spprops = NULL; 
	spprops = (spProps*) GDKmalloc(sizeof (spProps) ); 
	spprops->num = num; 
	spprops->subj = BUN_NONE; 
	spprops->lstProps = (char **) GDKmalloc(sizeof(char *) * num); 
	spprops->lstPropIds = (oid *) GDKmalloc(sizeof(oid) * num); 
	spprops->lstAlias = (char **) GDKmalloc(sizeof(char *) * num);
	spprops->lst_o_constraints = (o_constraint *) GDKmalloc(sizeof(o_constraint) * num); 
	spprops->lstPOs = (sp_po *) GDKmalloc(sizeof(sp_po) * num); 

	for (i = 0; i < num; i++){
		spprops->lstProps[i] = NULL; 
		spprops->lstAlias[i] = NULL; 
		spprops->lstPropIds[i] = BUN_NONE; 
		spprops->lst_o_constraints[i].cmp_type = -1; 
		spprops->lst_o_constraints[i].low = BUN_NONE; 
		spprops->lst_o_constraints[i].hi = BUN_NONE; 
		
		spprops->lstPOs[i] = NAV; 
	}
	spprops->lstctype = (ctype *) GDKmalloc(sizeof(ctype) * num); 

	spprops->exps = new_exp_list(c->sa);
	(void) c; 

	return spprops; 
}

static
void add_props_and_subj_to_spprops(spProps *spprops, int idx, sp_po po, jgnode *node){

	if (node->prop){
		str tmpalias = NULL; 
		sql_rel *tmprel = (sql_rel*) (node->data);
		assert(tmprel->op == op_select);
		assert(((sql_rel*)tmprel->l)->op == op_basetable); 

		tmpalias = get_relname_from_basetable(tmprel->l); 
		spprops->lstProps[idx] = GDKstrdup(node->prop); 
		spprops->lstAlias[idx] = GDKstrdup(tmpalias); 
		printf("\nTable alias in spprops is %s\n", spprops->lstAlias[idx]);

		assert(node->poid != BUN_NONE); 
		spprops->lstPropIds[idx] = node->poid; 
		spprops->lstPOs[idx] = po;

		//without any information, assuming that the column is single-valued col
		spprops->lstctype[idx] = CTYPE_SG; 
	}
	
	//For subject
	if (node->soid){
		if (spprops->subj == BUN_NONE){
			spprops->subj = node->soid; 
		}
		else
			assert(spprops->subj == node->soid); 
	}
}

	

static
void print_spprops(spProps *spprops){
	int i; 
	
	printf("List of properties from spProps: \n");
	for (i = 0; i < spprops->num; i++){
		printf("%s (Id: "BUNFMT "): " ,spprops->lstProps[i], spprops->lstPropIds[i]);	
		if (spprops->lstPOs[i] == REQUIRED) printf("[REQUIRED]"); 
		else printf("[NAV]");
		
		if (spprops->lst_o_constraints[i].low != BUN_NONE) printf(" [low = "BUNFMT"]", spprops->lst_o_constraints[i].low);
		if (spprops->lst_o_constraints[i].hi != BUN_NONE) printf(" [hi = "BUNFMT"]", spprops->lst_o_constraints[i].hi);

		printf("\n");
		
	}
	printf("\n"); 
}

static 
void free_sp_props(spProps *spprops){
	int i; 
	for (i = 0; i < spprops->num; i++){
		if (spprops->lstProps[i]) GDKfree(spprops->lstProps[i]); 
		if (spprops->lstAlias[i]) GDKfree(spprops->lstAlias[i]);
	}
	GDKfree(spprops->lstProps); 
	GDKfree(spprops->lstAlias);
	GDKfree(spprops->lstPropIds);
	GDKfree(spprops->lstPOs); 
	GDKfree(spprops->lstctype);
	GDKfree(spprops->lst_o_constraints);
	list_destroy(spprops->exps); 
	GDKfree(spprops); 
}


static
sql_exp* get_atom_oid(mvc *c, sql_exp *re){
	oid newoid; 
	atom *at = re->l;
	sql_exp *newre = NULL; 

	newoid = BUN_NONE; 
	assert(at != NULL); 

	printf("Atom expression \n");
	exp_print(c, THRdata[0] , re, 0,0,0);

	get_encodedOid_from_atom(at, &newoid);
	newre = exp_atom_oid(c->sa, newoid);

	return newre; 
}
/*
 * Get column name from exp of p in a sql_rel
 * Example: s12_t0.p = oid[sys.rdf_strtoid(char(67) "<http://www/product>")]
 * ==> column name = product.
 * This column must match with the table/column created from Characteristic Sets.
 * */
/*
static
void get_col_name_from_p (char **col, char *p){
	getPropNameShort(col, p);
}
*/

/*
 * Modify the tablename.colname in an
 * expression. 
 *
 * E.g., s12_t0.o = oid[sys.rdf_strtoid(char(85) "<http://www/Product9>"]
 * will be convert to tbl1.p = oid[sys.rdf_strtoid(char(85) "<http://www/Product9>"]
 * UPDATE: oid[s12_t0.o] = sys.rdf_strtoid(char(85) "<http://www/Product9>"
 * will be convert to tbl1.p = type_of_column_p[sys.rdf_strtoid(char(85) "<http://www/Product9>"]
 *
 * UPDATE: 
 * As everything will be stored as oid, we have to remove 
 *
 * */		

static
void modify_exp_col(mvc *c, sql_exp *m_exp,  char *_rname, char *_name, char *_arname, char *_aname, int update_e_convert, int dummy_exps){
	sql_exp *tmpe = NULL;
	sql_exp *ne = NULL;
	sql_exp *le = NULL; //right expression, should be e_convert
	
	str rname = GDKstrdup(_rname); 
	str name = GDKstrdup(_name);
	str arname = GDKstrdup(_arname);
	str aname = GDKstrdup(_aname);

	
	//exp_setname(sa, e, rname, name); 
	assert(m_exp->type == e_cmp); 

	le = (sql_exp *)m_exp->l; 
	assert(le->type == e_convert); 

	tmpe = le->l; 

	assert(tmpe->type == e_column); 
	
 	if (dummy_exps)
		ne = exp_column(c->sa, rname, name, exp_subtype(tmpe), exp_card(tmpe), has_nil(tmpe), 0);
	else 
		ne = exp_column(c->sa, arname, aname, exp_subtype(tmpe), exp_card(tmpe), has_nil(tmpe), 0);

	m_exp->l = ne; 
	
	if (update_e_convert){
		sql_exp *newre = NULL;
		sql_column *col = get_rdf_column(c, rname, name);
		sql_subtype totype = col->type;
		sql_exp *re = m_exp->r;
		(void) re;
		(void) totype; 
		assert(le->type == e_convert && ne); 
		
		#if EVERYTHING_AS_OID
		//first: Convert the compared value into oid
		if (re->type == e_atom){
			newre = get_atom_oid(c, re);
		} else {
			newre = exp_convert(c->sa, m_exp->r, exp_fromtype(le), &totype);	
		}
		#else
		newre = exp_convert(c->sa, m_exp->r, exp_fromtype(le), &totype);
		#endif

		m_exp->r = newre; 

	}

}


/*
 * oid[sys.rdf_strtoid(char(67) "<http://www/product>")] 
 * UPDATE: sys.rdf_strtoid(char(67) "<http://www/product>")
 * returns <http://www/product>
 * */
static 
void extractURI_from_exp(mvc *c, char **uri, sql_exp *exp){

	sql_exp *tmpexp;
	node *tmpen; 
	str s = NULL;
	list *lst = NULL; 
	char *funcname; 

	assert(exp->type == e_func); 

	funcname = ((sql_subfunc *)exp->f)->func->base.name; 

	assert(strcmp(funcname, "rdf_strtoid") == 0); 
	
	lst = exp->l;
	
	//There should be only one parameter for the function which is the property name
	tmpen = lst->h; 
	tmpexp = (sql_exp *) tmpen->data;
				
	s = atom2string(c->sa, (atom *) tmpexp->l); 
	*uri = GDKstrdup(s); 

	//get_col_name_from_p (&col, s);
	//printf("%s --> corresponding column %s\n", *prop,  col); 

}

static
void get_o_constraint_value(mvc *c, sql_exp *m_exp, oid *tmpvalue){
	oid newoid; 
	sql_exp *re = m_exp->r;

	assert(m_exp->type == e_cmp); 

	//first: Convert the compared value into oid
	newoid = BUN_NONE; 

	if (re->type == e_atom){
		atom *at = re->l;
		assert(at != NULL); 

		get_encodedOid_from_atom(at, &newoid);
	} else if (re->type == e_func) {
		//Check whether this is the function of rdf_strtoid
		char *funcname = ((sql_subfunc *)re->f)->func->base.name;
		str uri = NULL;

		if (strcmp(funcname, "rdf_strtoid") == 0){
			extractURI_from_exp(c, &uri, re);
			SQLrdfstrtoid(&newoid, &uri);
                        assert (newoid != BUN_NONE);

		} else {
			printf("TODO: The function %s is not handled yet\n", funcname);		
		}

		if (!uri) GDKfree(uri); 
	} else {
		printf("TODO: This is not handled yet\n");
	}

	*tmpvalue = newoid; 
}



/*
 * //Example: [s12_t0.p = oid[sys.rdf_strtoid(char(67) "<http://www/product>")], s12_t0.o = oid[sys.rdf_strtoid(char(85) "<http://www/Product9>"]
 * // UPDATED: Example: [oid[s12_t0.p] = sys.rdf_strtoid(char(67) "<http://www/product>"), oid[s12_t0.o] = sys.rdf_strtoid(char(85) "<http://www/Product9>" 
 *
 * */
static 
void get_predicate_from_exps(mvc *c, list *tmpexps, char **prop, char **subj, int get_subject_only){


	node *en;
	int num_p_cond = 0; 

	assert (tmpexps != NULL);
	for (en = tmpexps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data; 
		sql_exp *colexp = NULL;

		assert(tmpexp->type == e_cmp); //TODO: Handle other exps for op_select

		if (tmpexp->flag != cmp_equal) {
			printf("CANNOT get predicate/subj info from non-equal comparasion\n"); 
			continue; 
		}
		
		//Example: [s12_t0.p = oid[sys.rdf_strtoid(char(67) "<http://www/product>")], s12_t0.o = oid[sys.rdf_strtoid(char(85) "<http://www/Product9>"]
		assert(((sql_exp *)tmpexp->l)->type == e_convert); 

		colexp = ((sql_exp *)tmpexp->l)->l; 

		assert(colexp->type == e_column); 
		//Check if the column name is p, then
		//extract the input property name
		if (strcmp(colexp->name, "p") == 0){
			num_p_cond++; 
			if (get_subject_only) continue; 	
			extractURI_from_exp(c, prop, (sql_exp *)tmpexp->r);	
			//In case the column name is not in the abstract table, add it
			if (0) add_abstract_column(c, *prop);

		} else if (strcmp(colexp->name, "s") == 0) {
			extractURI_from_exp(c, subj, (sql_exp *)tmpexp->r);	
		} else{ 
			continue; 
		}


	}

	if (!get_subject_only)
		assert(num_p_cond == 1 && (*prop) != NULL); //Verify that there is only one p in this op_select sql_rel 

}


/*
 * From op_select sql_rel, get the condition on p  
 * can indicate the column name of the corresponding table
 *
 * */
static
void verify_rel(sql_rel *r){
	
	list *tmpexps = NULL; 
	char select_s = 0, select_p = 0, select_o = 0; 
	//str col; 

	assert(r->op == op_select);
	assert(((sql_rel*)r->l)->op == op_basetable); 
	
	//Verify that this select function select all three columns s, p, o
		
	tmpexps = ((sql_rel*)r->l)->exps;
	if (tmpexps){
		node *en;
	
		for (en = tmpexps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data; 
			assert(tmpexp->type == e_column);
			if (strcmp(tmpexp->name, "s") == 0) select_s = 1; 
			if (strcmp(tmpexp->name, "p") == 0) select_p = 1; 
			if (strcmp(tmpexp->name, "o") == 0) select_o = 1; 
		}
	}

	assert(select_s && select_p && select_o);
}

static
void get_o_constraint(mvc *c, o_constraint *o_cst, list *exps){
	node *en;
	for (en = exps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data;
		sql_exp *e = (sql_exp *)tmpexp->l;

		assert(tmpexp->type == e_cmp); //TODO: Handle other exps for op_select
		assert(e->type == e_convert); 

		e = e->l; 

		assert(e->type == e_column); 

		if (strcmp(e->name, "o") == 0){
			int cmp = get_cmp(tmpexp); 		
			oid tmp_o_value = BUN_NONE; 
			get_o_constraint_value(c, tmpexp, &tmp_o_value); 

			o_cst->cmp_type = cmp; 

			switch(cmp) {
			case cmp_equal:
				o_cst->low = tmp_o_value; 
				o_cst->hi = tmp_o_value; 
				break; 
			case cmp_gt: 
				if (tmp_o_value != BUN_NONE) o_cst->low = tmp_o_value + 1;  
				break;
			case cmp_gte: 	
				o_cst->low = tmp_o_value;
				break;
			case cmp_lte: 	
				o_cst->hi = tmp_o_value;	
				break;
			case cmp_lt: 	
				if (tmp_o_value != BUN_NONE) o_cst->hi = tmp_o_value - 1; 
				break;
			//All other cases are not handled yet
			case cmp_notequal:
			case cmp_all: 
			case cmp_or: 
			case cmp_in:
			case cmp_notin: 
			case cmp_filter: 
			default:
				o_cst->cmp_type = -1; 		
				break; 
			}
		}
	}
}

static 
void get_transform_select_exps(mvc *c, list *exps, list *trans_select_exps, str tblname, str colname, int *isConstrain_o){
	
	node *en;
	int num_o_cond = 0;
	int num_s_cond = 0; 
	sql_allocator *sa = c->sa;

	for (en = exps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data; 
		sql_exp *e = (sql_exp *)tmpexp->l; 

		assert(tmpexp->type == e_cmp); //TODO: Handle other exps for op_select
		assert(e->type == e_convert); 

		e = e->l; 

		assert(e->type == e_column); 

		if (strcmp(e->name, "p") == 0){
			continue; 

		} else if (strcmp(e->name, "o") == 0){
			sql_exp *m_exp = exp_copy(sa, tmpexp);
			modify_exp_col(c, m_exp, tblname, colname, e->rname, e->name, 1, 0);

			//append this exp to list
			append(trans_select_exps, m_exp);
			num_o_cond++;
			*isConstrain_o = 1; 

		} else if (strcmp(e->name, "s") == 0){
			char subj_colname[50] = "subject";
			sql_exp *m_exp = exp_copy(sa, tmpexp);
			modify_exp_col(c, m_exp, tblname, subj_colname, e->rname, e->name, 1, 0);

			//append this exp to list
			append(trans_select_exps, m_exp);
			num_s_cond++;
		} else{ 
			printf("The exp of other predicates (not s, p, o) is not handled\n"); 
		}


	}
}



#if HANDLING_EXCEPTION
static 
void get_transform_dummy_select_exps(mvc *c, list *exps, list *trans_select_exps, str tblname){
	
	node *en;
	sql_allocator *sa = c->sa;

	for (en = exps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data; 
		sql_exp *e = (sql_exp *)tmpexp->l; 

		assert(tmpexp->type == e_cmp); //TODO: Handle other exps for op_select
		assert(e->type == e_convert); 

		e = e->l; 

		assert(e->type == e_column); 

		if (strcmp(e->name, "p") == 0){
			continue; 

		} else if (strcmp(e->name, "o") == 0 || strcmp(e->name, "s") == 0){
			sql_exp *m_exp = exp_copy(sa, tmpexp);
			char dum_col[100]; 
			sprintf(dum_col,"dummy_%s_%s", e->rname, e->name); 
			modify_exp_col(c, m_exp, tblname, dum_col, e->rname, e->name, 1, 1);

			//append this exp to list
			append(trans_select_exps, m_exp);

		} else{ 
			printf("The exp of other predicates (not s, p, o) is not handled\n"); 
		}


	}
}
#endif

/*
 * If there is a missing column in the rel 
 * (this only happens to the rel of optional pattern group)
 * then create rel of null columns
 * e.g., select s11.s, null as s11.o from t
 * */
static
void create_null_exps(mvc *c, sql_rel *r, list *trans_tbl_exps, list *opt_exps, list *sp_prj_exps, str tblname){

	list *tmp_tbl_exps = NULL; 
	sql_allocator *sa = c->sa; 
	sql_rel *tbl_rel = NULL;
	
	
	/*
	 * Change the list of column from base table
	 * */
	assert (((sql_rel *)r->l)->op == op_basetable);
	tbl_rel = (sql_rel *)r->l;
	tmp_tbl_exps = tbl_rel->exps; 
	
	if (tmp_tbl_exps){
		node *en; 
		for (en = tmp_tbl_exps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data;
			assert(tmpexp->type == e_column); 
			if (strcmp(tmpexp->name, "o") == 0){

				//New e with old alias
				sql_exp *opt_e = NULL; 
				sql_exp *proj_e = NULL; 
				sql_exp *exp_null = exp_atom(sa, atom_general(sa, exp_subtype(tmpexp), NULL));
				exp_setname(sa, exp_null, tmpexp->rname, tmpexp->name);
				append(trans_tbl_exps, exp_null); 
				opt_e = exp_copy(sa, exp_null); 
				proj_e = exp_copy(sa, exp_null); 
				append(opt_exps, opt_e); 
				if (sp_prj_exps) append(sp_prj_exps, proj_e);
			}

			if (strcmp(tmpexp->name, "s") == 0){
				//New e with old alias
				char subj_colname[50] = "subject";
				str origcolname = GDKstrdup(subj_colname);
				str origtblname = GDKstrdup(tblname);
				sql_column *tmpcol = get_rdf_column(c, origtblname, origcolname);
				sql_exp *e = exp_alias(sa, tmpexp->rname, tmpexp->name, origtblname, origcolname, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *proj_e = exp_alias(sa, tmpexp->rname, tmpexp->name, tmpexp->rname, tmpexp->name, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *opt_e = exp_copy(sa, proj_e); 

				append(trans_tbl_exps, e); 
				append(opt_exps, opt_e);
				if (sp_prj_exps) append(sp_prj_exps, proj_e);
			}

		}
	}
}

static
void tranforms_exps(mvc *c, sql_rel *r, list *trans_select_exps, list *trans_tbl_exps, str tblname, int colIdx, oid tmpPropId, str *atblname, str *asubjcolname, list *sp_prj_exps, list *base_column_exps, int isOptionalGroup){

	list *tmpexps = NULL; 
	list *tmp_tbl_exps = NULL; 
	sql_allocator *sa = c->sa; 
	char tmpcolname[100]; //TODO: Should we use char[]
	sql_rel *tbl_rel = NULL;
	int isConstrain_o = 0; 
	
	printf("Converting op_select in star pattern to sql_rel of corresponding table\n"); 
	//Get the column name by checking exps of r
	

	getColSQLname(tmpcolname, colIdx, -1, tmpPropId, global_mapi, global_mbat);

	printf("In transform column %d --> corresponding column %s\n", colIdx,  tmpcolname); 
	
	tmpexps = r->exps;

	if (tmpexps){
		get_transform_select_exps(c, tmpexps, trans_select_exps, tblname, tmpcolname, &isConstrain_o); 
	}


	/*
	 * Change the list of column from base table
	 * */
	assert (((sql_rel *)r->l)->op == op_basetable);
	tbl_rel = (sql_rel *)r->l;
	tmp_tbl_exps = tbl_rel->exps; 
	
	if (tmp_tbl_exps){
		node *en; 
		for (en = tmp_tbl_exps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data;
			assert(tmpexp->type == e_column); 
			if (strcmp(tmpexp->name, "o") == 0){
				//New e with old alias
				str origcolname = GDKstrdup(tmpcolname);
				str origtblname = GDKstrdup(tblname);
				sql_column *tmpcol = get_rdf_column(c, tblname, origcolname);
				sql_exp *e = exp_alias(sa, tmpexp->rname, tmpexp->name, origtblname, origcolname, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *proj_e = exp_alias(sa, tmpexp->rname, tmpexp->name, tmpexp->rname, tmpexp->name, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);	
				sql_exp *base_col_e = exp_copy(sa, proj_e);
				sql_exp *notnull_m_exp = NULL; 

				printf("tmpcolname in rdf basetable is %s\n", tmpcolname);
				append(trans_tbl_exps, e); 
				if (sp_prj_exps) append(sp_prj_exps, proj_e); 
				if (base_column_exps) append(base_column_exps, base_col_e);
				
				if (isConstrain_o == 0 && isOptionalGroup == 0){
					//Add not NULL condition if
					//there is no constrain on o yet.
					sql_exp *base_col_dup = exp_copy(sa, base_col_e); 
					notnull_m_exp = exp_isnotnull(c, base_col_dup); 
					append(trans_select_exps, notnull_m_exp);
				}
			}

			if (strcmp(tmpexp->name, "s") == 0){
				//New e with old alias
				char subj_colname[50] = "subject";
				str origcolname = GDKstrdup(subj_colname);
				str origtblname = GDKstrdup(tblname);
				sql_column *tmpcol = get_rdf_column(c, origtblname, origcolname);
				sql_exp *e = exp_alias(sa, tmpexp->rname, tmpexp->name, origtblname, origcolname, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *proj_e = exp_alias(sa, tmpexp->rname, tmpexp->name, tmpexp->rname, tmpexp->name, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *base_col_e = exp_copy(sa, proj_e);

				if (*atblname == NULL){
					*atblname = GDKstrdup(tmpexp->rname);
					*asubjcolname = GDKstrdup(tmpexp->name);
				}
				append(trans_tbl_exps, e); 
				if (sp_prj_exps) append(sp_prj_exps, proj_e); 
				if (base_column_exps) append(base_column_exps, base_col_e); 
			}

		}
	}
}


static
void tranforms_mvprop_exps(mvc *c, sql_rel *r, mvPropRel *mvproprel, int tblId, oid tblnameoid, int colIdx, oid tmpPropId, int isMVcol, list *sp_prj_exps, list *base_column_exps){

	list *tmpexps = NULL; 
	list *trans_select_exps = NULL; 
	list *trans_tbl_exps = NULL; 
	list *tmp_tbl_exps = NULL; 
	sql_allocator *sa = c->sa; 
	char tmpmvcolname[100]; //TODO: Should we use char[]
	sql_rel *tmptbl_rel = NULL, *rel_mv_basetbl = NULL, *rel_mv_select = NULL;
	char mvtblname[100];
	
	printf("Converting op_select in star pattern to sql_rel of corresponding table\n"); 
	//Get the column name by checking exps of r
	
	if (isMVcol > 1){
		printf("TODO: HANDLE the case of multi-type prop\n");
	}

	//Default-type column in MV table 
	getMvTblSQLname(mvtblname, tblId, colIdx, tblnameoid, tmpPropId, global_mapi, global_mbat);
	getColSQLname(tmpmvcolname, colIdx, 0, tmpPropId, global_mapi, global_mbat);	

	printf("In transform mv col %d --> corresponding column %s\n", colIdx,  tmpmvcolname); 
	
	tmpexps = r->exps;
	
	//Store column and table names
	mvproprel->cname = GDKstrdup(tmpmvcolname);
	mvproprel->mvtblname = GDKstrdup(mvtblname);


	if (tmpexps){
		node *en;
	
		trans_select_exps = new_exp_list(c->sa);
		for (en = tmpexps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data; 
			sql_exp *e = (sql_exp *)tmpexp->l; 

			assert(tmpexp->type == e_cmp); //TODO: Handle other exps for op_select
			assert(e->type == e_convert); 

			e = e->l; 

			assert(e->type == e_column); 

			if (strcmp(e->name, "p") == 0){
				continue; 

			} else if (strcmp(e->name, "o") == 0){
				sql_exp *m_exp = exp_copy(sa, tmpexp);
				modify_exp_col(c, m_exp, mvtblname, tmpmvcolname, e->rname, e->name, 1, 0);
				
				//append this exp to list
				append(trans_select_exps, m_exp);

			} else if (strcmp(e->name, "s") == 0){
				char subj_colname[50] = "mvsubj";
				sql_exp *m_exp = exp_copy(sa, tmpexp);
				modify_exp_col(c, m_exp, mvtblname, subj_colname, e->rname, e->name, 1, 0);

				//append this exp to list
				append(trans_select_exps, m_exp);
			} else{ 
				printf("The exp of other predicates (not s, p, o) is not handled\n"); 
			}


		}

	}


	/*
	 * Change the list of column from base table
	 * */
	assert (((sql_rel *)r->l)->op == op_basetable);
	tmptbl_rel = (sql_rel *)r->l;
	tmp_tbl_exps = tmptbl_rel->exps; 
	
	if (tmp_tbl_exps){
		node *en; 

		trans_tbl_exps = new_exp_list(c->sa);
		for (en = tmp_tbl_exps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data;
			assert(tmpexp->type == e_column); 
			if (strcmp(tmpexp->name, "o") == 0){
				//New e with old alias
				str origcolname = GDKstrdup(tmpmvcolname);
				str origtblname = GDKstrdup(mvtblname);
				sql_column *tmpcol = get_rdf_column(c, mvtblname, origcolname);
				sql_exp *e = exp_alias(sa, tmpexp->rname, tmpexp->name, origtblname, origcolname, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *proj_e = exp_alias(sa, tmpexp->rname, tmpexp->name, tmpexp->rname, tmpexp->name, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *base_col_e = exp_copy(sa, proj_e);

				printf("tmpmvcolname in rdf basetable is %s\n", tmpmvcolname);
				append(trans_tbl_exps, e); 
				append(sp_prj_exps, proj_e);
				if (base_column_exps) append(base_column_exps, base_col_e);
			}

			if (strcmp(tmpexp->name, "s") == 0){
				//New e with old alias
				char subj_colname[50] = "mvsubj";
				str origcolname = GDKstrdup(subj_colname);
				str origtblname = GDKstrdup(mvtblname);
				sql_column *tmpcol = get_rdf_column(c, origtblname, origcolname);
				sql_exp *e = exp_alias(sa, tmpexp->rname, tmpexp->name, origtblname, origcolname, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *proj_e = exp_alias(sa, tmpexp->rname, tmpexp->name, tmpexp->rname, tmpexp->name, &tmpcol->type, CARD_MULTI, tmpcol->null, 0);
				sql_exp *base_col_e = exp_copy(sa, proj_e);

				append(trans_tbl_exps, e); 
				append(sp_prj_exps, proj_e);
				if (base_column_exps) append(base_column_exps, base_col_e);

				mvproprel->atblname = GDKstrdup(tmpexp->rname);
				mvproprel->asubjcolname = GDKstrdup(tmpexp->name);

			}

		}
	}


	rel_mv_basetbl = rel_basetable(c, get_rdf_table(c,mvtblname), mvtblname); 

	rel_mv_basetbl->exps = trans_tbl_exps;

	//rel_print(c, rel_mv_basetbl, 0); 
	
	rel_mv_select = rel_select_copy(c->sa, rel_mv_basetbl, trans_select_exps); 

	mvproprel->mvrel = rel_mv_select; 	
	
	//rel_print(c, mvproprel->mvrel, 0);

	//if (trans_select_exps) list_destroy(trans_select_exps);

}

/*
 * Output: 
 * num_match_tbl: Number of table having columns of these input props
 * rettbId: List of matching tables 
 *
 * */
static 
void get_matching_tbl_from_spprops(int **rettbId, spProps *spprops, int *num_match_tbl){

	oid *lstprop = NULL; 	//list of distinct prop, sorted by prop ids
	int num; 		//number of of distinct sorted props
	int i,j; 
	int **tmptblId; 	//lists of tables corresonding to each prop
	int *count; 		//number of tables per prop	
	int *tblId; 		//list of matching tblId
	int numtbl = 0; 	
	str tblname; 

	oid *tmplst = NULL;
	int tmpnum = 0;

	tmplst = (oid *) malloc(sizeof(oid) * spprops->num); 
	for (i = 0; i < spprops->num; i++){
		#if GETMATCHING_TBL_BY_RP_ONLY
		if (spprops->lstPOs[i] == REQUIRED){
			tmplst[tmpnum] = spprops->lstPropIds[i];
			tmpnum++;
		}
		#else
		tmplst[tmpnum] = spprops->lstPropIds[i];
		tmpnum++;
		#endif
	} 

	get_sorted_distinct_set(tmplst, &lstprop, tmpnum, &num);

	free(tmplst); 

	if (spprops->subj != BUN_NONE){
		int tblIdx;
		oid baseSoid; 
		getTblIdxFromS(spprops->subj, &tblIdx, &baseSoid);
		numtbl = 1;
		tblId = (int *) malloc(sizeof(int));
		tblId[0] = tblIdx;

		printf("Table Id found based on known subj is: %d\n", tblIdx); 
	}
	else{
		
		tmptblId = (int **) malloc(sizeof(int *) * num); 
		count = (int *) malloc(sizeof(int) * num); 

		printf("Table Id for set of props [");
		for (i = 0; i < num; i++){
			//Postinglist pl = get_p_postingList(global_p_propstat, lstprop[i]);
			Postinglist *pl = get_p_postingList(global_c_propstat, lstprop[i]);
			if (pl != NULL){
				tmptblId[i] = pl->lstIdx;
				count[i] = pl->numAdded; 
				printf("  " BUNFMT, lstprop[i]);
			} else {
				printf(" NO TABLE"); 
				break; 
			}

		}
		
		if (i == num)	//All props have matching tabe
			intersect_intsets(tmptblId, count, num, &tblId,  &numtbl);

		printf(" ] --> ");

		free(count); 

	}
	
	*rettbId = (int *) malloc(sizeof(int) * numtbl); 

	for (i = 0; i < numtbl; i++){
		int tId = tblId[i];
		oid tblnameoid = global_csset->items[tId]->tblname; 

		(*rettbId)[i] = tId; 
		
		tblname = (str) GDKmalloc(sizeof(char) * 100); 

		getTblSQLname(tblname, tId, 0,  tblnameoid, global_mapi, global_mbat);

		printf("  %d [Name of the table  %s]", tId, tblname);  

		//Get the corresponding column names in this table
		printf("\n--> Corresponding column names: \n");
		for (j = 0; j < num; j++){
			char tmpcolname[100];
			int colIdx = getColIdx_from_oid(tId, global_csset, lstprop[j]);
			int isMVcol = isMVCol(tId, colIdx, global_csset);
			getColSQLname(tmpcolname, colIdx, -1, lstprop[j], global_mapi, global_mbat);
			printf("Col %d: %s (isMV: %d)\n",j, tmpcolname, isMVcol);

		}
	}

	printf("\n"); 

	*num_match_tbl = numtbl; 

}

/* 
 * lstRP: list of required props
 * num: Num of RP
 * subj: Subject oid
 * num_possible_tbl: Number of tables that, by combining with exceptions, can contain these set of RP props 
 * ret_pos_tbId: List of possible tables
 */
void get_possible_matching_tbl_from_RPs(int **rettbId, int *num_match_tbl, oid *lstRP, int num, oid subj){

	int i; 
	int **tmptblId; 	//lists of tables corresonding to each prop
	int *count; 		//number of tables per prop	
	int *tblId; 		//list of matching tblId
	int numtbl = 0; 	

	if (subj != BUN_NONE){
		int tblIdx;
		oid baseSoid; 
		getTblIdxFromS(subj, &tblIdx, &baseSoid);
		numtbl = 1;
		tblId = (int *) malloc(sizeof(int));
		tblId[0] = tblIdx;
		printf("Possible table Id found based on known subj is: %d\n", tblIdx); 
	}
	else{
		
		tmptblId = (int **) malloc(sizeof(int *) * num); 
		count = (int *) malloc(sizeof(int) * num); 

		printf("Possible table Id for set of props [");
		for (i = 0; i < num; i++){
			Postinglist *pl = get_p_postingList(global_p_propstat, lstRP[i]);
			if (pl != NULL){
				tmptblId[i] = pl->lstIdx;
				count[i] = pl->numAdded; 
				printf("  " BUNFMT, lstRP[i]);
			} else {
				printf(" NO TABLE"); 
				break; 
			}

		}
		
		if (i == num)	//All props have matching tabe
			intersect_intsets(tmptblId, count, num, &tblId,  &numtbl);

		printf(" ] --> ");

		free(count); 

	}
	
	*rettbId = (int *) malloc(sizeof(int) * numtbl); 
	for (i = 0; i < numtbl; i++){
		(*rettbId)[i] = tblId[i];
	}

	*num_match_tbl = numtbl; 

}

static
mvPropRel* init_mvPropRelSet(int n){
	int i; 
	mvPropRel *mvPropRels;
	mvPropRels = (mvPropRel *) GDKmalloc(sizeof(mvPropRel) * n); 
	for (i = 0; i < n; i++){
		mvPropRels[i].cname = NULL;
		mvPropRels[i].mvtblname = NULL; 
		mvPropRels[i].mvrel = NULL; 
		mvPropRels[i].atblname = NULL; 
		mvPropRels[i].asubjcolname = NULL; 
		//mvPropRels[i].mvjoinexps = NULL; 
	}

	return mvPropRels;
}

static void
free_mvPropRelSet(mvPropRel *mvPropRels, int n){
	int i;
	for (i = 0; i < n; i++){
		if (mvPropRels[i].cname)
			GDKfree(mvPropRels[i].cname);
		if (mvPropRels[i].mvtblname)
			GDKfree(mvPropRels[i].mvtblname);
		if (mvPropRels[i].atblname)
			GDKfree(mvPropRels[i].atblname);
		if (mvPropRels[i].asubjcolname)
			GDKfree(mvPropRels[i].asubjcolname);
		//if (mvPropRels[i].mvjoinexps)
		//	list_destroy(mvPropRels[i].mvjoinexps);
	}

	GDKfree(mvPropRels);
}


static
sql_rel *rel_innerjoin(sql_allocator *sa, sql_rel *l, sql_rel *r, list *exps, operator_type join)
{
	sql_rel *rel = rel_create(sa);

	rel->l = l;
	rel->r = r;
	rel->op = join;
	rel->exps = exps;
	rel->card = CARD_MULTI;
	rel->nrcols = l->nrcols + r->nrcols;
	return rel;
}

static 
sql_rel *join_two_rels(mvc *c, sql_rel *l, str lrname, str lcolname, sql_rel *r, str rrname, str rcolname){
	sql_rel *rel = NULL; 
	sql_exp *tmpexp = NULL; 
	sql_exp *expcol1 = NULL; 
	sql_exp *expcol2 = NULL; 
	list *joinexps = NULL; 
	sql_subtype tpe; 

	sql_find_subtype(&tpe, "oid", 31, 0);

	//TODO: Check the input params
	
	expcol1 = exp_column(c->sa, lrname, lcolname, &tpe , CARD_MULTI, 1, 0);
	expcol2 = exp_column(c->sa, rrname, rcolname, &tpe , CARD_MULTI, 1, 0);

	tmpexp = exp_compare(c->sa, expcol1, expcol2, cmp_equal);

	joinexps = new_exp_list(c->sa);

	append(joinexps, tmpexp);

	rel = rel_innerjoin(c->sa, l, r, joinexps, op_join);
	
	return rel; 
}

static 
sql_rel *connect_sp_select_and_mv_prop(mvc *c, sql_rel *rel_wo_mv, mvPropRel *mvPropRels, str tblname, str atblname, str asubjcolname, int nnode){
	int i; 
	sql_rel *rel = NULL; 
	(void) tblname; 

	if (rel_wo_mv == NULL){ //NO select from non-columns, all are MV cols
		rel = mvPropRels[0].mvrel; 
		for (i = 1; i < nnode; i++){
			rel = join_two_rels(c, rel,  mvPropRels[0].atblname,  mvPropRels[0].asubjcolname, mvPropRels[i].mvrel,
					mvPropRels[i].atblname,  mvPropRels[i].asubjcolname);
		}
	}
	else{
		for (i = 0; i < nnode; i++){
			//sql_rel *tmprel = NULL; 
			if (mvPropRels[i].mvrel != NULL){
				//str subjcolname = "subject"; 
				if (rel == NULL){
					rel = join_two_rels(c, rel_wo_mv, atblname, asubjcolname, mvPropRels[i].mvrel,
							 mvPropRels[i].atblname,  mvPropRels[i].asubjcolname);
					//tmprel = rel_copy(c->sa, rel); 
				}
				else {
					
					rel = join_two_rels(c, rel, atblname, asubjcolname, mvPropRels[i].mvrel, 
							mvPropRels[i].atblname,  mvPropRels[i].asubjcolname); 
					//tmprel = rel_copy(c->sa, rel); 
				}
			}

		}
	}

	return rel; 

}



/*
 * Create exps for optional set of columns
 * e.g., base_column_exps  [s1.p1, s1.p2, s1.p3]
 * Then, the opt_exps will be
 * sys.ifthenelse(
 * 	sys.ifthenelse(
 * 		sys.isnull(
 * 		sys.or(sys.isnull(s1.p1),
 * 			sys.or(sys.isnull(s1.p2), sys.isnull(s1.p3))
 * 		)),
 *		boolean "false",
 * 		sys.or(sys.isnull(s1.p1),
 * 			sys.or(sys.isnull(s1.p2), sys.isnull(s1.p3))
 * 		)
 * 	),
 * 	NULL, 
 * 	s1.p1
 * )
 *
 * Look complicated :)
 *
 * If it is a set of required columns, then put sys.isnotnull()
 *
 * NOTE THAT as set of base columns can look like s1.s, s1.o, s2.s, s2.o,...
 * we only put the condition on o, while keeping s as original
 *
 * */

static 
list *create_optional_exps(mvc *sql, list *base_column_exps, int isOptionalGroup, int contain_mv_col){
	list *opt_exps = NULL ;
	list *req_exps = NULL; 
	sql_exp *or_exp = NULL; 
	node *en = NULL; 
	sql_allocator *sa = sql->sa; 

	list *only_o_exps = NULL; //keeping only o
	only_o_exps = new_exp_list(sa); 

	for (en = base_column_exps->h; en; en = en->next){
		sql_exp *o_exp = (sql_exp *) en->data;
		assert(o_exp->type == e_column);
		if (strcmp(o_exp->name, "o") == 0){
			sql_exp *tmp_o_exp =  exp_copy(sa, o_exp); 
			append(only_o_exps, tmp_o_exp);  
		}
	}

	if (isOptionalGroup){
		opt_exps = new_exp_list(sa); 
		if (contain_mv_col){
			printf("Do nothing for group of optional prop containing mv col \n");
			return base_column_exps; 
		} else {
			node *first_node = only_o_exps->h;
			sql_exp *first_exp = (sql_exp *) first_node->data;
			sql_exp *first_isnull_exp = exp_isnull(sql, first_exp); 

			if (first_node->next){
				for (en = first_node->next; en; en = en->next){
					sql_exp *tmpexp = (sql_exp *) en->data;
					sql_exp *tmp_isnull_exp = NULL; 
					assert(tmpexp->type == e_column); 
					tmp_isnull_exp = exp_isnull(sql, tmpexp); 

					if (or_exp == NULL){
						or_exp = rel_binop_(sql, first_isnull_exp, tmp_isnull_exp, NULL, "or", card_value);
					} else {
						or_exp = rel_binop_(sql, or_exp, tmp_isnull_exp, NULL, "or", card_value);
					}
				}
			}

			if (or_exp == NULL){
				or_exp = exp_copy(sa, first_isnull_exp); 
			}

			//Replace each column by ifthenelse
			//e.g.: col1 = ifthenelse ( or (col1 == null, col2 == null, col3 ==null), null, col1)
			for (en =base_column_exps->h; en; en = en->next){	
				sql_exp *tmpexp = (sql_exp *) en->data;
				sql_exp *exp_null = NULL; 

				assert(tmpexp->type == e_column);
				assert(or_exp != NULL); 	

				if (strcmp(tmpexp->name, "o") == 0){
					sql_exp *if_exp = NULL; 
					sql_exp *res = exp_copy(sa, tmpexp); 
					exp_null = exp_atom(sa, atom_general(sa, exp_subtype(tmpexp), NULL));
					
					if_exp = rel_nop_(sql, or_exp, exp_null, res, NULL, NULL, "ifthenelse", card_value);	
					
					assert (if_exp != NULL); 

					//Set the name
					exp_setname(sa, if_exp, tmpexp->rname, tmpexp->name);


					append(opt_exps, if_exp); 
				} else {
					sql_exp *s_exp = NULL;
					assert (strcmp(tmpexp->name, "s") == 0); 

					s_exp = exp_copy(sa, tmpexp); 

					append(opt_exps, s_exp); 
				}

			}

		}

		return opt_exps;	
	} else {
		req_exps = new_exp_list(sa); 
		for (en = base_column_exps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data;
			sql_exp *r_exp = exp_copy(sa, tmpexp); 
			
			append(req_exps, r_exp);
		}

		return req_exps; 
	}

}

static
void append_sp_opt_proj_exps(sql_allocator *sa, list *opt_col_exps, list *sp_opt_proj_exps){
	node *en; 
	for (en = opt_col_exps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data;
		sql_exp *proj_exp = exp_copy(sa, tmpexp);
		append(sp_opt_proj_exps, proj_exp); 
	}
}

/*
 * Input: 
 * - A sub-join graph (jsg) that all nodes are connected by using inner join
 * - The table (tId) that the node belongs to has been identified 
 *   (The table corresponding to the star pattern is known)
 * 
 * Output: 
 *  - A select from a relational table with list of columns. Or a join between
 *  select from a table and mv_table if there is mv prop. 
 *  - sp_prj_exps stores all the columns should be selected in the "original order" 
 * */
static
sql_rel* transform_inner_join_subjg (mvc *c, jgraph *jg, int tId, int *jsg, int nnode, list *sp_prj_exps, list *sp_opt_proj_exps, int *is_contain_mv, int isOptionalGroup, int *contain_missing_prop){

	sql_rel *rel = NULL;
	str tblname; 
	oid tblnameoid;
	str atblname = NULL; 		//alias for table of sp
	str asubjcolname = NULL; 	//alias for subject column of sp table
	char tmp[50]; 
	list *trans_select_exps = NULL; 	//Store the exps in op_select
	list *trans_table_exps = NULL; 		//Store the list of column for basetable in op_select
	sql_rel *rel_basetbl = NULL; 
	sql_rel *rel_wo_mv = NULL;
	sql_allocator *sa = c->sa;
	int num_mv_col = 0;
	int i; 
	int has_nonMV_col = 0; 
	int missingcol = 0; 	//[Happen only with optional group] a column is missing
	
	list *base_column_exps = NULL; 
	list *opt_exps = NULL; 

	mvPropRel *mvPropRels = init_mvPropRelSet(nnode); 

	num_mv_col = 0;

	trans_select_exps = new_exp_list(sa);
	trans_table_exps = new_exp_list(sa); 
	base_column_exps = new_exp_list(sa); 
	opt_exps = new_exp_list(sa); 

	printf("Get real expressions from tableId %d\n", tId);

	tblnameoid = global_csset->items[tId]->tblname;

	tblname = (str) GDKmalloc(sizeof(char) * 100); 

	getTblSQLname(tblname, tId, 0,  tblnameoid, global_mapi, global_mbat);

	printf("  [Name of the table  %s]", tblname);  
	
	
	for (i = 0; i < nnode; i++){
		jgnode *tmpnode = jg->lstnodes[jsg[i]];
		sql_rel *tmprel = (sql_rel*) (tmpnode->data);
		int colIdx; 
		int isMVcol = 0; 
		oid tmpPropId;

		assert(tmprel->op == op_select);
		assert(((sql_rel*)tmprel->l)->op == op_basetable); 
		
		tmpPropId = tmpnode->poid; 

		assert(tmpPropId != BUN_NONE); 
	
		colIdx = getColIdx_from_oid(tId, global_csset, tmpPropId);

		//If the column is not there, it can only happen for
		//optional group
		if (colIdx == -1) {
			assert(isOptionalGroup == 1); 
			missingcol = 1; 
			has_nonMV_col = 1; 
			isMVcol = 0; 
		}

		if (missingcol == 1){
			create_null_exps(c, tmprel, trans_table_exps, opt_exps, sp_prj_exps, tblname); 
		}
		else {
			//Check whether the column is multi-valued prop
			isMVcol = isMVCol(tId, colIdx, global_csset);

			if (isMVcol == 0){
				tranforms_exps(c, tmprel, trans_select_exps, trans_table_exps, tblname, colIdx, tmpPropId, &atblname, &asubjcolname, sp_prj_exps, base_column_exps, isOptionalGroup); 
				has_nonMV_col=1; 
			}
			else{
				printf("Table %d, column %d is multi-valued prop\n", tId, colIdx);
				assert (mvPropRels[i].mvrel == NULL); 
				tranforms_mvprop_exps(c, tmprel, &(mvPropRels[i]), tId, tblnameoid, colIdx, tmpPropId, isMVcol, sp_prj_exps, base_column_exps);
				num_mv_col++;

				//rel_print(c, mvPropRels[i].mvrel, 0);
				//rel_print(c, mvPropRels[i].mvrel, 0);
				//rel_print(c, mvPropRels[i].mvrel, 0);
				//rel_print(c, mvPropRels[i].mvrel, 0);
			}
		}
	}
	


	sprintf(tmp, "[Real Pattern] after grouping: "); 
	exps_print_ext(c, trans_select_exps, 0, tmp);
	sprintf(tmp, "  Base table expression: \n"); 
	exps_print_ext(c, trans_table_exps, 0, tmp);	

	rel_basetbl = rel_basetable(c, get_rdf_table(c,tblname), tblname); 

	rel_basetbl->exps = trans_table_exps;
	
	if (has_nonMV_col) rel_wo_mv = rel_select_copy(c->sa, rel_basetbl, trans_select_exps); 

	if (missingcol == 1){ //Return the rel with null in the column list
		*contain_missing_prop = 1; 
		*is_contain_mv = 0; 
	}
	
	if (num_mv_col > 0){	//missingcol == 0
		*is_contain_mv = 1; 
		rel = connect_sp_select_and_mv_prop(c, rel_wo_mv, mvPropRels, tblname, atblname, asubjcolname, nnode); 

		opt_exps = create_optional_exps(c, base_column_exps, isOptionalGroup, 1);
	}
	else{
		*is_contain_mv = 0;
		rel = rel_wo_mv; 
		
		if (missingcol == 0){ //in case missingcol == 1, opt_exps has been created
			opt_exps = create_optional_exps(c, base_column_exps, isOptionalGroup, 0);
		}
			
		printf("OPTIONAL Expressions\n");
		exps_print_ext(c, opt_exps, 0, NULL); 
	}


	append_sp_opt_proj_exps(c->sa, opt_exps, sp_opt_proj_exps);
	//rel_print(c, rel, 0); 
	GDKfree(tblname); 

	//TODO: Handle other cases. By now, we only handle 
	//the case where each sql_rel is a op_select. 

	list_destroy(trans_select_exps);
	list_destroy(base_column_exps); 

	if (0) free_mvPropRelSet(mvPropRels, nnode);

	return rel; 

}

#if HANDLING_EXCEPTION
static 
void get_removed_tid_exps(mvc *c, list *trans_base_exps, sql_rel *r){
	list *tmpexps = NULL; 
	node *en;

	assert (r->op == op_basetable);
	tmpexps = r->exps; 

	assert(tmpexps); 

	for (en = tmpexps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data;
		assert (tmpexp->type == e_column); 

		if (strcmp(tmpexp->name, "%TID%") == 0){
			continue; 
		} else {
			sql_exp *m_exp = exp_copy(c->sa, tmpexp);
			//append this exp to list
			append(trans_base_exps, m_exp);
		}
	}
}

/*
 * Input: 
 * - A sub-join graph (jsg) that all nodes are connected by using inner join
 * - The table (tId) that the node belongs to has been identified 
 *   (The table corresponding to the star pattern is known)
 * 
 * Output: 
 *  - A select from a relational table with list of columns. Or a join between
 *  select from a table and mv_table if there is mv prop. 
 *  - sp_prj_exps stores all the columns should be selected in the "original order" 
 * */

static
sql_rel* build_rdfexception (mvc *c, int tId, jgraph *jg, list *union_rdfscan_exps, int nijgroup, int **ijgroup, int *nnodes_per_ijgroup, spProps *spprops){

	sql_rel *rel_rdfscan = NULL;
	str tblname; 
	char dummy_tblname[100]; 
	oid tblnameoid;
	sql_rel *rel_basetbl = NULL; 
	str dup_tblname = NULL;
	int gr, i; 

	sql_table *tbl; 
	list *trans_select_exps = NULL; 
	list *trans_base_exps = NULL; 

	//Constraints for o values
	oid *los; 
	oid *his; 
	
	if (tId != -1){
		printf("Get real expressions from tableId %d\n", tId);

		tblnameoid = global_csset->items[tId]->tblname;

		tblname = (str) GDKmalloc(sizeof(char) * 50); 

		getTblSQLname(tblname, tId, 0,  tblnameoid, global_mapi, global_mbat);
	
		sprintf(dummy_tblname,"dummy_%s",tblname); 

		printf("  [Name of the table  %s]", tblname);  

		dup_tblname  = sa_strdup(c->sa, dummy_tblname); 

	} else {
		sprintf(dummy_tblname,"dummy_tbl"); 
		dup_tblname  = sa_strdup(c->sa, dummy_tblname); 
	}

	
	tbl = create_dummy_table(c, dup_tblname, union_rdfscan_exps, spprops->num);
	rel_basetbl = rel_basetable(c, tbl, dup_tblname); 

	trans_base_exps = new_exp_list(c->sa); 

	get_removed_tid_exps(c, trans_base_exps, rel_basetbl);

	rel_basetbl->exps = trans_base_exps; 


	printf("\nDUMMY TABLE\n"); 

	_rel_print(c, rel_basetbl);

	
	if(0){
	trans_select_exps = new_exp_list(c->sa);
	for (gr = 0; gr < nijgroup; gr++){
		for (i = 0; i < nnodes_per_ijgroup[gr]; i++){
			int nodeid = ijgroup[gr][i];
			jgnode *tmpnode = jg->lstnodes[nodeid];
			sql_rel *tmprel = (sql_rel*) (tmpnode->data);
			list *tmpexps = NULL; 

			assert(tmprel->op == op_select);
			assert(((sql_rel*)tmprel->l)->op == op_basetable); 

			tmpexps = tmprel->exps;

			if (tmpexps){
				get_transform_dummy_select_exps(c, tmpexps, trans_select_exps, dup_tblname); 
			}
		}
	}
	
	exps_print_ext(c, trans_select_exps, 0, "[RDFexception] select exprs: ");
	}
	
	los = (oid *) malloc(sizeof(oid) * spprops->num);
	his = (oid *) malloc(sizeof(oid) * spprops->num);
	for (i = 0; i < spprops->num; i++){
		los[i] = spprops->lst_o_constraints[i].low;
		his[i] = spprops->lst_o_constraints[i].hi;
	}
 
	rel_rdfscan = rel_rdfscan_func(c, tbl, spprops->num, nnodes_per_ijgroup[0], spprops->lstPropIds, los, his, spprops->exps, spprops->lstAlias); 
	//rel_rdfscan = rel_rdfscan_func(c, tbl, spprops->num, nnodes_per_ijgroup[0], spprops->lstPropIds, los, his, NULL); 
	
	printf("\nRDFSCAN \n");
	_rel_print(c, rel_rdfscan);
	
	return rel_rdfscan; 

}

#endif

#if 0
/*
 * Input: 
 * - A sub-join graph (jsg) that all nodes are connected by using inner join
 * - The table (tId) that the node belongs to has been identified 
 *   (The table corresponding to the star pattern is known)
 * 
 * Output: 
 *  - A select from a relational table with list of columns. Or a join between
 *  select from a table and mv_table if there is mv prop. 
 *  - sp_prj_exps stores all the columns should be selected in the "original order" 
 * */
static
sql_rel* build_rdfscan (mvc *c, jgraph *jg, int tId, int ncol, int nijgroup, int **ijgroup, int *nnodes_per_ijgroup){

	sql_rel *rel_rdfscan = NULL;
	str tblname; 
	oid tblnameoid;
	str atblname = NULL; 		//alias for table of sp
	str asubjcolname = NULL; 	//alias for subject column of sp table
	char tmp[50]; 
	list *trans_select_exps = NULL; 	//Store the exps in op_select
	list *trans_table_exps = NULL; 		//Store the list of column for basetable in op_select
	sql_rel *rel_basetbl = NULL; 
	sql_allocator *sa = c->sa;
	int num_mv_col = 0;
	int i, gr; 
	int num_nonMV_col = 0; 
	rdf_rel_prop *r_r_prop = NULL; 

	num_mv_col = 0;

	trans_select_exps = new_exp_list(sa);
	trans_table_exps = new_exp_list(sa); 
	r_r_prop = init_rdf_rel_prop(ncol, nijgroup, nnodes_per_ijgroup);

	printf("Get real expressions from tableId %d\n", tId);

	tblnameoid = global_csset->items[tId]->tblname;

	tblname = (str) GDKmalloc(sizeof(char) * 100); 

	getTblSQLname(tblname, tId, 0,  tblnameoid, global_mapi, global_mbat);

	printf("  [Name of the table  %s]", tblname);  
	
	for (gr = 0; gr < nijgroup; gr++){
		for (i = 0; i < nnodes_per_ijgroup[gr]; i++){
			int nodeid = ijgroup[gr][i];
			jgnode *tmpnode = jg->lstnodes[nodeid];
			sql_rel *tmprel = (sql_rel*) (tmpnode->data);
			int colIdx; 
			int isMVcol = 0; 
			oid tmpPropId;

			assert(tmprel->op == op_select);
			assert(((sql_rel*)tmprel->l)->op == op_basetable); 

			tmpPropId = tmpnode->poid; 

			colIdx = getColIdx_from_oid(tId, global_csset, tmpPropId);

			//Check whether the column is multi-valued prop
			isMVcol = isMVCol(tId, colIdx, global_csset);

			//Only for RDFscan, otherwise we need to handle Multi-valued prop
			tranforms_exps(c, tmprel, trans_select_exps, trans_table_exps, tblname, colIdx, tmpPropId, &atblname, &asubjcolname, NULL, NULL, 0); 

			if (isMVcol == 0){
				num_nonMV_col++; 
			}
			else{
				num_mv_col++;
			}
		}
	}
	
	sprintf(tmp, "[RDFscan] select exprs: "); 
	exps_print_ext(c, trans_select_exps, 0, tmp);
	sprintf(tmp, "          base table expression: \n"); 
	exps_print_ext(c, trans_table_exps, 0, tmp);	



	rel_basetbl = rel_basetable(c, get_rdf_table(c,tblname), tblname); 

	rel_basetbl->exps = trans_table_exps;

	if (num_mv_col > 0) r_r_prop->containMV = 1; 
	
	rel_rdfscan = rel_rdfscan_create(c->sa, rel_basetbl, trans_select_exps, r_r_prop); 

	
	list_destroy(trans_select_exps);

	return rel_rdfscan; 

}

#endif



static
void tranforms_join_exps(mvc *c, sql_rel *r, list *sp_edge_exps, int is_outer_join, int need_add_exps){

	node *en; 
	list *tmp_exps = NULL; 

	tmp_exps = r->exps;
	for (en = tmp_exps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data;
		if(tmpexp->type == e_cmp){
			//Check and put to the final
			//experssion list if the exp is not the comparison 
			//between .s = .s since this compare should belong to the
			//same graph pattern
			sql_exp *l; 
			sql_exp *r; 
			JP tmpjp; 

			l = tmpexp->l; 
			r = tmpexp->r; 

			//For case of q5 bsbm
			if (l->type != e_column && r->type != e_column && need_add_exps == 1){
				sql_exp *m_exp = exp_copy(c->sa, tmpexp);	
				append(sp_edge_exps, m_exp);
				continue; 
			}

			assert(l->type == e_column);
			assert(r->type == e_column); 

			get_jp(l->name, r->name, &tmpjp); 
			
			if (tmpjp != JP_S || is_outer_join){
				sql_exp *m_exp = exp_copy(c->sa, tmpexp);
				//append this exp to list
				append(sp_edge_exps, m_exp);
			}
		}
		else{	// rarely happen, for example, [ tinyint "1" ]
			sql_exp *m_exp = exp_copy(c->sa, tmpexp);
			//append this exp to list
			append(sp_edge_exps, m_exp);
		}
	
	}
}


static
void build_exps_from_join_jgedge(mvc *c, jgedge *edge, list *sp_edge_exps, operator_type *op, int is_combine_ij){
	sql_rel *tmpjoin = NULL; 
	char tmp[50];

	tmpjoin = (sql_rel*) edge->data; 

	if (is_combine_ij && tmpjoin->op == op_join) assert(0); 

	sprintf(tmp, "Expression of edge [%d,%d] \n", edge->from, edge->to);
	exps_print_ext(c, tmpjoin->exps, 0, tmp);
	if (is_combine_ij) tranforms_join_exps(c, tmpjoin,sp_edge_exps, 1, 0);
	else tranforms_join_exps(c, tmpjoin,sp_edge_exps, 0, 0);
	exps_print_ext(c, sp_edge_exps, 0, "Update expression:");

	if (tmpjoin->op != op_join) //May be op_left, op_right	
		*op = tmpjoin->op;
		//TODO: Need to recheck this since not all edges 
		//between two pattern can be outter joins.

}

/*
 * Create edges between star pattern 
 * (or between inner join groups) in order to replace
 * edges connecting each pair of nodes coming from 
 * different star pattern (or inner join group).
 * E.g., starpattern0: Node 0, 1
 *       starpattern1: Node 2, 3, 4
 *       starpattern2: Node 5,6 
 * An edge between sp0 and sp1 will be created by combining edge between
 * 0,2  0,3   0,4    1,2   1,3   1,4. This edge is an join where left is sp0, right is sp1 and
 * expression is the combination of expression from these edges.
 *
 * Input: is_combine_ij: Are these two groups inner join groups
 * */
static
sql_rel *_group_edge_between_two_groups(mvc *c, jgraph *jg, int pId, int *group1, int nnode1, 
					int *group2, int nnode2, sql_rel *left, sql_rel *right, int is_combine_ij){
	int i, j; 
	sql_rel *rel_edge = NULL;
	list *sp_edge_exps = NULL;
	operator_type op = op_join;

	assert(left);
	assert(right); 

	sp_edge_exps = new_exp_list(c->sa);


	printf("Create edge between pattern %d and %d\n", pId, (pId + 1)); 
	for (i = 0; i < nnode1; i++){
		for (j = 0; j < nnode2; j++){
			//Get the edge between group1[i], group2[j]
			jgedge *edge = get_edge_jp(jg, group1[i], group2[j]);
			if (edge) {	
				build_exps_from_join_jgedge(c, edge, sp_edge_exps, &op, is_combine_ij);	
			} else {
				printf("No edge between  [%d,%d] \n", group1[i], group2[j]);
			}


		}
	}
	
	printf("Expression for join between pattern %d and %d\n", pId, (pId + 1));
	exps_print_ext(c, sp_edge_exps, 0, "Exp:");

	
	rel_edge = rdf_rel_join(c->sa, left, right, sp_edge_exps, op);

	return rel_edge; 
}

static 
sql_rel *group_pattern_by_cross_edge(mvc *c, sql_rel *left, sql_rel *right, jgedge *edge){
	sql_rel *rel_edge = NULL;
	list *sp_edge_exps = NULL;
	operator_type op = op_join;

	assert(left);
	assert(right); 
	
	sp_edge_exps = new_exp_list(c->sa);
	printf("Build rel for cross edge from %d to %d\n", edge->from, edge->to);

	build_exps_from_join_jgedge(c, edge, sp_edge_exps, &op, 0);

	printf("Expression for this cross edge:");
	exps_print_ext(c, sp_edge_exps, 0, "Exp:");

	rel_edge = rdf_rel_join(c->sa, left, right, sp_edge_exps, op);

	return rel_edge; 

}

/*
 * The algorithm for building rels from cross edge is as following:
 * - Init [Star Pattern] to [CrossEdge] mapping (-1 by default meaning
 *   that this star pattern has not been included into any cross edge rel yet)
 * - We loop over all the cross edge. 
 *   For each cross edge, we join/outerjoin two sp's connected by this edge.
 * - To prevent different cross edges applied to the same pair of sp's
 *   we update group represent ID after joining them. If the sp's already
 *   share the same group represent ID, then do not apply that cross edge.
 *   Group represent ID is chosen as the smallest sp Id.
 * */
static 
void build_all_rels_from_cross_edges(mvc *c, int num_cross_edges, jgedge **lst_cross_edges, 
	int *cr_ed_orders, jgraph *jg, sql_rel **lst_cross_edge_rels, sql_rel **lstRels, int numsp, int *last_cre){
	
	int i; 
	int e_id; 
	int *sp_cre_map = NULL; //Star pattern - cross edge rel mapping
	int *gr_rep_id = NULL; //Group represent ID.

	sp_cre_map = (int *) malloc(sizeof(int) * numsp); 
	gr_rep_id = (int *) malloc(sizeof(int) * numsp); 

	for (i = 0; i < numsp; i++){
		sp_cre_map[i] = -1; 
		gr_rep_id[i] = i; 
	}
	for (i = 0; i < num_cross_edges; i++){
		lst_cross_edge_rels[i] = NULL; 
	}

	for (i = 0; i < num_cross_edges; i++){
		int spId1, spId2; 
		jgnode *n1, *n2; 
		jgedge *edge;
		e_id = cr_ed_orders[i]; 
		edge = lst_cross_edges[e_id]; 

		n1 = jg->lstnodes[edge->from]; 
		n2 = jg->lstnodes[edge->to];
		spId1 = n1->patternId;
		spId2 = n2->patternId; 

		if (gr_rep_id[spId1] == gr_rep_id[spId2]){
			if (edge->need_add_exps == 1){ //Handling special case for q5 bsbm
				int cre_id = sp_cre_map[spId1];
				assert(sp_cre_map[spId1] == sp_cre_map[spId2]); 
				tranforms_join_exps(c, edge->data, lst_cross_edge_rels[cre_id]->exps, 0, 1);
			} else {
				printf("Already belong to the same group. Do not apply cross edges [%d,%d]\n",edge->from,edge->to);
			}
			continue;
		}


		if (sp_cre_map[spId1] == -1 && sp_cre_map[spId2] == -1){
			lst_cross_edge_rels[i] = group_pattern_by_cross_edge(c, lstRels[spId1], lstRels[spId2], edge); 
		}
		else if (sp_cre_map[spId1] == -1 && sp_cre_map[spId2] != -1){
			int cre_id2 =  sp_cre_map[spId2];
			lst_cross_edge_rels[i] = group_pattern_by_cross_edge(c, lstRels[spId1], lst_cross_edge_rels[cre_id2], edge);
		}
		else if (sp_cre_map[spId1] != -1 && sp_cre_map[spId2] == -1){
			 int cre_id1 =  sp_cre_map[spId1];
			 lst_cross_edge_rels[i] = group_pattern_by_cross_edge(c, lst_cross_edge_rels[cre_id1], lstRels[spId2], edge);
		}
		else{	//Both the star patterns belong to some cross edge rels
			int cre_id1 =  sp_cre_map[spId1];
			int cre_id2 =  sp_cre_map[spId2];

			if (cre_id1 == cre_id2) {
				printf("Already connected. Do not apply cross edges [%d,%d]\n",edge->from,edge->to);
				continue;  //These patterns are connected already
			}
			else
				lst_cross_edge_rels[i] = group_pattern_by_cross_edge(c, lst_cross_edge_rels[cre_id1], lst_cross_edge_rels[cre_id2], edge);
		}

		//Update sp_cre_map
		sp_cre_map[spId1] = i;
		sp_cre_map[spId2] = i;
		
		if (gr_rep_id[spId1] > gr_rep_id[spId2]){
			gr_rep_id[spId1] = gr_rep_id[spId2];
		} else
			gr_rep_id[spId2] = gr_rep_id[spId1];


		*last_cre = i;
	}

	free(sp_cre_map);
	free(gr_rep_id); 
}


/*
 * Get inner-join groups from 
 * nodes in a star pattern
 * */

static
int** get_inner_join_groups_in_sp_group(jgraph *jg, int* group, int nnode, int *nijgroup, int **nnodes_per_ijgroup){

	int max_ijId = 0;	//max inner join pattern Id
	int **ijgroup = NULL;
	int *idx;
	int i, j; 

	for (i = 0; i < nnode; i++){
		if (max_ijId < (jg->lstnodes[group[i]])->ijpatternId)
			max_ijId = (jg->lstnodes[group[i]])->ijpatternId;
	}

	ijgroup = (int **) malloc(sizeof (int *) * (max_ijId + 1));
	*nnodes_per_ijgroup = (int *) malloc(sizeof(int) * (max_ijId + 1));
	idx = (int *) malloc(sizeof(int) * (max_ijId + 1));
	
	for (i = 0; i < (max_ijId + 1); i++){
		(*nnodes_per_ijgroup)[i] = 0; 
		idx[i] = 0;

	}

	for (i = 0; i < nnode; i++){
		jgnode *node = jg->lstnodes[group[i]];
		(*nnodes_per_ijgroup)[node->ijpatternId]++;

	}
	for (i = 0; i < (max_ijId + 1); i++){
		ijgroup[i] = (int *) malloc(sizeof(int) * (*nnodes_per_ijgroup)[i]); 		
	}

	for (i = 0; i < nnode; i++){
		int groupId = (jg->lstnodes[group[i]])->ijpatternId;
		ijgroup[groupId][idx[groupId]] = group[i];
		idx[groupId]++;
	}
	
	*nijgroup = (max_ijId + 1); 

	free(idx);


	printf("Number of inner join group is: %d\n", *nijgroup);

	for (i = 0; i < *nijgroup; i++){
		printf("Group %d: ", i);
		for (j = 0; j < (*nnodes_per_ijgroup)[i]; j++){
			printf(" %d",ijgroup[i][j]);
		}
		printf("\n"); 
		
	}

	return ijgroup; 
}

static
void free_inner_join_groups(int **ijgroup, int nijgroup, int *nnodes_per_ijgroup){
	int i; 
	//Free
	for (i = 0; i < nijgroup; i++){
		free(ijgroup[i]);
	}
	free(ijgroup); 
	free(nnodes_per_ijgroup);
}
/*
 * Get union expression when there are multiple matching tables
 * */

/*
Recursively go throuhg all op_select  cho mot rel
join
   join
     join
       join
          select
	  select
       select
   select
select
*/   

static
void _append_union_expr(mvc *c, sql_rel *sel_rel, list *union_exps){

	list *tmpexps = NULL; 
	sql_allocator *sa = c->sa; 
	sql_rel *tbl_rel = NULL;

	assert(sel_rel->op == op_select || sel_rel->op == op_rdfscan); 
	assert (((sql_rel *)sel_rel->l)->op == op_basetable);
	tbl_rel = (sql_rel *)sel_rel->l;
	tmpexps = tbl_rel->exps; 

	if (tmpexps){
		node *en;
	
		for (en = tmpexps->h; en; en = en->next){
			sql_exp *tmpexp = (sql_exp *) en->data;
			sql_exp *m_exp = exp_copy(sa, tmpexp);

			assert(tmpexp->type == e_column); 
			//append this exp to list
			append(union_exps, m_exp);

		}

	}

}


static
void get_union_expr(mvc *c, sql_rel *r, list *union_exps){
	
	sql_rel *tmp_rel = NULL;
	
	tmp_rel = r; 
	//Because, the select op may be included 
	//inside an join for the case of mvprop
	if (tmp_rel->op == op_project){ //For the case of having more projection on top of the rel
		get_union_expr(c, tmp_rel->l, union_exps);	
	}
	else if (tmp_rel->op != op_select){
		assert(tmp_rel->op == op_join || tmp_rel->op == op_left);
		get_union_expr(c, tmp_rel->l, union_exps);
		get_union_expr(c, tmp_rel->r, union_exps);
	}
	else{
		_append_union_expr(c, tmp_rel, union_exps); 
	}


}

static 
void _generate_ijpatternId(jgraph *jg, jgnode *node, int ijpId){
	jgedge *tmpedge;
	tmpedge = node->first;
	while (tmpedge != NULL){
		if (tmpedge->jp == JP_S){
			jgnode *tonode = jg->lstnodes[tmpedge->to];
			assert(tonode->patternId == node->patternId); 
			if (tmpedge->op == op_join){
				if (tonode->ijpatternId == -1) tonode->ijpatternId = ijpId;
				else{
					assert (tonode->ijpatternId == ijpId); 
				}
			}
		}
		tmpedge = tmpedge->next;
	}
}
/*
 * Generate ijpatternId for each node in a
 * star pattern.
 * Algorithm: 
 * All nodes connected by inner join belong to the same ijgroup
 * */
static
void generate_ijpatternId(jgraph *jg, int *group, int nnode){
	int i; 
	int ijpId = -1; 
	for (i = 0; i < nnode; i++){
		int idx = group[i]; 
		jgnode *node = jg->lstnodes[idx]; 
		if (node->ijpatternId == -1){
			ijpId++; 
			node->ijpatternId = ijpId; 
			_generate_ijpatternId(jg, node, ijpId); 
		}
	
	}
	
	printf("Number of ijgroup is: %d\n", (ijpId + 1)); 
}

static 
sql_rel* _group_star_pattern_for_single_table(mvc *c, jgraph *jg, 
		int **ijgroup, int nijgroup, int *nnodes_per_ijgroup, 
		int tId, list **sp_proj_exps, list **sp_opt_proj_exps, 
		int *contain_mv_col){

	int i; 

	sql_rel **ijrels = NULL;    //rel for inner join groups
	sql_rel **edge_ijrels = NULL;  //sql_rel connecting each pair of ijrels
	sql_rel *tbl_m_rel = NULL; 
	int is_contain_mv = 0; 
	int *ingroup_contain_mv = NULL; sql_rel *tmprel_rdfscan = NULL; 
	int *contain_missing_prop = NULL; 

	*sp_proj_exps = new_exp_list(c->sa);
	*sp_opt_proj_exps = new_exp_list(c->sa);




	ijrels = (sql_rel **) malloc(sizeof(sql_rel*) * nijgroup);
	edge_ijrels = (sql_rel **) malloc(sizeof(sql_rel*) * (nijgroup - 1));
	ingroup_contain_mv  = (int *) malloc(sizeof(int) * nijgroup); 
	contain_missing_prop = (int *) malloc(sizeof(int) * nijgroup); 
	
	for (i = 0; i < nijgroup; i++){
		int isOptionalGroup = 0;

		ingroup_contain_mv[i] = 0;
		contain_missing_prop[i] = 0; 

		if (i > 0) isOptionalGroup = 1;
		ingroup_contain_mv[i] = 0;
		ijrels[i] = transform_inner_join_subjg (c, jg, tId, ijgroup[i], nnodes_per_ijgroup[i], *sp_proj_exps, *sp_opt_proj_exps, &(ingroup_contain_mv[i]), isOptionalGroup, &(contain_missing_prop[i]));
		if (ingroup_contain_mv[i]){
			 is_contain_mv = 1;
		}

	}
	
	*contain_mv_col = is_contain_mv;
	if (*contain_mv_col) printf("Contain MV cols \n"); 
	printf("Original Projection of all columns (w/o considering mv col): \n"); 
	exps_print_ext(c, *sp_proj_exps, 0, NULL); 

	printf("Expression handling OPTIONAL for table matching %d\n",tId);
	exps_print_ext(c, *sp_opt_proj_exps, 0, NULL); 

	if (nijgroup > 1){
		#if (APPLY_OPTIMIZATION_FOR_OPTIONAL == 0) 	
		//Always use left outer join for connecting ijgroup
		printf("APPLY_OPTIMIZATION_FOR_OPTIONAL \n"); 
		//Connect these ijrels by outer joins
		for (i = 0; i < (nijgroup - 1); i++){
			edge_ijrels[i] = _group_edge_between_two_groups(c, jg, i, ijgroup[i], nnodes_per_ijgroup[i], 
						ijgroup[i+1], nnodes_per_ijgroup[i+1], ijrels[i], ijrels[i+1], 1);
		}
		connect_groups(nijgroup, ijrels, edge_ijrels);

		tbl_m_rel = edge_ijrels[0];	

		#else
		//if the inner group does not have mv prop, then use the IFTHENELSE approach
		//First: Connect all non-mv prop groups into single rel (with the first rel)
		sql_rel *non_mv_rep_rel = NULL; //Represent all non-mv props. 
		int 	n_mv_groups = 1; 	//Number of groups with mv prop
						//Start from 1 (the first is the combination of 
						//all non-mv groups)
		int 	n_non_mv_groups = 0;
		int 	*old_idx_map = (int *) malloc(sizeof(int) * nijgroup); 

		sql_rel **newijrels = (sql_rel **) malloc(sizeof(sql_rel*) * nijgroup);

		non_mv_rep_rel = ijrels[0]; 
		for (i = 1; i < nijgroup; i++){
			if (ingroup_contain_mv[i] == 0){
				n_non_mv_groups++;
				non_mv_rep_rel =  rdf_rel_simple_combine_with_optional_cols(c->sa, non_mv_rep_rel, ijrels[i]);
			}
			else {
				old_idx_map[n_mv_groups] = i; 
				newijrels[n_mv_groups] = ijrels[i]; 
				n_mv_groups++; 
			}
		}

		newijrels[0] = non_mv_rep_rel; 
		old_idx_map[0] = 0; 	
		
		if (n_mv_groups > 1){
			//Then, connect with groups having mv props
			for (i = 0; i < (n_mv_groups - 1); i++){
				int id1 = old_idx_map[i];
				int id2 = old_idx_map[i+1];

				edge_ijrels[i] = _group_edge_between_two_groups(c, jg, i, ijgroup[id1], nnodes_per_ijgroup[id1],
							ijgroup[id2], nnodes_per_ijgroup[id2], newijrels[i], newijrels[i+1], 1);
			}

			connect_groups(n_mv_groups, newijrels, edge_ijrels);	
			tmprel_rdfscan = edge_ijrels[0];
		}
		else{
			tmprel_rdfscan = newijrels[0];
		}
		if (n_non_mv_groups > 0){	//Add IFTHENELSE project
			sql_rel *tmp_proj_rel = rel_project(c->sa, tmprel_rdfscan, *sp_opt_proj_exps);
			tbl_m_rel = tmp_proj_rel; 
		}
		else{
			tbl_m_rel = tmprel_rdfscan; 			
		}
		#endif
	}
	else{	//nijgroup = 1
		tbl_m_rel = ijrels[0]; 
	}



	free(edge_ijrels);
	free(ijrels);


	return tbl_m_rel; 
}

static
sql_rel* union_sp_from_all_matching_tbls(mvc *c, int num_match_tbl, int *contain_mv_col, sql_rel **tbl_m_rels, list **sp_proj_exps){
	
	sql_rel *rel = NULL; 
	#if USING_UNION_FOR_MULTIPLE_MATCH
	int tblIdx; 
	//using union
	sql_rel *tmprel = NULL;
	list *union_exps = NULL;
	union_exps = new_exp_list(c->sa);
	if (contain_mv_col[0]){		//cover by a project with original columns order
		tmprel = rel_project(c->sa, tbl_m_rels[0], sp_proj_exps[0]);	
	} else {
		tmprel = tbl_m_rels[0];
	}

	if (num_match_tbl > 1){
		//_rel_print(c, tbl_m_rels[0]);
		get_union_expr(c, tbl_m_rels[0], union_exps); 
		printf("Union expresion is: \n"); 
		exps_print_ext(c, union_exps, 0, "");
	}

	for (tblIdx = 1; tblIdx < num_match_tbl; tblIdx++){
		sql_rel *tmp_proj_rel = NULL; 
		sql_rel *tmprel2 = NULL; 
		list *tmpexps;

		if (contain_mv_col[tblIdx]){
			tmp_proj_rel = rel_project(c->sa, tbl_m_rels[tblIdx], sp_proj_exps[tblIdx]);
		}else{
			tmp_proj_rel = tbl_m_rels[tblIdx];
		}

		tmprel2 = rel_setop(c->sa, tmprel, tmp_proj_rel, op_union); 
		tmpexps = exps_copy(c->sa, union_exps);
		tmprel2->exps = tmpexps; 
		tmprel = tmprel2;
	}

	rel = tmprel; 

	
	#else
	assert(num_match_tbl == 1); 	//TODO: Handle the case of matching multiple table
	rel = tbl_m_rels[0]; 
	#endif

	return rel; 
}

static void
add_spprops_exps(mvc *c, list *retexps, list *exps){

	node *en;
	sql_allocator *sa = c->sa;

	for (en = exps->h; en; en = en->next){
		sql_exp *tmpexp = (sql_exp *) en->data; 
		sql_exp *e = (sql_exp *)tmpexp->l; 

		assert(tmpexp->type == e_cmp); //TODO: Handle other exps for op_select
		assert(e->type == e_convert); 

		e = e->l; 

		assert(e->type == e_column); 

		if (strcmp(e->name, "p") == 0){
			continue; 

		} else if (strcmp(e->name, "o") == 0){
			sql_exp *m_exp = exp_copy(sa, tmpexp);
			sql_exp *re = (sql_exp *)m_exp->r; 
			sql_exp *le = (sql_exp *)m_exp->l; 
			sql_exp *cole = NULL; 
			assert(le->type == e_convert); 

			cole = le->l; 

			if (re->type == e_atom){
				sql_exp *newre = get_atom_oid(c, re);
				m_exp->r = newre;
				assert(cole->type == e_column);
				m_exp->l = cole; 
			}
			

			//append this exp to list
			append(retexps, m_exp);

		} else if (strcmp(e->name, "s") == 0){
			sql_exp *m_exp = exp_copy(sa, tmpexp);
			//append this exp to list
			append(retexps, m_exp);
		}
	}
}

static
void update_RP_and_O_constraint(mvc *c, jgraph *jg, int *ijgroup, int nnode, spProps *spprops){
	int i, j; 
	
	
	for (i = 0; i < nnode; i++){
		int pidx; 
		jgnode *tmpnode = jg->lstnodes[ijgroup[i]];
		sql_rel *tmprel = (sql_rel*) (tmpnode->data);
		oid tmpPropId;

		assert(tmprel->op == op_select);
		assert(((sql_rel*)tmprel->l)->op == op_basetable); 
		
		tmpPropId = tmpnode->poid; 
		
		//Get index of this prop
		for (j = 0; j < spprops->num; j++){
			if (spprops->lstPropIds[j] == tmpPropId) break; 
		}
		pidx = j; 
		spprops->lstPOs[pidx] = REQUIRED;

		assert(j < spprops->num); 

		//Check for o_constraint

		get_o_constraint(c, &(spprops->lst_o_constraints[pidx]), tmprel->exps);
		add_spprops_exps(c, spprops->exps, tmprel->exps); 
		(void) c; 
	}

}

/*
 * Create a select sql_rel from a star pattern
 * */

static 	
sql_rel* _group_star_pattern(mvc *c, jgraph *jg, int *group, int nnode, int pId){
	sql_rel *rel = NULL, *rel_alltable = NULL; 
	int 	i, tblIdx; 
	char 	is_all_select = 1; 
	char 	is_only_basetable = 1;
	spProps *spprops = NULL; 
	int 	*tmptbId = NULL; 
	int 	num_match_tbl = 0;

	int 	**ijgroup = NULL; 
	int 	*nnodes_per_ijgroup = NULL; 
	int 	nijgroup = 0;
	
	#if	HANDLING_EXCEPTION
	sql_rel *rel_rdfscan = NULL;	//For handling exception
	list *union_rdfscan_exps = NULL; //Union for rel_rdfscan and other regular matching tables
	#endif

	//This transformed exps list contain exps list from op_select
	 //on the object

	printf("Group %d contain %d nodes: ", pId, nnode); 

	generate_ijpatternId(jg, group, nnode); 

	//Check whether all nodes are SELECT nodes 
	for (i = 0; i < nnode; i++){
		sql_rel *tmprel = (sql_rel*) (jg->lstnodes[group[i]]->data);
		printf(" %d ", group[i]); 
		//rel_print(c, tmprel, 0); 
		if (tmprel->op != op_select) is_all_select = 0; 	
		if (tmprel->op != op_basetable) is_only_basetable = 0;
	}
	printf("\n"); 

	if (nnode > 1) is_only_basetable = 0;
	
	//Convert to sql_rel of abstract table
	if (is_all_select){
		sql_rel **tbl_m_rels; 	//each rel (table matching rel) replaces all the triples matching with 
					//a specific table
		list **sp_proj_exps; 	//Store the simple project expressions for each star pattern w/o 
					//regarding the availability of multi-valued prop
		list **sp_opt_proj_exps = NULL; 	//Store the project experessions for each star pattern
					// with ifthenelse statement for optional keywords
		int *contain_mv_col = NULL; 			

		spprops = init_sp_props(c, nnode); 	

		for (i = 0; i < nnode; i++){
			jgnode *tmpnode = jg->lstnodes[group[i]]; 
			sql_rel *tmprel = (sql_rel*) (tmpnode->data);
			verify_rel(tmprel); 
			printf("Column %d name is %s\n", i, tmpnode->prop);
			add_props_and_subj_to_spprops(spprops, i, NAV, tmpnode); 		

		}

		ijgroup = get_inner_join_groups_in_sp_group(jg, group, nnode, &nijgroup, &nnodes_per_ijgroup);

		//TODO: Add a function update_Require_Optional_prop() 
		//to specify which prop is optional, which is required, and o_contrains
			
		update_RP_and_O_constraint(c, jg, ijgroup[0], nnodes_per_ijgroup[0], spprops); 

		print_spprops(spprops);
	
		get_matching_tbl_from_spprops(&tmptbId, spprops, &num_match_tbl);

		printf("Number of matching table is: %d\n", num_match_tbl);
		
		tbl_m_rels = (sql_rel **) malloc(sizeof(sql_rel *) * num_match_tbl);
		sp_proj_exps = (list **) malloc(sizeof(list *) * num_match_tbl); 
		contain_mv_col = (int *) malloc(sizeof(int) * num_match_tbl);
		
		sp_opt_proj_exps = (list **) malloc(sizeof(list *) * num_match_tbl);



		//num_match_tbl = 1; 
		
		printf("Grouping star pattern for each matching table of %d candidates\n",num_match_tbl);
		for (tblIdx = 0; tblIdx < num_match_tbl; tblIdx++){
			tbl_m_rels[tblIdx] = _group_star_pattern_for_single_table(c, jg, ijgroup, nijgroup, nnodes_per_ijgroup, tmptbId[tblIdx], &(sp_proj_exps[tblIdx]), &(sp_opt_proj_exps[tblIdx]), 
					&(contain_mv_col[tblIdx]));
			//Check 
			_rel_print(c, tbl_m_rels[tblIdx]);

			//_rel_print(c, tbl_m_rels[0]);

		}
		

		printf("Done grouping star pattern\n"); 

		//if (num_match_tbl > 0)_rel_print(c, tbl_m_rels[0]);

		if (num_match_tbl > 0){ 
			rel_alltable = union_sp_from_all_matching_tbls(c, num_match_tbl, contain_mv_col, tbl_m_rels, sp_proj_exps); 
	
			printf("RDF Regular Rel\n"); 
			printf("Number of ijgroups %d  -  Is contain MV %d\n", nijgroup, contain_mv_col[0]);
			exps_print_ext(c, sp_proj_exps[0], 0, "sp_proj_exps ==> "); 
			_rel_print(c, rel_alltable); 

			//Union with RDFscan
			/*
			union_rdfscan_exps = new_exp_list(c->sa);
			get_union_expr(c,tbl_m_rels[0] , union_rdfscan_exps);
			exps_print_ext(c, union_rdfscan_exps, 0, "union_rdfscan_exps: ");
			*/
			
			#if HANDLING_EXCEPTION

			rel_rdfscan = build_rdfexception(c, tmptbId[0], jg, union_rdfscan_exps, nijgroup, ijgroup, nnodes_per_ijgroup, spprops);
				
			printf("RDF exception\n"); 
			_rel_print(c, rel_rdfscan); 

			rel = rel_setop(c->sa, rel_alltable, rel_rdfscan, op_union); 
			rel->exps = sp_proj_exps[0]; 
			#else
			rel = rel_alltable; 
			#endif

		}
		else {
			#if HANDLING_EXCEPTION
			rel_rdfscan = build_rdfexception(c, -1, jg, NULL,  nijgroup, ijgroup, nnodes_per_ijgroup, spprops);
			rel = rel_rdfscan; 
			#else
			printf("There must be a matching table!!!!!\n"); 
			assert(0); 
			#endif
		}



		free_inner_join_groups(ijgroup, nijgroup, nnodes_per_ijgroup); 

		free_sp_props(spprops);
		free(contain_mv_col);
		free(sp_opt_proj_exps); 
		free(tbl_m_rels);
	}

		
	//Only basetable --> this node has only one pattern from basetable
	if (is_only_basetable){
		sql_rel *tmprel = (sql_rel*) (jg->lstnodes[group[0]]->data);
		rel = rel_copy(c->sa, tmprel);
	}
	
	return rel; 
}

static 
void group_star_pattern(mvc *c, jgraph *jg, int numsp, sql_rel** lstRels){

	int i; 
	int** group; //group of nodes in a same pattern
	int* nnode_per_group; 
	int* idx; 
	
	group = (int **)malloc(sizeof(int*) * numsp); 
	nnode_per_group = (int *) malloc(sizeof(int) * numsp);
	idx = (int *) malloc(sizeof(int) * numsp);


	for (i = 0; i < numsp; i++){
		nnode_per_group[i] = 0;
		idx[i] = 0;
	}

	for (i = 0; i < jg->nNode; i++){
		jgnode *node = jg->lstnodes[i]; 
		assert(node->patternId < numsp); 
		nnode_per_group[node->patternId]++; 	
	}

	//Init for group
	for (i = 0; i < numsp; i++){
		group[i] = (int*) malloc(sizeof(int) * nnode_per_group[i]); 
	}

	//add nodeIds for each group
	
	for (i = 0; i < jg->nNode; i++){
		jgnode *node = jg->lstnodes[i];
		int spId = node->patternId; 
		group[spId][idx[spId]] = node->vid; 
		idx[spId]++; 
	}

	//Merge sql_rels in each group into one sql_rel
	for (i = 0; i < numsp; i++){
		lstRels[i] = _group_star_pattern(c, jg, group[i], nnode_per_group[i], i); 
		if (!lstRels[i]){
			printf("Group pattern %d cannot be converted to select from rel table\n", i); 
		}
	}

	//Free
	for (i = 0; i < numsp; i++){
		free(group[i]);
	}
	free(group); 
	free(nnode_per_group); 
	free(idx); 
}




static
void detect_star_pattern(jgraph *jg, int *numsp){
	
	int i; 
	int pId = -1; 
	int num = jg->nNode;
	int optionalMode = 0; 	
	//optinal mode will be turn on 
	//when there is an outer join edge

	for (i = 0; i < num; i++){
		jgnode *node = jg->lstnodes[i]; 
		if (node->patternId == -1){
			pId++;
			node->patternId = pId; 
			optionalMode = 0;
			_detect_star_pattern(jg, node, pId, optionalMode); 	
		}
	}

	*numsp = pId + 1; 
}

void buildJoinGraph(mvc *c, sql_rel *r, int depth){
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
	char **isConnect; 	//Matrix storing state whether two nodes are conneccted	
			  	//In case of large sparse graph, this should not be used.
	int numsp = 0; 	 	//Number of star pattern
	sql_rel** lstRels = NULL; 	//One rel for replacing one star-pattern


	jgedge** lst_cross_edges = NULL; //Cross pattern edges
	int num_cross_edges = 0; 
	int *cr_ed_orders = NULL; 	//orders of applying cross edges for connecting sp
	sql_rel** lst_cross_edge_rels; 	//One rel for replacing edges connecting nodes from 2 star patterns
	int last_cre = -1; 		//The last cross edge, which will be connected to the root of the plan

	int last_rel_join_id = -1; 

	int e_start_level = 0; 	//Depth of the relational plan where the sp transformating of join edge is processed.
	int n_start_level = 0; 	//Depth of the relational plan where the sp transformating of join edge is processed.
	sql_rel *node_root = NULL; //Root from that point to sp
	sql_rel *edge_root = NULL; 

	int hasOuter = 0;  	

	(void) c; 
	(void) r; 
	(void) depth; 


	jg = initJGraph(); 
	
	addRelationsToJG(c, NULL, r, depth, jg, 0, &subjgId, &n_start_level, 0, &node_root, &hasOuter); 

	//This hacking is specially for handling 
	//query without outer join of known subj patterns (e.g., q5 bsbm)
	if (hasOuter == 0){		
		connect_same_subj_node(jg); 	
	}

	nm = create_nMap(MAX_JGRAPH_NODENUMBER); 
	add_relNames_to_nmap(jg, nm); 

	isConnect = createMatrix(jg->nNode, 0); 

	addJoinEdgesToJG(c, NULL, r, depth, jg, 0, &subjgId2, nm, isConnect, &last_rel_join_id, -1, &e_start_level, 0, &edge_root);
	

	detect_star_pattern(jg, &numsp); 

	printRel_JGraph(jg, c); 

	if (0) create_abstract_table(c);
	
	lstRels = (sql_rel**) malloc(sizeof(sql_rel*) * numsp); 


	group_star_pattern(c, jg, numsp, lstRels); 
	
	{
		int i; 
		printf("--------- Each star pattern ----------\n");
		for (i = 0; i < numsp; i++){
			printf("++++Star pattern %d \n",i);
			_rel_print(c, lstRels[i]); 
			printf("\n"); 
		}
	}

	lst_cross_edges = get_cross_sp_edges(jg, &num_cross_edges);

	cr_ed_orders = get_crossedge_apply_orders(jg, lst_cross_edges, num_cross_edges);

	lst_cross_edge_rels = (sql_rel**) malloc(sizeof(sql_rel*) * num_cross_edges);

	//Change the pointer pointing to the first join
	//to the address of the lstRels[0], the rel for the first star-pattern
	printf("e_start_level = %d   |   n_start_level = %d\n", e_start_level, n_start_level); 
	if (numsp == 1) connect_rel_with_sprel(r, lstRels[0], e_start_level, n_start_level, node_root, edge_root); 
	
	//Check the projection operator and then refine the expressions

	if (numsp > 1){
		//Connect to the first edge between sp0 and sp1
		
		build_all_rels_from_cross_edges(c, num_cross_edges, lst_cross_edges, cr_ed_orders, jg, lst_cross_edge_rels, lstRels, numsp, &last_cre); 
		
					
		connect_rel_with_sprel(r, lst_cross_edge_rels[last_cre], e_start_level, n_start_level, node_root, edge_root); 
	}

	//rel_print(c, r, 0); 


	//Check global_csset
	//print_simpleCSset(global_csset);   

	free_nMap(nm); 
	freeMatrix(jg->nNode, isConnect); 
	free(cr_ed_orders);
	//printJGraph(jg); 

	freeJGraph(jg); 
	
}

void transform_to_rel_plan(mvc *c, sql_rel *r){
	int hasUnion = 0; 

	handling_Union(c, r, 0, &hasUnion);

	if (hasUnion == 0) {
		buildJoinGraph(c, r, 0);
	}
}
