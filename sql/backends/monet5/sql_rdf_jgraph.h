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

#ifndef _SQL_RDF_JGRAPH_H_
#define _SQL_RDF_JGRAPH_H_
#include "sql.h"		
#ifdef HAVE_RAPTOR
#include <rdfjgraph.h>
#endif

#ifdef WIN32
#ifndef LIBRDF
#define rdf_export extern __declspec(dllimport)
#else
#define rdf_export extern __declspec(dllexport)
#endif
#else
#define rdf_export extern
#endif

#define tbl_abstract_name "tblabstract"
#define subj_col_name "subject"
#define default_abstbl_col_type "varchar"
#define USING_UNION_FOR_MULTIPLE_MATCH 1

/* Map for the name -> node Id */
typedef struct nMap {
	BAT *lmap; 
	BAT *rmap; 
} nMap; 

/*Star pattern property option */
typedef enum PropOption{
	REQUIRED, 
	OPTIONAL,
	NAV		
} sp_po; 		

typedef enum ColumnType {
	CTYPE_SG,		//single-valued column
	CTYPE_MV		//multi-valued column
} ctype; 

typedef struct o_constraint{
	int cmp_type; 
	oid low; 
	oid hi; 
} o_constraint;

typedef struct propertyList {
	int num; 
	oid subj; 	
	char** lstProps; 
	oid* lstPropIds; 
	char** lstAlias; //Table alias from original query
	sp_po *lstPOs; 
	ctype *lstctype;
	list *exps; 	//Expression of all 
	o_constraint *lst_o_constraints; 
} spProps; 		//star pattern property list

typedef struct mvProp{
	char *cname; 
	char *mvtblname;
	char *atblname;		//alias tbl name (used for joining with base table)
	char *asubjcolname; 	//alias subject column name (used for joining with base table)
	sql_rel *mvrel;		//SELECT statement to mvtable
	//list *mvjoinexps; 	//expression for the join with main table	
} mvPropRel; 		

#define GETMATCHING_TBL_BY_RP_ONLY 1

rdf_export
void buildJoinGraph(mvc *c, sql_rel *r, int depth); 

rdf_export
void transform_to_rel_plan(mvc *c, sql_rel *r); 

#endif /* _SQL_RDF_JGRAPH_H_ */
