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
#include "sql.h"
#include "sql_result.h"
#include "sql_gencode.h"
#include <sql_storage.h>
#include <sql_scenario.h>
#include <store_sequence.h>
#include <sql_optimizer.h>
#include <sql_datetime.h>
#include <rel_optimizer.h>
#include <rel_distribute.h>
#include <rel_select.h>
#include <rel_exp.h>
#include <rel_dump.h>
#include <rel_bin.h>
#include <bbp.h>
#include <cluster.h>
#include <opt_pipes.h>
#include "clients.h"
#ifdef HAVE_RAPTOR
# include <rdf.h>
# include <rdfschema.h>
#include <rdfretrieval.h>
#endif
#include "mal_instruction.h"

/*
 * Shredding RDF documents through SQL
 * Wrapper around the RDF shredder of the rdf module of M5.
 *
 * An rdf file can be now shredded with SQL command:
 * CALL rdf_shred('/path/to/location','graph name');
 *
 * The table rdf.graph will be updated with an entry of the form:
 * [graph name, graph id] -> [gname,gid].
 *
 * In addition all permutation of SPO for the specific rdf document will be
 * created. The name of the triple tables are rdf.pso$gid$, rdf.spo$gid$ etc.
 * For example if gid = 3 then rdf.spo3 is the triple table ordered on subject,
 * property, object. Finally, there is one more table called rdf.map$gid$ that
 * maps oids to strings (i.e., the lexical representation).
 */

str
SQLrdfShred(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
#ifdef HAVE_RAPTOR
	BAT *b[128];
	BAT *p, *s, *o;
	sql_schema *sch;
	sql_table *g_tbl;
	sql_column *gname, *gid;
#if STORE == TRIPLE_STORE
	#if IS_COMPACT_TRIPLESTORE == 0				       
	sql_table *spo_tbl, *sop_tbl, *pso_tbl, *pos_tbl, *osp_tbl, *ops_tbl;
	#else
	sql_table *spo_tbl;
	#endif
#elif STORE == MLA_STORE
	sql_table *spo_tbl;
#endif /* STORE */
	sql_table *map_tbl;
	sql_subtype tpe;
	str *location = (str *) getArgReference(stk,pci,1);
	str *name = (str *) getArgReference(stk,pci,2);
	str *schema = (str *) getArgReference(stk,pci,3);
	char buff[24];
	mvc *m = NULL;
	int id = 0;
	oid rid = oid_nil;
	str msg;
#if IS_DUPLICATE_FREE == 0
	BATiter si, pi, oi; 
	BUN	pb, qb; 
	oid 	*sbt, *pbt, *obt; 
	oid	curS = 0, curP = 0, curO = 0;
#endif /* IS_DUPLICATE_FREE */

	BAT *ontbat = NULL; 
	clock_t tmpbeginT, tmpendT, beginT, endT; 
	
	beginT = clock();

	rethrow("sql.rdfShred", msg, getSQLContext(cntxt, mb, &m, NULL));

	if ((sch = mvc_bind_schema(m, *schema)) == NULL)
		throw(SQL, "sql.rdfShred", "3F000!schema missing");

	g_tbl = mvc_bind_table(m, sch, "graph");
	gname = mvc_bind_column(m, g_tbl, "gname");
	gid = mvc_bind_column(m, g_tbl, "gid");

	rid = table_funcs.column_find_row(m->session->tr, gname, *name, NULL);
	if (rid != oid_nil)
		throw(SQL, "sql.rdfShred", "graph name already exists in rdf.graph");

	id = (int) store_funcs.count_col(m->session->tr, gname, 1);
	store_funcs.append_col(m->session->tr, gname, *name, TYPE_str);
	store_funcs.append_col(m->session->tr, gid, &id, TYPE_int);

	ontbat = mvc_bind(m, "sys", "ontlist", "mont",0);

	tmpbeginT = clock(); 
	if (ontbat == NULL){
		printf("There is no column ontlist/mont \n"); 
		rethrow("sql.rdfShred", msg, RDFParser(b, location, name, schema, NULL));
	}
	else{
		rethrow("sql.rdfShred", msg, RDFParser(b, location, name, schema, &ontbat->batCacheid));

		BBPunfix(ontbat->batCacheid);
	}
	tmpendT = clock(); 
	printf ("Parsing process took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

	if (sizeof(oid) == 8) {
		sql_find_subtype(&tpe, "oid", 31, 0);
		/* todo for niels: if use int/bigint the @0 is serialized */
		/* sql_find_subtype(&tpe, "bigint", 64, 0); */
	} else {
		sql_find_subtype(&tpe, "oid", 31, 0);
		/* sql_find_subtype(&tpe, "int", 32, 0); */
	}
#if STORE == TRIPLE_STORE
	sprintf(buff, "spo%d", id);
	spo_tbl = mvc_create_table(m, sch, buff, tt_table, 0,
				   SQL_PERSIST, 0, 3);
	mvc_create_column(m, spo_tbl, "subject", &tpe);
	mvc_create_column(m, spo_tbl, "property", &tpe);
	mvc_create_column(m, spo_tbl, "object", &tpe);

	#if IS_COMPACT_TRIPLESTORE == 0 
	tmpbeginT = clock(); 
	sprintf(buff, "sop%d", id);
	sop_tbl = mvc_create_table(m, sch, buff, tt_table, 0,
				   SQL_PERSIST, 0, 3);
	mvc_create_column(m, sop_tbl, "subject", &tpe);
	mvc_create_column(m, sop_tbl, "object", &tpe);
	mvc_create_column(m, sop_tbl, "property", &tpe);

	sprintf(buff, "pso%d", id);
	pso_tbl = mvc_create_table(m, sch, buff, tt_table, 0,
				   SQL_PERSIST, 0, 3);
	mvc_create_column(m, pso_tbl, "property", &tpe);
	mvc_create_column(m, pso_tbl, "subject", &tpe);
	mvc_create_column(m, pso_tbl, "object", &tpe);

	sprintf(buff, "pos%d", id);
	pos_tbl = mvc_create_table(m, sch, buff, tt_table, 0,
				   SQL_PERSIST, 0, 3);
	mvc_create_column(m, pos_tbl, "property", &tpe);
	mvc_create_column(m, pos_tbl, "object", &tpe);
	mvc_create_column(m, pos_tbl, "subject", &tpe);

	sprintf(buff, "osp%d", id);
	osp_tbl = mvc_create_table(m, sch, buff, tt_table, 0,
				   SQL_PERSIST, 0, 3);
	mvc_create_column(m, osp_tbl, "object", &tpe);
	mvc_create_column(m, osp_tbl, "subject", &tpe);
	mvc_create_column(m, osp_tbl, "property", &tpe);

	sprintf(buff, "ops%d", id);
	ops_tbl = mvc_create_table(m, sch, buff, tt_table, 0,
				   SQL_PERSIST, 0, 3);
	mvc_create_column(m, ops_tbl, "object", &tpe);
	mvc_create_column(m, ops_tbl, "property", &tpe);
	mvc_create_column(m, ops_tbl, "subject", &tpe);

	tmpendT = clock(); 
	printf ("Creating remaining triple tables took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

	#endif /* IS_COMPACT_TRIPLESTORE == 0 */

#elif STORE == MLA_STORE
	sprintf(buff, "spo%d", id);
	spo_tbl = mvc_create_table(m, sch, buff, tt_table,
				   0, SQL_PERSIST, 0, 3);
	mvc_create_column(m, spo_tbl, "subject", &tpe);
	mvc_create_column(m, spo_tbl, "property", &tpe);
	mvc_create_column(m, spo_tbl, "object", &tpe);
#endif /* STORE */

	sprintf(buff, "map%d", id);
	map_tbl = mvc_create_table(m, sch, buff, tt_table, 0, SQL_PERSIST, 0, 2);
	mvc_create_column(m, map_tbl, "sid", &tpe);
	sql_find_subtype(&tpe, "varchar", 1024, 0);
	mvc_create_column(m, map_tbl, "lexical", &tpe);

	s = b[MAP_LEX];
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, map_tbl, "lexical"),
			BATmirror(BATmark(BATmirror(s),0)), TYPE_bat);
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, map_tbl, "sid"),
			BATmirror(BATmark(s, 0)),
			TYPE_bat);
	BBPunfix(s->batCacheid);

#if STORE == TRIPLE_STORE
	#if IS_DUPLICATE_FREE == 0
		
		s = b[S_sort];
		p = b[P_PO];
		o = b[O_PO];
		si = bat_iterator(s); 
		pi = bat_iterator(p); 
		oi = bat_iterator(o); 

		BATloop(s, pb, qb){
			sbt = (oid *) BUNtloc(si, pb);
			pbt = (oid *) BUNtloc(pi, pb);
			obt = (oid *) BUNtloc(oi, pb);

			if (*sbt != curS || *pbt != curP || *obt != curO){

				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, spo_tbl, "subject"),
						       sbt, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, spo_tbl, "property"),
						       pbt, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, spo_tbl, "object"),
						       obt, TYPE_oid);
				/* Update current value */		       
				curS = *sbt; 
				curP = *pbt; 
				curO = *obt; 
			}
		}
		
		#if IS_COMPACT_TRIPLESTORE == 0
		tmpbeginT = clock(); 
		s = b[S_sort];
		p = b[P_OP];
		o = b[O_OP];

		si = bat_iterator(s); 
		pi = bat_iterator(p); 
		oi = bat_iterator(o); 
		
		curS = 0;
		curP = 0;
		curO = 0;

		BATloop(s, pb, qb){
			sbt = (oid *) BUNtloc(si, pb);
			pbt = (oid *) BUNtloc(pi, pb);
			obt = (oid *) BUNtloc(oi, pb);

			if (*sbt != curS || *pbt != curP || *obt != curO){
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, sop_tbl, "subject"),
						       s, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, sop_tbl, "property"),
						       p, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, sop_tbl, "object"),
						       o, TYPE_oid);
				
				/* Update current value */		       
				curS = *sbt; 
				curP = *pbt; 
				curO = *obt; 
			}
		}

		s = b[S_SO];
		p = b[P_sort];
		o = b[O_SO];

		si = bat_iterator(s); 
		pi = bat_iterator(p); 
		oi = bat_iterator(o); 
		
		curS = 0;
		curP = 0;
		curO = 0;

		BATloop(s, pb, qb){
			sbt = (oid *) BUNtloc(si, pb);
			pbt = (oid *) BUNtloc(pi, pb);
			obt = (oid *) BUNtloc(oi, pb);

			if (*sbt != curS || *pbt != curP || *obt != curO){
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, pso_tbl, "subject"),
						       s, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, pso_tbl, "property"),
						       p, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, pso_tbl, "object"),
						       o, TYPE_oid);		
				
				/* Update current value */		       
				curS = *sbt; 
				curP = *pbt; 
				curO = *obt; 
			}
		}
		s = b[S_OS];
		p = b[P_sort];
		o = b[O_OS];
		
		si = bat_iterator(s); 
		pi = bat_iterator(p); 
		oi = bat_iterator(o); 
		
		curS = 0;
		curP = 0;
		curO = 0;

		BATloop(s, pb, qb){
			sbt = (oid *) BUNtloc(si, pb);
			pbt = (oid *) BUNtloc(pi, pb);
			obt = (oid *) BUNtloc(oi, pb);

			if (*sbt != curS || *pbt != curP || *obt != curO){
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, pos_tbl, "subject"),
						       s, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, pos_tbl, "property"),
						       p, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, pos_tbl, "object"),
						       o, TYPE_oid);
				
				/* Update current value */		       
				curS = *sbt; 
				curP = *pbt; 
				curO = *obt; 
			}
		}

		s = b[S_SP];
		p = b[P_SP];
		o = b[O_sort];
		
		si = bat_iterator(s); 
		pi = bat_iterator(p); 
		oi = bat_iterator(o); 
		
		curS = 0;
		curP = 0;
		curO = 0;

		BATloop(s, pb, qb){
			sbt = (oid *) BUNtloc(si, pb);
			pbt = (oid *) BUNtloc(pi, pb);
			obt = (oid *) BUNtloc(oi, pb);

			if (*sbt != curS || *pbt != curP || *obt != curO){
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, osp_tbl, "subject"),
						       s, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, osp_tbl, "property"),
						       p, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, osp_tbl, "object"),
						       o, TYPE_oid);
				
				/* Update current value */		       
				curS = *sbt; 
				curP = *pbt; 
				curO = *obt; 
			}
		}

		s = b[S_PS];
		p = b[P_PS];
		o = b[O_sort];
		
		si = bat_iterator(s); 
		pi = bat_iterator(p); 
		oi = bat_iterator(o); 
		
		curS = 0;
		curP = 0;
		curO = 0;

		BATloop(s, pb, qb){
			sbt = (oid *) BUNtloc(si, pb);
			pbt = (oid *) BUNtloc(pi, pb);
			obt = (oid *) BUNtloc(oi, pb);

			if (*sbt != curS || *pbt != curP || *obt != curO){
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, ops_tbl, "subject"),
						       s, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, ops_tbl, "property"),
						       p, TYPE_oid);
				store_funcs.append_col(m->session->tr,
						       mvc_bind_column(m, ops_tbl, "object"),
						       o, TYPE_oid);

				/* Update current value */		       
				curS = *sbt; 
				curP = *pbt; 
				curO = *obt; 
			}
		}
		
		tmpendT = clock(); 

		printf ("Inserting from BATs to remaining triple tables took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

		#endif	/* IS_COMPACT_TRIPLESTORE == 0 */		 
			       
	#else  /* IS_DUPLICATE_FREE == 1*/
	
		s = b[S_sort];
		p = b[P_PO];
		o = b[O_PO];
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, spo_tbl, "subject"),
				       s, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, spo_tbl, "property"),
				       p, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, spo_tbl, "object"),
				       o, TYPE_bat);
				       
		#if IS_COMPACT_TRIPLESTORE == 0				       

		s = b[S_sort];
		p = b[P_OP];
		o = b[O_OP];
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, sop_tbl, "subject"),
				       s, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, sop_tbl, "property"),
				       p, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, sop_tbl, "object"),
				       o, TYPE_bat);
				       

		s = b[S_SO];
		p = b[P_sort];
		o = b[O_SO];
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, pso_tbl, "subject"),
				       s, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, pso_tbl, "property"),
				       p, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, pso_tbl, "object"),
				       o, TYPE_bat);
		s = b[S_OS];
		p = b[P_sort];
		o = b[O_OS];
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, pos_tbl, "subject"),
				       s, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, pos_tbl, "property"),
				       p, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, pos_tbl, "object"),
				       o, TYPE_bat);
		s = b[S_SP];
		p = b[P_SP];
		o = b[O_sort];
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, osp_tbl, "subject"),
				       s, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, osp_tbl, "property"),
				       p, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, osp_tbl, "object"),
				       o, TYPE_bat);
		s = b[S_PS];
		p = b[P_PS];
		o = b[O_sort];
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, ops_tbl, "subject"),
				       s, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, ops_tbl, "property"),
				       p, TYPE_bat);
		store_funcs.append_col(m->session->tr,
				       mvc_bind_column(m, ops_tbl, "object"),
				       o, TYPE_bat);
		#endif /* IS_COMPACT_TRIPLESTORE == 0 */	       
	#endif /* IS_DUPLICATE_FREE */

#elif STORE == MLA_STORE
	s = b[S_sort];
	p = b[P_sort];
	o = b[O_sort];
	store_funcs.append_col(m->session->tr,
			       mvc_bind_column(m, spo_tbl, "subject"),
			       s, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			       mvc_bind_column(m, spo_tbl, "property"),
			       p, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			       mvc_bind_column(m, spo_tbl, "object"),
			       o, TYPE_bat);
#endif /* STORE */

	/* unfix graph */
	for(id=0; b[id]; id++) {
		BBPunfix(b[id]->batCacheid);
	}
	
	endT = clock(); 
	printf ("Full rdf_shred() took %f seconds.\n", ((float)(endT - beginT))/CLOCKS_PER_SEC);

	return MAL_SUCCEED;
#else
	(void) cntxt; (void) mb; (void) stk; (void) pci;
	throw(SQL, "sql.rdfShred", "RDF support is missing from MonetDB5");
#endif /* HAVE_RAPTOR */
}

static
void getTblSQLname(char *tmptbname, int tblIdx, int isExTbl, CStableStat *cstablestat, BATiter mapi, BAT *mbat){
	str	baseTblName;
	char	tmpstr[20]; 

	if (isExTbl ==0) 
		sprintf(tmpstr, "%d",tblIdx);
	else //isExTbl == 1
		sprintf(tmpstr, "ex%d",tblIdx);

	getTblName(&baseTblName, cstablestat->lstcstable[tblIdx].tblname, mapi, mbat); 
	sprintf(tmptbname, "%s", baseTblName);
	strcat(tmptbname,tmpstr);

	GDKfree(baseTblName);
}

//If colType == -1, ==> default col
//If not, it is a ex-type column
static
void getColSQLname(char *tmpcolname, int tblIdx, int colIdx, int colType, CStableStat *cstablestat, BATiter mapi, BAT *mbat){
	str baseColName;
	char    tmpstr[20];

	if (colType == -1) sprintf(tmpstr, "%d",colIdx);
	else 
		sprintf(tmpstr, "%dtype%d",colIdx, colType); 
	getTblName(&baseColName, cstablestat->lstcstable[tblIdx].lstProp[colIdx], mapi, mbat);
	sprintf(tmpcolname, "%s", baseColName);
	strcat(tmpcolname,tmpstr); 


	GDKfree(baseColName);
}

static
void getMvTblSQLname(char *tmpmvtbname, int tblIdx, int colIdx, CStableStat *cstablestat, BATiter mapi, BAT *mbat){
	str baseTblName;
	str baseColName; 

	getTblName(&baseTblName, cstablestat->lstcstable[tblIdx].tblname, mapi, mbat);
	getTblName(&baseColName, cstablestat->lstcstable[tblIdx].lstProp[colIdx], mapi, mbat);

	sprintf(tmpmvtbname, "mv%s%d_%s%d", baseTblName, tblIdx, baseColName, colIdx);

	GDKfree(baseTblName);
	GDKfree(baseColName);
}

/*
static
addFKs(CStableStat* cstablestat, CSPropTypes *csPropTypes){
	FILE            *fout;
	char            filename[100];
	int		i;
	char		fromTbl[100]; 
	char		fromTblCol[100]; 
	char		toTbl[100];
	char		toTblCol[100]; 
	int		refTblId; 

	strcpy(filename, "fkCreate.sql");
	fout = fopen(filename, "wt");
	for (i = 0; i < cstablestat->numTables; i++){
		for(j = 0; j < csPropTypes[i].numProp; j++){
			if (csPropTypes[i].lstPropTypes[j].isFKProp == 1){
				refTblId = csPropTypes[i].lstPropTypes[j].refTblId;					
			}
		}
	}
	fclose(fout); 	

}
*/

/* Re-organize triple table by using clustering storage
 * CALL rdf_reorganize('schema','tablename', 1);
 * e.g., rdf_reorganize('rdf','spo0');
 *
 */
str
SQLrdfreorganize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
#ifdef HAVE_RAPTOR
	mvc *m = NULL;
	str *schema = (str *) getArgReference(stk,pci,1);
	str *tbname = (str *) getArgReference(stk,pci,2);
	int *threshold = (int *) getArgReference(stk,pci,3);
	int *mode = (int *) getArgReference(stk,pci,4);
	sql_schema *sch; 
	int ret = 0; 
	CStableStat *cstablestat; 
	char	tmptbname[100]; 
	char	tmpmvtbname[100];
	char	tmptbnameex[100];
	//char	tmpviewname[100]; 
	char	tmpcolname[100]; 
	char	tmpmvcolname[100];
	//char	viewcommand[500];
	sql_subtype tpe; 	
	sql_subtype tpes[50];

	sql_table	**cstables; 
	sql_table	***csmvtables; 	//table for storing multi-values 
	#if CSTYPE_TABLE == 1
	sql_table	**cstablesEx; 
	sql_table	**viewcstables; 
	#endif
	sql_table 	*psotbl;
	int 	i, j, k; 
	int	tmpNumMVCols = 0;
	int	nonullmvtables = 0;
	int	totalNoTablesCreated = 0;
	int	totalNoViewCreated = 0;
	int	totalNoExTables = 0; 
	int	totalNumDefCols = 0; 
	int	totalNumNonDefCols = 0; 

	str msg;
	BAT	*sbat, *pbat, *obat, *mbat; 
	BAT	*tmpbat; 
	BATiter	mapi; 
	clock_t tmpbeginT, tmpendT, beginT, endT;

	BAT *ontbat = NULL;

	beginT = clock();
	
	rethrow("sql.rdfShred", msg, getSQLContext(cntxt, mb, &m, NULL));

	if ((sch = mvc_bind_schema(m, *schema)) == NULL)
		throw(SQL, "sql.rdfShred", "3F000!schema missing");

	sbat = mvc_bind(m, *schema, *tbname, "subject",0);
	pbat = mvc_bind(m, *schema, *tbname, "property",0);
	obat = mvc_bind(m, *schema, *tbname, "object",0);
	mbat = mvc_bind(m, *schema, "map0", "lexical",0);

	cstablestat = (CStableStat *) malloc (sizeof (CStableStat));
	
	ontbat = mvc_bind(m, "sys", "ontlist", "mont",0);

	tmpbeginT = clock();

	if (ontbat == NULL){
		printf("[PROBLEM] There is no column ontlist/mont \n"); 
		throw(SQL, "sql.rdfreorganize", "Colunm ontlist/mont is missing");
	}
	else{
		rethrow("sql.rdfreorganize", msg, RDFreorganize(&ret, cstablestat, &sbat->batCacheid, &pbat->batCacheid, 
				&obat->batCacheid, &mbat->batCacheid, &ontbat->batCacheid, threshold, mode));

		BBPunfix(ontbat->batCacheid);
	}

	tmpendT = clock(); 
	printf ("Sql.mx: Reorganizing process process took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

	//if (*mode == EXPLOREONLY){
	if (*mode < 3){
		BBPunfix(sbat->batCacheid); 
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid); 
		BBPunfix(mbat->batCacheid);
		freeCStableStat(cstablestat); 
		//free(cstablestat);
		return MAL_SUCCEED; 
	}
	
	mapi = bat_iterator(mbat); 	
	
	tmpbeginT = clock();
	cstables = (sql_table **)malloc(sizeof(sql_table*) * cstablestat->numTables);
	csmvtables = (sql_table ***)malloc(sizeof(sql_table**) * cstablestat->numTables);

	#if CSTYPE_TABLE == 1
	cstablesEx = (sql_table **)malloc(sizeof(sql_table*) * cstablestat->numTables);
	viewcstables = (sql_table **)malloc(sizeof(sql_table*) * cstablestat->numTables);
	#endif

	// Put to SQL tables
	sql_find_subtype(&tpe, "oid", 31, 0);

	// Add irregular triples to pso tbale
	psotbl = mvc_create_table(m, sch, "pso", tt_table, 0,
			                                   SQL_PERSIST, 0, 3);
	totalNoTablesCreated++;
	mvc_create_column(m, psotbl, "p",  &tpe);
	mvc_create_column(m, psotbl, "s",  &tpe);
	mvc_create_column(m, psotbl, "o",  &tpe);


	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, psotbl,"p" ), 
			cstablestat->pbat, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, psotbl,"s" ), 
			cstablestat->sbat, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, psotbl,"o" ), 
			cstablestat->obat, TYPE_bat);
	// Add regular triple to cstables and cstablesEx

	printf("Starting creating SQL Table -- \n");
	
	

	sql_find_subtype(&tpes[TYPE_oid], "oid", 31 , 0);
	//printf("Tpes %d Type name is: %s \n", TYPE_oid, tpes[TYPE_oid].type->sqlname);

	sql_find_subtype(&tpes[TYPE_str], "varchar", 500 , 0);
	//printf("Tpes %d Type name is: %s \n", TYPE_str, tpes[TYPE_str].type->sqlname);

	sql_find_subtype(&tpes[TYPE_dbl], "double", 53 , 0);
	//printf("Tpes %d Type name is: %s \n", TYPE_dbl, tpes[TYPE_dbl].type->sqlname);

	sql_find_subtype(&tpes[TYPE_int], "int", 9 , 0);
	//printf("Tpes %d Type name is: %s \n", TYPE_int, tpes[TYPE_int].type->sqlname);

	sql_find_subtype(&tpes[TYPE_flt], "real", 23, 0);
	//printf("Tpes %d Type name is: %s \n", TYPE_flt, tpes[TYPE_flt].type->sqlname);
	
	sql_find_subtype(&tpes[TYPE_timestamp],"timestamp",128,0);

	/*
	sql_find_subtype(&tpe, "float", 0 , 0);
	printf("Test Type name is: %s \n", tpe.type->sqlname);
	sql_find_subtype(&tpe, "int", 9 , 0);
	printf("Test Type name is: %s \n", tpe.type->sqlname);
	sql_find_subtype(&tpe, "oid", 31 , 0);
	printf("Test Type name is: %s \n", tpe.type->sqlname);
	*/
	{
	char*   rdfschema = "rdf";
	if (TKNZRopen (NULL, &rdfschema) != MAL_SUCCEED) {
		throw(RDF, "SQLrdfreorganize","could not open the tokenizer\n");
	}
	}
	for (i = 0; i < cstablestat->numTables; i++){
		//printf("creating table %d \n", i);

		getTblSQLname(tmptbname, i, 0, cstablestat, mapi, mbat);
		printf("Table %d:||  %s ||\n",i, tmptbname);

		cstables[i] = mvc_create_table(m, sch, tmptbname, tt_table, 0,
				   SQL_PERSIST, 0, 3);
		totalNoTablesCreated++;
		//Multivalues tables for each column
		csmvtables[i] = (sql_table **)malloc(sizeof(sql_table*) * cstablestat->numPropPerTable[i]);
		
		#if APPENDSUBJECTCOLUMN
		mvc_create_column(m, cstables[i], "subject",  &tpes[TYPE_oid]);
		#endif
		for (j = 0; j < cstablestat->numPropPerTable[i]; j++){

			//TODO: Use propertyId from Propstat
			getColSQLname(tmpcolname, i, j, -1, cstablestat, mapi, mbat);


			tmpbat = cstablestat->lstcstable[i].colBats[j];

			mvc_create_column(m, cstables[i], tmpcolname,  &tpes[tmpbat->ttype]);
			
			//For multi-values table
			tmpNumMVCols = cstablestat->lstcstable[i].lstMVTables[j].numCol;
			if (tmpNumMVCols != 0){
				getMvTblSQLname(tmpmvtbname, i, j, cstablestat, mapi, mbat);
				csmvtables[i][j] = mvc_create_table(m, sch, tmpmvtbname, tt_table, 0, SQL_PERSIST, 0, 3); 
				totalNoTablesCreated++;

				//One column for key
				sprintf(tmpcolname, "mvKey");
				tmpbat = cstablestat->lstcstable[i].lstMVTables[j].keyBat;
				mvc_create_column(m, csmvtables[i][j], tmpcolname,  &tpes[tmpbat->ttype]);

				//Value columns 
				for (k = 0; k < tmpNumMVCols; k++){
					getColSQLname(tmpmvcolname, i, j, k, cstablestat, mapi, mbat);

					tmpbat = cstablestat->lstcstable[i].lstMVTables[j].mvBats[k];
					mvc_create_column(m, csmvtables[i][j], tmpmvcolname,  &tpes[tmpbat->ttype]);
				}

			}
			else
				nonullmvtables++;
		}
		
		totalNumDefCols += cstablestat->lstcstable[i].numCol;

		#if CSTYPE_TABLE == 1
		// Add non-default type table
		if (cstablestat->lstcstableEx[i].numCol != 0){	

			getTblSQLname(tmptbnameex, i, 1, cstablestat, mapi, mbat);
			printf("TableEx %d: || %s || \n",i, tmptbnameex);

			cstablesEx[i] = mvc_create_table(m, sch, tmptbnameex, tt_table, 0,
					   SQL_PERSIST, 0, 3);
			totalNoTablesCreated++;
			totalNoExTables++;
			for (j = 0; j < cstablestat->lstcstableEx[i].numCol; j++){
				//TODO: Use propertyId from Propstat
				getColSQLname(tmpcolname, i, cstablestat->lstcstableEx[i].mainTblColIdx[j], (int)(cstablestat->lstcstableEx[i].colTypes[j]), cstablestat, mapi, mbat);

				tmpbat = cstablestat->lstcstableEx[i].colBats[j];
				mvc_create_column(m, cstablesEx[i], tmpcolname,  &tpes[tmpbat->ttype]);				
			}
			totalNumNonDefCols += cstablestat->lstcstableEx[i].numCol;
		}

		#endif

		#if APPENDSUBJECTCOLUMN
		{
			BAT* subjBat = createEncodedSubjBat(i,BATcount(cstablestat->lstcstable[i].colBats[0]));
                	store_funcs.append_col(m->session->tr,
					mvc_bind_column(m, cstables[i],"subject"), 
					subjBat, TYPE_bat);
			BBPreclaim(subjBat);
		}
		#endif
		for (j = 0; j < cstablestat->numPropPerTable[i]; j++){

			//TODO: Use propertyId from Propstat
			getColSQLname(tmpcolname, i, j, -1, cstablestat, mapi, mbat);

			tmpbat = cstablestat->lstcstable[i].colBats[j];

			//printf("Column %d: \n",j); 
			//BATprint(tmpbat);
                	store_funcs.append_col(m->session->tr,
					mvc_bind_column(m, cstables[i],tmpcolname ), 
					tmpbat, TYPE_bat);

			//For multi-values table
			tmpNumMVCols = cstablestat->lstcstable[i].lstMVTables[j].numCol;
			if (tmpNumMVCols != 0){

				//One column for key
				sprintf(tmpcolname, "mvKey");
				tmpbat = cstablestat->lstcstable[i].lstMVTables[j].keyBat;
				store_funcs.append_col(m->session->tr,
					mvc_bind_column(m, csmvtables[i][j],tmpcolname), 
					tmpbat, TYPE_bat);

				//Value columns
				for (k = 0; k < tmpNumMVCols; k++){

					getColSQLname(tmpmvcolname, i, j, k, cstablestat, mapi, mbat);

					tmpbat = cstablestat->lstcstable[i].lstMVTables[j].mvBats[k];
					
					//printf("MVColumn %d: \n",k); 
					//BATprint(tmpbat);
                			store_funcs.append_col(m->session->tr,
						mvc_bind_column(m, csmvtables[i][j],tmpmvcolname), 
						tmpbat, TYPE_bat);
				}
			}
			else
				nonullmvtables++;
		}


		#if CSTYPE_TABLE == 1
		// Add non-default type table
		if (cstablestat->lstcstableEx[i].numCol != 0){	
			for (j = 0; j < cstablestat->lstcstableEx[i].numCol; j++){
				//TODO: Use propertyId from Propstat
				getColSQLname(tmpcolname, i, cstablestat->lstcstableEx[i].mainTblColIdx[j], (int)(cstablestat->lstcstableEx[i].colTypes[j]), cstablestat, mapi, mbat);

				tmpbat = cstablestat->lstcstableEx[i].colBats[j];
				
				//printf("ColumnEx %d: \n",j); 
				//BATprint(tmpbat);
				store_funcs.append_col(m->session->tr,
						mvc_bind_column(m, cstablesEx[i],tmpcolname ), 
						tmpbat, TYPE_bat);
			}
		}

		#endif
		//Create a view to combine these tables
		/*
		sprintf(tmpviewname, "viewcstable%d",i);
		sprintf(viewcommand, "SELECT * from rdf.%s UNION SELECT * from rdf.%s;", tmptbname, tmptbnameex); 
		//printf("Create view %s \n", viewcommand);
		viewcstables[i] = mvc_create_view(m, sch, tmpviewname, SQL_PERSIST,viewcommand, 1); 
		totalNoViewCreated++;
		for (j = 0; j < cstablestat->numPropPerTable[i]; j++){
			//TODO: Use propertyId from Propstat
			sprintf(tmpcolname, "col%d",j);
			mvc_create_column(m, viewcstables[i], tmpcolname,  &tpe);
		}
		*/

		//printf("Done creating table %d with %d cols \n", i,cstablestat->numPropPerTable[i]);

	}
	printf("... Done ( %d tables (in which %d tables are non-default) + %d views created. Already exclude %d Null mvtables ) \n", 
			totalNoTablesCreated, totalNoExTables, totalNoViewCreated, nonullmvtables);
	printf("Number of default-type columns: %d \n ", totalNumDefCols);
	printf("Number of non-default-type columns: %d  (%f ex-types per prop) \n ", totalNumNonDefCols, (float)totalNumNonDefCols/totalNumDefCols);

	TKNZRclose(&ret);

	BBPunfix(sbat->batCacheid); 
	BBPunfix(pbat->batCacheid);
	BBPunfix(obat->batCacheid); 
	BBPunfix(mbat->batCacheid);
	for (i = 0; i < cstablestat->numTables; i++){
		free(csmvtables[i]);
	}
	free(csmvtables);

	freeCStableStat(cstablestat); 
	free(cstables);
	free(cstablesEx); 
	free(viewcstables); 

	tmpendT = clock(); 
	printf ("Sql.mx: Put Bats to Relational Table  process took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

	endT = clock(); 
	printf ("Sql.mx: All processes took  %f seconds.\n", ((float)(endT - beginT))/CLOCKS_PER_SEC);

	return MAL_SUCCEED; 
#else
	(void) cntxt; (void) mb; (void) stk; (void) pci;
	throw(SQL, "sql.SQLrdfreorganize", "RDF support is missing from MonetDB5");
#endif /* HAVE_RAPTOR */	
}


str
SQLrdfRetrieveSubschema(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
#ifdef HAVE_RAPTOR
	sql_schema	*schOld, *schNew;
	str		*oldSchema = (str *) getArgReference(stk,pci,1); // schema name (old)
	int		*count = (int *) getArgReference(stk,pci,2); // number of tables in the sub schema
	str		*keyword = (str *) getArgReference(stk,pci,3); // keyword to find the root table
	mvc		*m = NULL;
	str		msg;
	int		i;

	BAT		*table_id_freq_id, *table_id_freq_name, *table_id_freq_freq;
	BAT		*adjacency_list_from, *adjacency_list_to, *adjacency_list_freq;
	BATiter		table_id_freq_name_i;
	BUN		p, q;

	BUN		tableCount = 0;
	int		*table_id = NULL;
	str		*table_name = NULL;
	int		*table_freq = NULL;
	int		*adjacency_from = NULL;
	int		*adjacency_to = NULL;
	int		*adjacency_freq = NULL;
	BUN		adjacencyCount = 0;

	int		root;
	int		countActual;
	int		*nodes = NULL;
	str		name; // new schema name

	// init
	rethrow("sql.rdfRetrieveSubschema", msg, getSQLContext(cntxt, mb, &m, NULL));

	if ((schOld = mvc_bind_schema(m, *oldSchema)) == NULL)
		throw(SQL, "sql.rdfRetrieveSubschema", "3F000!schema missing");

	table_id_freq_id = mvc_bind(m, "sys", "table_id_freq", "id", 0);
	table_id_freq_name = mvc_bind(m, "sys", "table_id_freq", "name", 0);
	table_id_freq_freq = mvc_bind(m, "sys", "table_id_freq", "frequency", 0);
	adjacency_list_from = mvc_bind(m, "sys", "adjacency_list", "from_id", 0);
	adjacency_list_to = mvc_bind(m, "sys", "adjacency_list", "to_id", 0);
	adjacency_list_freq = mvc_bind(m, "sys", "adjacency_list", "frequency", 0);

	tableCount = BATcount(table_id_freq_id);
	table_name = GDKmalloc(sizeof(str) * tableCount);

	if (!table_name) {
		BBPreclaim(table_id_freq_id);
		BBPreclaim(table_id_freq_name);
		BBPreclaim(table_id_freq_freq);
		BBPreclaim(adjacency_list_from);
		BBPreclaim(adjacency_list_to);
		BBPreclaim(adjacency_list_freq);
		throw(SQL, "sql.rdfRetrieveSubschema", "ERROR: Couldn't GDKmalloc table_name array!\n");
	}

	table_id = (int*) Tloc(table_id_freq_id, BUNfirst(table_id_freq_id));
	table_freq = (int*) Tloc(table_id_freq_freq, BUNfirst(table_id_freq_freq));
	table_id_freq_name_i = bat_iterator(table_id_freq_name);

	tableCount = 0;
	BATloop(table_id_freq_name, p, q) {
		table_name[tableCount++] = (str) BUNtail(table_id_freq_name_i, p);
	}

	adjacencyCount = BATcount(adjacency_list_from);

	adjacency_from = (int*) Tloc(adjacency_list_from, BUNfirst(adjacency_list_from));
	adjacency_to = (int*) Tloc(adjacency_list_to, BUNfirst(adjacency_list_to));
	adjacency_freq = (int*) Tloc(adjacency_list_freq, BUNfirst(adjacency_list_freq));

	// TODO find root node using the keyword
	root = 0;
	(void) keyword;

	// call retrieval function
	countActual = 0;
	nodes = retrieval(root, *count, &countActual, table_id, table_name, table_freq, tableCount, adjacency_from, adjacency_to, adjacency_freq, adjacencyCount);

	// create schema
	name = "s123"; // TODO create schema name
	schNew = mvc_create_schema(m, name, schOld->auth_id, schOld->owner);

	for (i = 0; i < countActual; ++i) {
		sql_table *t;
		char query[500];
		sql_table *view;
		node *n;

		// get chosen table
		t = mvc_bind_table(m, schOld, table_name[nodes[i]]);
		assert(t != NULL); // else: inconsistency in my data!

		// create view
		sprintf(query, "select * from %s.%s;", *oldSchema, table_name[nodes[i]]);
		view = mvc_create_view(m, schNew, table_name[nodes[i]], SQL_PERSIST, query, 0);

		// loop through columns and copy them
		for (n = t->columns.set->h; n; n = n->next) {
			sql_column *c = n->data;

			mvc_copy_column(m, view, c);
		}
	}

	BBPreclaim(table_id_freq_id);
	BBPreclaim(table_id_freq_name);
	BBPreclaim(table_id_freq_freq);
	BBPreclaim(adjacency_list_from);
	BBPreclaim(adjacency_list_to);
	BBPreclaim(adjacency_list_freq);

	GDKfree(table_id);
	GDKfree(table_name);
	GDKfree(table_freq);
	GDKfree(adjacency_from);
	GDKfree(adjacency_to);
	GDKfree(adjacency_freq);

	return MAL_SUCCEED;
#else
	(void) cntxt; (void) mb; (void) stk; (void) pci;
	throw(SQL, "sql.rdfRetrieveSubschema", "RDF support is missing from MonetDB5");
#endif /* HAVE_RAPTOR */
}
