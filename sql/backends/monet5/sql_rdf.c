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
 * Copyright August 2008-2015 MonetDB B.V.
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
#include <opt_pipes.h>
#include "clients.h"
#include "sql_rdf.h"
#include "mal_instruction.h"
#include "rdfontologyload.h"

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
	str *location = getArgReference_str(stk, pci, 1);
	str *name = getArgReference_str(stk, pci, 2);
	str *schema = getArgReference_str(stk, pci, 3);
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
			s, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, map_tbl, "sid"),
			BATdense(s->hseqbase, s->hseqbase, BATcount(s)),
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

void getTblSQLname(char *tmptbname, int tblIdx, int isExTbl, oid tblname, BATiter mapi, BAT *mbat){
	str	baseTblName;
	char	tmpstr[20]; 

	if (isExTbl ==0) 
		sprintf(tmpstr, "%d",tblIdx);
	else //isExTbl == 1
		sprintf(tmpstr, "ex%d",tblIdx);

	getSqlName(&baseTblName, tblname, mapi, mbat); 
	sprintf(tmptbname, "%s", baseTblName);
	strcat(tmptbname,tmpstr);

	GDKfree(baseTblName);
}

//If colType == -1, ==> default col
//If not, it is a ex-type column

void getColSQLname(char *tmpcolname, int colIdx, int colType, oid propid, BATiter mapi, BAT *mbat){
	str baseColName;
	char    tmpstr[20];

	if (colType == -1) sprintf(tmpstr, "%d",colIdx);
	else 
		sprintf(tmpstr, "%dtype%d",colIdx, colType); 
	getSqlName(&baseColName, propid, mapi, mbat);
	sprintf(tmpcolname, "%s", baseColName);
	strcat(tmpcolname,tmpstr); 


	GDKfree(baseColName);
}

void getMvTblSQLname(char *tmpmvtbname, int tblIdx, int colIdx, oid tblname, oid propid, BATiter mapi, BAT *mbat){
	str baseTblName;
	str baseColName; 

	getSqlName(&baseTblName, tblname, mapi, mbat);
	getSqlName(&baseColName, propid, mapi, mbat);

	sprintf(tmpmvtbname, "mv%s%d_%s%d", baseTblName, tblIdx, baseColName, colIdx);

	GDKfree(baseTblName);
	GDKfree(baseColName);
}

static
void addPKandFKs(CStableStat* cstablestat, CSPropTypes *csPropTypes, str schema, BATiter mapi, BAT *mbat){
	FILE            *fout, *foutPK, *foutMV;
	char            filename[100];
	int		i, j;
	char		fromTbl[100]; 
	char		fromTblCol[100]; 
	char		toTbl[100];
	char		mvTbl[100]; 
	char		mvCol[100];
	int		refTblId; 
	int		tblColIdx; 

	strcpy(filename, "fkCreate.sql");
	fout = fopen(filename, "wt");
	foutPK = fopen("pkCreate.sql","wt");
	foutMV = fopen("mvRefCreate.sql","wt"); 

	for (i = 0; i < cstablestat->numTables; i++){

		//Add PKs to all subject columns
		getTblSQLname(fromTbl, i, 0, cstablestat->lstcstable[i].tblname, mapi, mbat);
		fprintf(foutPK, "ALTER TABLE %s.\"%s\" ADD PRIMARY KEY (subject);\n",schema,fromTbl);
	
		//Add unique key constraint and FKs between MVTable and its corresponding column
		//in the default table
		for (j = 0; j < cstablestat->numPropPerTable[i]; j++){
			if (cstablestat->lstcstable[i].lstMVTables[j].numCol != 0){	//MVColumn
				getColSQLname(fromTblCol, j, -1, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);
				getMvTblSQLname(mvTbl, i, j, cstablestat->lstcstable[i].tblname, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);
				fprintf(foutMV, "ALTER TABLE %s.\"%s\" ADD UNIQUE (\"%s\");\n",schema,fromTbl,fromTblCol);		
				fprintf(foutMV, "ALTER TABLE %s.\"%s\" ADD FOREIGN KEY (\"mvKey\") REFERENCES %s.\"%s\" (\"%s\");\n",schema, mvTbl, schema, fromTbl,fromTblCol);

				fprintf(foutMV, "ALTER TABLE %s.\"%s\" ADD FOREIGN KEY (\"mvsubj\") REFERENCES %s.\"%s\" (\"subject\");\n",schema, mvTbl, schema, fromTbl);
					
				//Add primary key for MV table
				getColSQLname(mvCol, j, 0, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);
				fprintf(foutPK, "ALTER TABLE %s.\"%s\" ADD PRIMARY KEY (mvsubj, \"%s\");\n",schema,mvTbl,mvCol);
			}
		}
	}

	for (i = 0; i < cstablestat->numTables; i++){
		for(j = 0; j < csPropTypes[i].numProp; j++){
			if (csPropTypes[i].lstPropTypes[j].defColIdx == -1)	continue;
			if (csPropTypes[i].lstPropTypes[j].isFKProp == 1){
				tblColIdx = csPropTypes[i].lstPropTypes[j].defColIdx; 
				getTblSQLname(fromTbl, i, 0, cstablestat->lstcstable[i].tblname, mapi, mbat);
				refTblId = csPropTypes[i].lstPropTypes[j].refTblId;
				getTblSQLname(toTbl, refTblId, 0, cstablestat->lstcstable[refTblId].tblname, mapi, mbat);

				if (cstablestat->lstcstable[i].lstMVTables[tblColIdx].numCol == 0){
					getColSQLname(fromTblCol, tblColIdx, -1, cstablestat->lstcstable[i].lstProp[tblColIdx], mapi, mbat);

					fprintf(fout, "ALTER TABLE %s.\"%s\" ADD FOREIGN KEY (\"%s\") REFERENCES %s.\"%s\" (subject);\n\n", schema, fromTbl, fromTblCol, schema, toTbl);

				}
				else{	//This is a MV col
					getMvTblSQLname(mvTbl, i, tblColIdx, cstablestat->lstcstable[i].tblname, cstablestat->lstcstable[i].lstProp[tblColIdx], mapi, mbat);
					getColSQLname(fromTblCol, tblColIdx, -1, cstablestat->lstcstable[i].lstProp[tblColIdx], mapi, mbat);
					getColSQLname(mvCol, tblColIdx, 0, cstablestat->lstcstable[i].lstProp[tblColIdx], mapi, mbat); //Use the first column of MVtable
					
					if (0){		//Do not create the FK from MVtable to the original table
							//Since that column in original table may contains lots of NULL value
					fprintf(fout, "ALTER TABLE %s.\"%s\" ADD PRIMARY KEY (\"%s\");\n",schema, fromTbl,fromTblCol);
					fprintf(fout, "ALTER TABLE %s.\"%s\" ADD FOREIGN KEY (mvKey) REFERENCES %s.\"%s\" (\"%s\");\n",schema, mvTbl, schema, fromTbl,fromTblCol);
					}
					fprintf(fout, "ALTER TABLE %s.\"%s\" ADD FOREIGN KEY (\"%s\") REFERENCES %s.\"%s\" (subject);\n\n",schema, mvTbl, mvCol, schema, toTbl);
					
				}
			}
		}
	}
	fclose(fout); 	
	fclose(foutPK); 
	fclose(foutMV); 

}

static
int isRightPropBAT(BAT *b){
	
	if (b->trevsorted == 1){ 
		printf("Prop of the BAT is violated\n"); 
		return 0; 
	}
	if (b->tsorted == 1){
		printf("Prop of the BAT is violated\n"); 
		return 0; 
	}
	
	return 1; 
}

/*
 * Order the property by their support
 * for each table
 * */
static
int** createColumnOrder(CStableStat* cstablestat, CSPropTypes *csPropTypes){
	int i, j, k; 
	int num = cstablestat->numTables;
	int **colOrder;
	int tblColIdx; 
	
	colOrder = (int **)GDKmalloc(sizeof(int*) * num);
	for (i = 0; i < num; i++){
		int* tmpPropFreqs; 
		tmpPropFreqs = (int *) GDKmalloc(sizeof(int) * cstablestat->numPropPerTable[i]);
		colOrder[i] = (int *) GDKmalloc(sizeof(int) * cstablestat->numPropPerTable[i]); 
		
		for (j = 0; j < cstablestat->numPropPerTable[i]; j ++){
			colOrder[i][j] = j; 
		}

		//Store the frequencies of table columns to tmp arrays
		for (j = 0; j < csPropTypes[i].numProp; j++){
			tblColIdx = csPropTypes[i].lstPropTypes[j].defColIdx;
			if (tblColIdx == -1) continue; 
			assert(tblColIdx < cstablestat->numPropPerTable[i]);
			tmpPropFreqs[tblColIdx] = csPropTypes[i].lstPropTypes[j].propFreq; 		
		}


		//Do insertion sort the the property ascending according to their support	
		for (j = 1; j < cstablestat->numPropPerTable[i]; j ++){
			int tmpPos = colOrder[i][j]; 
			int tmpFreq = tmpPropFreqs[tmpPos];
			k = j; 
			while (k > 0 && tmpFreq > tmpPropFreqs[colOrder[i][k-1]]){
				colOrder[i][k] = colOrder[i][k-1]; 					
				k--;
			}
			colOrder[i][k] = tmpPos;
		}
	}
	
	return colOrder; 	
}

static
void freeColOrder(int **colOrder, CStableStat* cstablestat){
	int i; 
	for (i = 0; i < cstablestat->numTables; i++){
		GDKfree(colOrder[i]);	
	}
	GDKfree(colOrder);
} 

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
	CSPropTypes     *csPropTypes;
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
	#if TRIPLEBASED_TABLE	
	sql_table	*respotbl; 	/*reorganized spo table*/
	#endif
	int 	i, j, k, colId; 
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
	int **colOrder = NULL; 

	(void) tmptbnameex;

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
		rethrow("sql.rdfreorganize", msg, RDFreorganize(&ret, cstablestat, &csPropTypes, &sbat->batCacheid, &pbat->batCacheid, 
				&obat->batCacheid, &mbat->batCacheid, &ontbat->batCacheid, threshold, mode));

		BBPunfix(ontbat->batCacheid);
	}

	tmpendT = clock(); 
	printf ("Sql.mx: Reorganizing process process took %f seconds.\n", ((float)(tmpendT - tmpbeginT))/CLOCKS_PER_SEC);

	//if (*mode == EXPLOREONLY){
	if (*mode < BUILDTABLE){
		BBPunfix(sbat->batCacheid); 
		BBPunfix(pbat->batCacheid);
		BBPunfix(obat->batCacheid); 
		BBPunfix(mbat->batCacheid);
		freeCSPropTypes(csPropTypes,cstablestat->numTables);
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
	
	#if TRIPLEBASED_TABLE
	printf("Create triple-based relational table ..."); 
	respotbl = mvc_create_table(m, sch, "triples", tt_table, 0,
			                                   SQL_PERSIST, 0, 3);
	totalNoTablesCreated++;
	mvc_create_column(m, respotbl, "p",  &tpe);
	mvc_create_column(m, respotbl, "s",  &tpe);
	mvc_create_column(m, respotbl, "o",  &tpe);


	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, respotbl,"p" ), 
			cstablestat->repbat, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, respotbl,"s" ), 
			cstablestat->resbat, TYPE_bat);
	store_funcs.append_col(m->session->tr,
			mvc_bind_column(m, respotbl,"o" ), 
			cstablestat->reobat, TYPE_bat);
	printf("Done\n");

	#endif

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

	//Re-order the columns 
	colOrder = createColumnOrder(cstablestat, csPropTypes);

	for (i = 0; i < cstablestat->numTables; i++){
		//printf("creating table %d \n", i);

		getTblSQLname(tmptbname, i, 0, cstablestat->lstcstable[i].tblname, mapi, mbat);
		printf("Table %d:||  %s ||\n",i, tmptbname);

		cstables[i] = mvc_create_table(m, sch, tmptbname, tt_table, 0,
				   SQL_PERSIST, 0, 3);
		totalNoTablesCreated++;
		//Multivalues tables for each column
		csmvtables[i] = (sql_table **)malloc(sizeof(sql_table*) * cstablestat->numPropPerTable[i]);
		
		#if APPENDSUBJECTCOLUMN
		mvc_create_column(m, cstables[i], "subject",  &tpes[TYPE_oid]);
		#endif
		for (colId = 0; colId < cstablestat->numPropPerTable[i]; colId++){
			j = colOrder[i][colId]; 
		//for (j = 0; j < cstablestat->numPropPerTable[i]; j++){

			//TODO: Use propertyId from Propstat
			getColSQLname(tmpcolname, j, -1, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);


			tmpbat = cstablestat->lstcstable[i].colBats[j];
			isRightPropBAT(tmpbat);

			mvc_create_column(m, cstables[i], tmpcolname,  &tpes[tmpbat->ttype]);
			
			//For multi-values table
			tmpNumMVCols = cstablestat->lstcstable[i].lstMVTables[j].numCol;
			if (tmpNumMVCols != 0){
				getMvTblSQLname(tmpmvtbname, i, j, cstablestat->lstcstable[i].tblname, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);
				csmvtables[i][j] = mvc_create_table(m, sch, tmpmvtbname, tt_table, 0, SQL_PERSIST, 0, 3); 
				totalNoTablesCreated++;

				//One column for key
				sprintf(tmpcolname, "mvKey");
				tmpbat = cstablestat->lstcstable[i].lstMVTables[j].keyBat;
				isRightPropBAT(tmpbat);
				mvc_create_column(m, csmvtables[i][j], tmpcolname,  &tpes[tmpbat->ttype]);

				//One column for subj oid
				sprintf(tmpcolname, "mvsubj");
				tmpbat = cstablestat->lstcstable[i].lstMVTables[j].subjBat;
				isRightPropBAT(tmpbat);
				mvc_create_column(m, csmvtables[i][j], tmpcolname,  &tpes[tmpbat->ttype]);

				//Value columns 
				for (k = 0; k < tmpNumMVCols; k++){
					#if STORE_ALL_EXCEPTION_IN_PSO == 1
					if (k > 0) continue; //Only keep the first default type column as other columns will be stored in pso
					#endif
					getColSQLname(tmpmvcolname, j, k, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);

					tmpbat = cstablestat->lstcstable[i].lstMVTables[j].mvBats[k];
					isRightPropBAT(tmpbat);
					mvc_create_column(m, csmvtables[i][j], tmpmvcolname,  &tpes[tmpbat->ttype]);
				}

			}
			else
				nonullmvtables++;
		}
		
		totalNumDefCols += cstablestat->lstcstable[i].numCol;

		#if CSTYPE_TABLE == 1
		#if STORE_ALL_EXCEPTION_IN_PSO == 0
		// Add non-default type table
		if (cstablestat->lstcstableEx[i].numCol != 0){	

			getTblSQLname(tmptbnameex, i, 1, cstablestat->lstcstable[i].tblname, mapi, mbat);
			printf("TableEx %d: || %s || \n",i, tmptbnameex);

			cstablesEx[i] = mvc_create_table(m, sch, tmptbnameex, tt_table, 0,
					   SQL_PERSIST, 0, 3);
			totalNoTablesCreated++;
			totalNoExTables++;
			for (j = 0; j < cstablestat->lstcstableEx[i].numCol; j++){
				//TODO: Use propertyId from Propstat
				int tmpcolidx = cstablestat->lstcstableEx[i].mainTblColIdx[j];
				getColSQLname(tmpcolname, tmpcolidx, (int)(cstablestat->lstcstableEx[i].colTypes[j]), 
						cstablestat->lstcstable[i].lstProp[tmpcolidx], mapi, mbat);

				tmpbat = cstablestat->lstcstableEx[i].colBats[j];
				isRightPropBAT(tmpbat);
				mvc_create_column(m, cstablesEx[i], tmpcolname,  &tpes[tmpbat->ttype]);				
			}
			totalNumNonDefCols += cstablestat->lstcstableEx[i].numCol;
		}
		#endif
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
		for (colId = 0; colId < cstablestat->numPropPerTable[i]; colId++){
			j = colOrder[i][colId]; 
		//for (j = 0; j < cstablestat->numPropPerTable[i]; j++){

			//TODO: Use propertyId from Propstat
			getColSQLname(tmpcolname, j, -1, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);

			tmpbat = cstablestat->lstcstable[i].colBats[j];
			isRightPropBAT(tmpbat);
			//printf("Column %d of tableId %d: Batid is %d \n",j, i, tmpbat->batCacheid); 
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
				isRightPropBAT(tmpbat);
				store_funcs.append_col(m->session->tr,
					mvc_bind_column(m, csmvtables[i][j],tmpcolname), 
					tmpbat, TYPE_bat);

				//One column for subj Bat
				sprintf(tmpcolname, "mvsubj");
				tmpbat = cstablestat->lstcstable[i].lstMVTables[j].subjBat;
				isRightPropBAT(tmpbat);
				store_funcs.append_col(m->session->tr,
					mvc_bind_column(m, csmvtables[i][j],tmpcolname), 
					tmpbat, TYPE_bat);

				//Value columns
				for (k = 0; k < tmpNumMVCols; k++){
					#if STORE_ALL_EXCEPTION_IN_PSO == 1
					if (k > 0) continue; //Only keep the first default type column as other columns will be stored in pso
					#endif
					getColSQLname(tmpmvcolname, j, k, cstablestat->lstcstable[i].lstProp[j], mapi, mbat);

					tmpbat = cstablestat->lstcstable[i].lstMVTables[j].mvBats[k];
					isRightPropBAT(tmpbat);
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
		#if STORE_ALL_EXCEPTION_IN_PSO == 0
		// Add non-default type table
		if (cstablestat->lstcstableEx[i].numCol != 0){	
			for (j = 0; j < cstablestat->lstcstableEx[i].numCol; j++){
				//TODO: Use propertyId from Propstat
				int tmpcolidx = cstablestat->lstcstableEx[i].mainTblColIdx[j];
				getColSQLname(tmpcolname, tmpcolidx, (int)(cstablestat->lstcstableEx[i].colTypes[j]), 
						cstablestat->lstcstable[i].lstProp[tmpcolidx], mapi, mbat);

				tmpbat = cstablestat->lstcstableEx[i].colBats[j];
				isRightPropBAT(tmpbat);
				//printf("ColumnEx %d: \n",j); 
				//BATprint(tmpbat);
				store_funcs.append_col(m->session->tr,
						mvc_bind_column(m, cstablesEx[i],tmpcolname ), 
						tmpbat, TYPE_bat);
			}
		}
		#endif
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

	printf("Generating script for FK creation ...");
	addPKandFKs(cstablestat, csPropTypes, *schema, mapi, mbat);
	printf("done\n");

	TKNZRclose(&ret);

	BBPunfix(sbat->batCacheid); 
	BBPunfix(pbat->batCacheid);
	BBPunfix(obat->batCacheid); 
	BBPunfix(mbat->batCacheid);
	for (i = 0; i < cstablestat->numTables; i++){
		free(csmvtables[i]);
	}
	free(csmvtables);
	freeColOrder(colOrder, cstablestat);
	freeCSPropTypes(csPropTypes,cstablestat->numTables);
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

#if 1
str
SQLrdfidtostr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	str msg; 
	mvc *m = NULL; 
	BAT *lmapBat = NULL, *rmapBat = NULL, *mBat = NULL; 
	bat lmapBatId, rmapBatId;
	str bnamelBat = "map_to_tknz_left";
	str bnamerBat = "map_to_tknz_right";
	char *schema = "rdf";
	sql_schema *sch;
	BUN pos; 
	oid *origId; 
	ObjectType objType; 
	oid *id = (oid *)getArgReference(stk,pci,1);
	str *ret = (str *) getArgReference(stk, pci, 0); 

	rethrow("sql.rdfidtostr", msg, getSQLContext(cntxt, mb, &m, NULL));
	
	if (*id == oid_nil){
		*ret = GDKstrdup(str_nil);
		return MAL_SUCCEED; 
	}

	objType = getObjType(*id);

	if (objType == STRING){
		str tmpObjStr;
		BATiter mapi; 
		if ((sch = mvc_bind_schema(m, schema)) == NULL)
			throw(SQL, "sql.rdfShred", "3F000!schema missing");

		mBat = mvc_bind(m, schema, "map0", "lexical",0);
		mapi = bat_iterator(mBat); 

		pos = (*id) - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the position of the string in the map bat */
		tmpObjStr = (str) BUNtail(mapi, BUNfirst(mBat) + pos);

		*ret = GDKstrdup(tmpObjStr);

	}
	else if (objType == URI || objType == BLANKNODE){
		lmapBatId = BBPindex(bnamelBat);
		rmapBatId = BBPindex(bnamerBat);

		if (lmapBatId == 0 || rmapBatId == 0){
			throw(SQL, "sql.SQLrdfidtostr", "The lmap/rmap Bats should be built already");
		}
		
		if ((lmapBat= BATdescriptor(lmapBatId)) == NULL) {
			throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
		}

		if ((rmapBat= BATdescriptor(rmapBatId)) == NULL) {
			throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
		}

		pos = BUNfnd(lmapBat,id);
		if (pos == BUN_NONE)	//this id is not converted to a new id
			origId = id; 
		else
			origId = (oid *) Tloc(rmapBat, pos);
		
		/*First convert the id to the original tokenizer odi */
		rethrow("sql.rdfidtostr", msg, takeOid(*origId, ret));
	} else {
		//throw(SQL, "sql.SQLrdfidtostr", "This Id cannot convert to str");
		getStringFormatValueFromOid(*id, objType, ret);  
	}

	if (msg != MAL_SUCCEED){
		throw(SQL, "sql.SQLrdfidtostr", "Problem in retrieving str from oid");
	}

	return msg; 
}


str
SQLrdfidtostr_bat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	str msg; 
	mvc *m = NULL; 
	BAT *lmapBat = NULL, *rmapBat = NULL, *mBat = NULL; 
	bat lmapBatId, rmapBatId;
	str bnamelBat = "map_to_tknz_left";
	str bnamerBat = "map_to_tknz_right";
	char *schema = "rdf";
	sql_schema *sch;
	BUN pos; 
	oid *origId; 
	ObjectType objType; 
	str tmpObjStr;
	BATiter mapi; 
	BAT *srcBat = NULL, *desBat = NULL; 
	BATiter srci; 
	BUN p, q; 
	bat *srcbid, *desbid; 
	oid *id; 
	str s; 
	srcbid = (bat *)getArgReference(stk,pci,1);
	desbid = (bat *) getArgReference(stk, pci, 0); 

	rethrow("sql.rdfidtostr", msg, getSQLContext(cntxt, mb, &m, NULL));
	
	if ((srcBat = BATdescriptor(*srcbid)) == NULL){
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}
	srci = bat_iterator(srcBat); 
	
	desBat = BATnew(TYPE_void, TYPE_str, BATcount(srcBat) + 1, TRANSIENT);
	BATseqbase(desBat, 0);
	
	/* Init the BATs for looking up the URIs*/
	lmapBatId = BBPindex(bnamelBat);
	rmapBatId = BBPindex(bnamerBat);

	if (lmapBatId == 0 || rmapBatId == 0){
		throw(SQL, "sqlbat.SQLrdfidtostr_bat", "The lmap/rmap Bats should be built already");
	}
	
	if ((lmapBat= BATdescriptor(lmapBatId)) == NULL) {
		throw(MAL, "sqlbat.SQLrdfidtostr_bat", RUNTIME_OBJECT_MISSING);
	}

	if ((rmapBat= BATdescriptor(rmapBatId)) == NULL) {
		throw(MAL, "sqlbat.SQLrdfidtostr_bat", RUNTIME_OBJECT_MISSING);
	}

	/* Init the map BAT for looking up the literal values*/
	if ((sch = mvc_bind_schema(m, schema)) == NULL)
		throw(SQL, "sql.rdfShred", "3F000!schema missing");

	mBat = mvc_bind(m, schema, "map0", "lexical",0);
	mapi = bat_iterator(mBat); 


	BATloop(srcBat, p, q){
		id = (oid *)BUNtail(srci, p);
		if (*id == oid_nil){
			BUNappend(desBat, str_nil, TRUE);
			continue; 
		}
		objType = getObjType(*id);

		if (objType == STRING){

			pos = (*id) - (objType*2 + 1) *  RDF_MIN_LITERAL;   /* Get the position of the string in the map bat */
			tmpObjStr = (str) BUNtail(mapi, BUNfirst(mBat) + pos);

			s = GDKstrdup(tmpObjStr);
		}
		else if (objType == URI || objType == BLANKNODE){

			pos = BUNfnd(lmapBat,id);
			if (pos == BUN_NONE)	//this id is not converted to a new id
				origId = id; 
			else
				origId = (oid *) Tloc(rmapBat, pos);
			
			/*First convert the id to the original tokenizer odi */
			rethrow("sql.rdfidtostr", msg, takeOid(*origId, &s));
		} else {
			//throw(SQL, "sql.SQLrdfidtostr", "This Id cannot convert to str");
			getStringFormatValueFromOid(*id, objType, &s);  
		}


		if (msg != MAL_SUCCEED){
			throw(SQL, "sql.SQLrdfidtostr", "Problem in retrieving str from oid");
		}

		//Append to desBAT
		BUNappend(desBat, s, TRUE);

	}
	
	*desbid = desBat->batCacheid;
	BBPkeepref(*desbid);
	
	BBPunfix(lmapBat->batCacheid);
	BBPunfix(rmapBat->batCacheid);
	BBPunfix(mBat->batCacheid);

	return msg; 
}

#else

str
SQLrdfidtostr(str *ret, oid *id){
	str msg; 
	BAT *lmapBat = NULL, *rmapBat = NULL; 
	bat lmapBatId, rmapBatId;
	str bnamelBat = "map_to_tknz_left";
	str bnamerBat = "map_to_tknz_right";
	BUN pos; 
	oid *origId; 

	lmapBatId = BBPindex(bnamelBat);
	rmapBatId = BBPindex(bnamerBat);

	if (lmapBatId == 0 || rmapBatId == 0){
		throw(SQL, "sql.SQLrdfidtostr", "The lmap/rmap Bats should be built already");
	}
	
	if ((lmapBat= BATdescriptor(lmapBatId)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	if ((rmapBat= BATdescriptor(rmapBatId)) == NULL) {
		throw(MAL, "rdf.RDFreorganize", RUNTIME_OBJECT_MISSING);
	}

	pos = BUNfnd(lmapBat,id);
	if (pos == BUN_NONE)	//this id is not converted to a new id
		origId = id; 
	else
		origId = (oid *) Tloc(rmapBat, pos);
	
	rethrow("SQLrdfidtostr", msg, takeOid(*origId, ret));

	//printf("String for "BUNFMT" is: %s \n",*id, *ret); 

	return msg; 
}
#endif

#if 0
str
SQLrdfstrtoid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	str msg; 
	mvc *m = NULL; 
	BAT *mapBat = NULL; 
	bat mapBatId;
	str bnameBat = "tknzr_to_map";
	oid *origId = NULL; 
	oid *id = NULL; 
	oid *ret = getArgReference_oid(stk, pci, 0);
	str *s = (str *)getArgReference(stk,pci,1);

	rethrow("sql.rdfstrtoid", msg, getSQLContext(cntxt, mb, &m, NULL));

	mapBatId = BBPindex(bnameBat);

	if ((mapBat= BATdescriptor(mapBatId)) == NULL) {
		throw(MAL, "SQLrdfstrtoid", RUNTIME_OBJECT_MISSING);
	}
	
	VALset(getArgReference(stk, pci, 1), TYPE_str, *s);

	rethrow("sql.rdfstrtoid", msg, TKNZRlocate(cntxt,mb,stk,pci));

	if (msg != MAL_SUCCEED){
		throw(SQL, "SQLrdfstrtoid", "Problem in locating string: %s\n", msg);
	}

	origId = (oid *) getArgReference(stk, pci, 0);
	
	if (*origId == oid_nil){
		throw(SQL, "SQLrdfstrtoid","String %s is not stored", *s);
	}


	id = (oid *) Tloc(mapBat, *origId); 

	if (id != NULL){
		*ret = *id; 
	}else{
		*ret = BUN_NONE; 
		throw(SQL, "SQLrdfstrtoid","No Id found for string %s", *s);
	}

	
	return msg; 
}

#else

str
SQLrdfstrtoid(oid *ret, str *s){
	str msg; 
	BAT *mapBat = NULL; 
	bat mapBatId;
	str bnameBat = "tknzr_to_map";
	oid origId; 
	oid *id; 

	//printf("Get the encoded id for the string %s\n", *s); 

	mapBatId = BBPindex(bnameBat);

	if ((mapBat= BATdescriptor(mapBatId)) == NULL) {
		throw(MAL, "SQLrdfstrtoid", RUNTIME_OBJECT_MISSING);
	}
	
	rethrow("sql.rdfstrtoid", msg, TKNRstringToOid(&origId, s));

	if (msg != MAL_SUCCEED){
		throw(SQL, "SQLrdfstrtoid", "Problem in locating string: %s\n", msg);
	}

	if (origId == oid_nil){
		throw(SQL, "SQLrdfstrtoid","String %s is not stored", *s); 
	}

	id = (oid *) Tloc(mapBat, origId); 

	if (id == NULL){
		*ret = BUN_NONE; 
		throw(SQL, "SQLrdfstrtoid","No Id found for string %s", *s); 
	}
	else
		*ret = *id; 
	
	return msg; 
}

str 
SQLrdftimetoid(oid *ret, str *datetime){
	
	lng tmp = BUN_NONE; 
	ValRecord vrec;

	//printf("SQLrdftimetoid: %s\n", *datetime);

	convertDateTimeToLong(*datetime, &tmp); 

	VALset(&vrec,TYPE_lng, &tmp);

	encodeValueInOid(&vrec, DATETIME, ret);

	return MAL_SUCCEED; 
}

#endif 

str 
SQLrdfScan_old(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	str msg; 
	int ret; 
	mvc *m = NULL; 

	str *params = (str *)getArgReference(stk,pci,1);	
	str *schema = (str *)getArgReference(stk,pci,2);	

	rethrow("sql.rdfScan", msg, getSQLContext(cntxt, mb, &m, NULL));

	rethrow("sql.rdfScan", msg, RDFscan_old(*params, *schema));

	(void) ret; 

	return MAL_SUCCEED; 
}

/*
 * This is the same as the function in sql.c
 * */
static void
bat2return(MalStkPtr stk, InstrPtr pci, BAT **b)
{
	int i;

	for (i = 0; i < pci->retc; i++) {
		*getArgReference_bat(stk, pci, i) = b[i]->batCacheid;
		BBPkeepref(b[i]->batCacheid);
	}
}

static 
void setBasicProps(BAT *b){
	b->hdense = 1;
	b->hseqbase = 0; 
	b->hsorted = 1; 
}

#if RDF_HANDLING_EXCEPTION_POSSIBLE_TBL_OPT
/*
 * Input: 
 * 	sbat: BAT of subjects
 * 	obat: BAT of objects
 * 	tblCand: Set of possible matching table Ids
 * 	nt: number of possible matching table
 * Output: 
 * 		
 * */
static void refine_BAT_with_possible_tblId(BAT *sbat, BAT *obat, BAT **retsbat, BAT **retobat, int *tblCand, int nt){
	oid *sbatpt = (oid *) Tloc(sbat, BUNfirst(sbat));				
	oid *obatpt = (oid *) Tloc(obat, BUNfirst(obat)); 
	int cnt = BATcount(sbat); 
	int i = 0; 
	int curId = -1; 	//Cur possible tblId
	int curpos = -1; 	//Cur possition in the tblCand
	BAT *r_sbat = NULL; 
	BAT *r_obat = NULL; 

	r_sbat = BATnew(TYPE_void, TYPE_oid, cnt, TRANSIENT); 
	setBasicProps(r_sbat); 
	r_sbat->tsorted = 1; 

	r_obat = BATnew(TYPE_void, TYPE_oid, cnt, TRANSIENT);

	for (i = 0; i < cnt; i++){
		//Check one by one from sbat and obat and then write to the output BAT
		oid tmps = sbatpt[i]; 
		oid tmpo = obatpt[i]; 
		int tmptbid = -1; 
		tmptbid = getTblId_from_S_simple(tmps); 
			
		while (tmptbid > curId){
			curpos++; 
			if (curpos >= nt) break; 
			curId = tblCand[curpos];
		}

		if (curpos >= nt) break; 

		if (tmptbid == curId){	
			//This row is possible candidate. Write tmps, tmpo to the ret BATs 	
			bunfastapp(r_sbat, &tmps);
			
			bunfastapp(r_obat, &tmpo);
		} else {
			continue; 
		}
	}
	
	*retsbat = r_sbat; 
	*retobat = r_obat; 
  	
	return; 

   bunins_failed:
	fprintf(stderr, "refine_BAT_with_possible_tblId: Failed in fast inserting\n");

}
#endif

static
void get_full_outerjoin_p_slices(oid *lstprops, int nrp, int np, oid *los, oid *his, BAT *full_obat, BAT *full_sbat, BAT **r_sbat, BAT ***r_obats){

	BAT **obats, **sbats; 
	int i; 
	clock_t start, end;
	#if RDF_HANDLING_EXCEPTION_POSSIBLE_TBL_OPT
	int num_match_tbl = 0; 
	int *regtblIds = NULL; 
	#endif

	(void) nrp; 
	obats = (BAT**)malloc(sizeof(BAT*) * np);
	sbats = (BAT**)malloc(sizeof(BAT*) * np);
	(*r_obats) = (BAT**)malloc(sizeof(BAT*) * np);

	#if RDF_HANDLING_EXCEPTION_POSSIBLE_TBL_OPT	
	get_possible_matching_tbl_from_RPs(&regtblIds, &num_match_tbl, lstprops, nrp, BUN_NONE);
	
	#if PRINT_FOR_DEBUG
	printf("Exception handling: Possible matching regular table [ ");
	for (i = 0; i < num_match_tbl; i++){
		printf(" %d ", regtblIds[i]); 
	}
	printf(" ]\n"); 
	#endif
	#endif

	for (i = 0; i < np; i++){
		BAT *tmpobat = NULL; 
		BAT *tmpsbat = NULL; 
		start = clock(); 
		#if PRINT_FOR_DEBUG
		printf("Slides of P = "BUNFMT " with o constraints from "BUNFMT" to " BUNFMT"\n", lstprops[i], los[i], his[i]);
		#endif
		getSlides_per_P(pso_propstat, &(lstprops[i]), los[i], his[i], full_obat, full_sbat, &tmpobat, &tmpsbat); 
		end = clock(); 
	
		#if PRINT_FOR_DEBUG
		printf(" [Took %f seconds)\n",((float)(end - start))/CLOCKS_PER_SEC); 
		#endif
	
		#if RDF_HANDLING_EXCEPTION_POSSIBLE_TBL_OPT
		refine_BAT_with_possible_tblId(tmpsbat, tmpobat, &(sbats[i]), &(obats[i]), regtblIds, num_match_tbl);
		#else
		sbats[i] = tmpsbat; 
		obats[i] = tmpobat;
		#endif
		
		#if PRINT_FOR_DEBUG
		if (sbats[i]){
			printf("   contains "BUNFMT " rows in sbat\n", BATcount(sbats[i]));
			//BATprint(sbats[i]);
			if (BATcount(sbats[i]) < 100){
				BATprint(sbats[i]);
			}
		}
		if (obats[i]){ 
			printf("   contains "BUNFMT " rows in obat\n", BATcount(obats[i]));
			if (BATcount(obats[i]) < 100){
				BATprint(obats[i]);
			}
		}
		#endif
	}

	start = clock(); 
	RDFmultiway_merge_outerjoins(np, sbats, obats, r_sbat, (*r_obats));
	end = clock(); 

	printf("Mutliway outer join result: ("BUNFMT" rows)  [Took %f seconds]\n", BATcount(*r_sbat), ((float)(end - start))/CLOCKS_PER_SEC);
	//BATprint(*r_sbat); 
	/*
	for (i = 0; i < np; i++){
		BATprint((*r_obats)[i]); 
	}
	*/
}

static void appendResult(BAT **r_obats, oid *tmpres, int np, oid sbt, int n_exp_value){
	int j = 0; 
	(void) sbt; 
	if (n_exp_value == 0) return; 

	for (j = 0; j < np; j++){
		//Output result
		BUNappend(r_obats[j], &(tmpres[j]), TRUE); 
	}
}
/*
static
void fetch_result(BAT **r_obats, oid **obatCursors, int pos, oid **regular_obat_cursors, oid **regular_obat_mv_cursors, BAT **regular_obats, BAT **regular_obat_mv, oid sbt, oid tmpS, int cur_p, int nrp, int np, oid *tmpres, int n_exp_value){
	if (obatCursors[cur_p][pos] == oid_nil){
		//Look for the result from regular bat. 
		//Check if the regular bat is pointing to a MVBat
		//Then, get all teh value from MVBATs

		assert(cur_p >= nrp || regular_obat_cursors[cur_p][tmpS] != oid_nil); 

		if (regular_obat_mv_cursors[cur_p] != NULL){		//mv col
			//Get the values from mvBat
			oid offset = regular_obat_cursors[cur_p][tmpS]; 
			oid nextoffset; 
			int numCand, i; 
			int nextS = tmpS + 1;
			int batCnt = regular_obats[cur_p]->batCount;
	
			//There can be oid_nil in the o value for the next subject
			while (nextS < batCnt){
				if (regular_obat_cursors[cur_p][nextS] == oid_nil){
					nextS++; 
				} else {
					nextoffset = regular_obat_cursors[cur_p][nextS]; 
					numCand = nextoffset - offset;
					break; 
				}
			}

			if (nextS == batCnt) {
				numCand = BUNlast(regular_obat_mv[cur_p]) - offset;
			}
			assert(numCand >= 0); 
			for (i = 0; i < numCand; i++){
				tmpres[cur_p] = regular_obat_mv_cursors[cur_p][offset + i]; 

				if (cur_p < (np -1))
					fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value); 

				else if (cur_p == (np - 1)){
					appendResult(r_obats, tmpres, np, sbt, n_exp_value); 
				}
			}
					
		}
		else{
			if (regular_obat_cursors[cur_p] != NULL) tmpres[cur_p] = regular_obat_cursors[cur_p][tmpS];
			else tmpres[cur_p] = oid_nil; 
			
			if (cur_p < (np -1))
				fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value); 

			else if (cur_p == (np - 1)){
				//Output result
				appendResult(r_obats, tmpres, np, sbt, n_exp_value); 
			}
		}

	}
	else{	
		//The result can also come from regular table
		//Case 1: When the property has multi object values
		//but we keep it as single-valued prop by putting some values to 
		//pso
		//Case 2: Some triples is moved to pso to keep FK relationship
		//so, there can also be regular  mv col
		if (regular_obat_cursors[cur_p] != NULL && regular_obat_cursors[cur_p][tmpS] != oid_nil){
			tmpres[cur_p] = regular_obat_cursors[cur_p][tmpS];
			if (cur_p < (np -1))
				fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value);	
			else if (cur_p == (np -1))
				appendResult(r_obats, tmpres, np, sbt, n_exp_value);

		}

		tmpres[cur_p] = obatCursors[cur_p][pos];		
	
		if (cur_p < (np -1))
			fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value + 1); 

		else if (cur_p == (np - 1)){
			//Output result
			appendResult(r_obats, tmpres, np, sbt, n_exp_value + 1); 		
		}
	}


}

*/


static
void fetch_result(BAT **r_obats, oid **obatCursors, int pos, oid **regular_obat_cursors, oid **regular_obat_mv_cursors, BAT **regular_obats, BAT **regular_obat_mv, oid sbt, oid tmpS, int cur_p, int nrp, int np, oid *tmpres, int n_exp_value){
		//Look for the result from regular bat. 
		//Check if the regular bat is pointing to a MVBat
		//Then, get all teh value from MVBATs

		//assert(cur_p >= nrp || regular_obat_cursors[cur_p][tmpS] != oid_nil); 
		int hasvalue = 0;

	        tmpres[cur_p] = oid_nil; 

		if (regular_obat_mv_cursors[cur_p] != NULL && regular_obat_cursors[cur_p][tmpS] != oid_nil){		//mv col
			//Get the values from mvBat
			oid offset = regular_obat_cursors[cur_p][tmpS]; 
			oid nextoffset; 
			int numCand = 0, i; 
			int nextS = tmpS + 1;
			int batCnt = regular_obats[cur_p]->batCount;
	
			//There can be oid_nil in the o value for the next subject
			while (nextS < batCnt){
				if (regular_obat_cursors[cur_p][nextS] == oid_nil){
					nextS++; 
				} else {
					nextoffset = regular_obat_cursors[cur_p][nextS]; 
					numCand = nextoffset - offset;
					break; 
				}
			}

			if (nextS == batCnt) {
				numCand = BUNlast(regular_obat_mv[cur_p]) - offset;
			}
			assert(numCand > 0); 
			for (i = 0; i < numCand; i++){
				tmpres[cur_p] = regular_obat_mv_cursors[cur_p][offset + i]; 

				if (cur_p < (np -1))
					fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value); 

				else if (cur_p == (np - 1)){
					appendResult(r_obats, tmpres, np, sbt, n_exp_value); 
				}
			}
			hasvalue = 1;
					
		}
		else if (regular_obat_cursors[cur_p] != NULL && regular_obat_cursors[cur_p][tmpS] != oid_nil){ 
			
			tmpres[cur_p] = regular_obat_cursors[cur_p][tmpS];
			
			if (cur_p < (np -1))
				fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value); 

			else if (cur_p == (np - 1)){
				//Output result
				appendResult(r_obats, tmpres, np, sbt, n_exp_value); 
			}

			hasvalue = 1;
		}

		
		if (obatCursors[cur_p][pos] != oid_nil){

			tmpres[cur_p] = obatCursors[cur_p][pos];		
	
			if (cur_p < (np -1))
				fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value + 1); 

			else if (cur_p == (np - 1)){
				//Output result
				appendResult(r_obats, tmpres, np, sbt, n_exp_value + 1); 		
			}
			hasvalue = 1; 
		}


		if (hasvalue == 0){
		
			if (cur_p < (np -1))
				fetch_result(r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, cur_p + 1, nrp, np, tmpres, n_exp_value); 

			else if (cur_p == (np - 1))
				//Output result
				appendResult(r_obats, tmpres, np, sbt, n_exp_value); 		
		
		}


}

/*
 * Combine exceptioins and regular tables
 * */

static
void combine_exception_and_regular_tables(mvc *c, BAT **r_sbat, BAT ***r_obats, BAT *sbat, BAT **obats, oid *lstProps, int nP, int nRP){
	oid *sbatCursor; 
	oid **obatCursors; 
	int i, j, pos; 
	int numS; 
	char *schema = "rdf";
	int curtid = -1; 
	BAT **regular_obats = NULL; 
	BAT **regular_obat_mv = NULL; 
	oid **regular_obat_cursors = NULL; 
	oid **regular_obat_mv_cursors = NULL; 	//If this column is MV col, then store the point to its MV BAT
	int accept = 0; 
	#if RDF_HANDLING_EXCEPTION_MISSINGPROP_OPT
	int *lst_missing_props = NULL; 	//Index of missing prop in the lstProp
	int num_mp = 0; 	//Number of missing prop
	#endif
	
	(void) r_sbat; 
	(void) r_obats; 
	(void) nRP; 

	//Init return BATs
	*r_sbat = BATnew(TYPE_void, TYPE_oid, BATcount(sbat), TRANSIENT); 
	setBasicProps(*r_sbat); 
	*r_obats = (BAT **) malloc(sizeof(BAT*) * nP); 
	for (i = 0; i < nP; i++){
		(*r_obats)[i] = BATnew(TYPE_void, TYPE_oid, BATcount(sbat), TRANSIENT); 
		setBasicProps((*r_obats)[i]); 
	}
	
	#if RDF_HANDLING_EXCEPTION_MISSINGPROP_OPT
	lst_missing_props = (int *) malloc(sizeof(int) * nP); 
	num_mp = 0; 
	#endif

	
	sbatCursor = (oid *) Tloc(sbat, BUNfirst(sbat));
	obatCursors = (oid **) malloc(sizeof(oid*) * nP); 
	
	regular_obats = (BAT **) malloc(sizeof(BAT *) * nP); 
	regular_obat_mv = (BAT **) malloc(sizeof(BAT *) * nP); 

	regular_obat_cursors = (oid **) malloc(sizeof(oid*) * nP); 
	regular_obat_mv_cursors = (oid **) malloc(sizeof(oid*) * nP); 
	for (i = 0; i < nP; i++){
		obatCursors[i] = (oid *) Tloc(obats[i], BUNfirst(obats[i]));
		assert (BATcount(obats[i]) == BATcount(sbat)); 
		regular_obats[i] = NULL; 
		regular_obat_mv[i] = NULL; 
		regular_obat_cursors[i] = NULL; 
		regular_obat_mv_cursors[i] = NULL;
	}



	
	numS = BATcount(sbat); 

	for (pos = 0; pos < numS; pos++){
		oid sbt = sbatCursor[pos]; 
		int tid = -1; 
	 	oid tmpS = BUN_NONE; 

		/*
		if (sbt == (oid)879609302220975){
			printf("[DEBUG] FOUND THAT SUBJECT "BUNFMT " HERE\n", sbt);
		}
		*/
		getTblIdxFromS(sbt, &tid, &tmpS);
		if (tid != curtid){
			curtid = tid; 
			#if RDF_HANDLING_EXCEPTION_MISSINGPROP_OPT
			num_mp = 0; 
			#endif
			//reload BATs for that table
			for (j = 0;  j < nP; j++){
				str tmpColname, tmptblname, tmpmvtblname, tmpmvdefcolname;
				int colIdx = getColIdx_from_oid(tid, global_csset, lstProps[j]);
				if (colIdx == -1) {
					regular_obats[j] = NULL; 
					regular_obat_mv[j] = NULL; 
					regular_obat_cursors[j] = NULL; 
					regular_obat_mv_cursors[j] = NULL; 
					#if RDF_HANDLING_EXCEPTION_MISSINGPROP_OPT
					if (j < nRP){ 
						lst_missing_props[num_mp] = j; 
						num_mp++; 
					}
					#endif
					continue; 
				}

				tmpColname = getColumnName(global_csset, tid, colIdx);
				tmptblname = (global_csset->items[tid])->tblsname;
				tmpmvtblname = (global_csset->items[tid])->lstmvtblname[colIdx]; 
				tmpmvdefcolname = (global_csset->items[tid])->lstmvdefaultcolname[colIdx];

				//Unfix old one
				if (regular_obats[j]) {
					BBPunfix(regular_obats[j]->batCacheid); 
					regular_obats[j] = NULL; 
				}

				if (regular_obat_mv[j]){
					BBPunfix(regular_obat_mv[j]->batCacheid);
					regular_obat_mv[j] = NULL; 
				}

				regular_obats[j] = mvc_bind(c, schema, tmptblname, tmpColname, 0);
				assert(regular_obats[j] != NULL); 
				regular_obat_cursors[j] = (oid *) Tloc(regular_obats[j], BUNfirst(regular_obats[j]));

				if (isMVCol(tid, colIdx, global_csset)){
					regular_obat_mv[j] = mvc_bind(c, schema, tmpmvtblname, tmpmvdefcolname, 0);
					regular_obat_mv_cursors[j] = (oid *) Tloc(regular_obat_mv[j], BUNfirst(regular_obat_mv[j]));
				} else {
					regular_obat_mv[j] = NULL; 
					regular_obat_mv_cursors[j] = NULL; 
				}
				 
				
			}
		}


		accept = 1; 
		#if RDF_HANDLING_EXCEPTION_MISSINGPROP_OPT
		for (j = 0; j < num_mp; j++){
			if (obatCursors[lst_missing_props[j]][pos] == oid_nil){
				accept = 0;
				break; 
			}	
		}
		if (accept == 0) continue; 
		#endif
		for (j = 0;  j < nRP; j++){
			if (obatCursors[j][pos] == oid_nil){
				if (regular_obat_cursors[j] == NULL){	//No corresponding regular column
					accept = 0; 
					break; 			
				}
				//Look for the value from main table
				if (regular_obat_cursors[j][tmpS] == oid_nil) {
					//TODO: Continue if this [j] is optional prop
					accept = 0;
					break; 
				}
			}	
		}


		//printf("At row "BUNFMT" of table %d for sbt "BUNFMT"...", tmpS, tid, sbt); 
		
		if (sbt == (oid)1460151441687511 && accept == 1){
			printf("[DEBUG2] THAT SUBJECT "BUNFMT " IS ACCEPTED\n",sbt);
		}
		

		if (accept == 1){	//Accept, can insert to the output bat			
			oid *tmpres = (oid *) malloc(sizeof(oid) * nP); 
			oid r_obat_oldsize = BATcount((*r_obats)[0]); 
			oid r_obat_newsize = BUN_NONE; 
			int n_exp_value = 0; 	//Count the number of exception value
			//printf("Accepted\n"); 
			for (j = 0; j < nP; j++){
				tmpres[j] = oid_nil; 
			}

			fetch_result(*r_obats, obatCursors, pos, regular_obat_cursors, regular_obat_mv_cursors, regular_obats, regular_obat_mv, sbt, tmpS, 0, nRP, nP, tmpres, n_exp_value);
			r_obat_newsize = BATcount((*r_obats)[0]); 
			for (j = 0; j < (int)(r_obat_newsize - r_obat_oldsize); j++){
				BUNappend(*r_sbat, &sbt, TRUE); 
			}
		} else {
			//printf("Rejected\n");
		}


	}


				
	//free
	for (i = 0; i < nP; i++){
		if (regular_obats[i]) BBPunfix(regular_obats[i]->batCacheid);
	}
	free(regular_obats); 
	free(regular_obat_cursors); 
}

#if PRINT_FOR_DEBUG
static void
BATprint_topn(BAT *b, int n){
	BAT *tmp = NULL; 
	if (BATcount(b) > (oid) n){
		tmp = BATslice(b, 0, n);
	} else {
		tmp = BATslice(b, 0, BATcount(b) - 1); 
	}

	BATprint(tmp); 

	BBPunfix(tmp->batCacheid); 
}
#endif

/*
 * The input for this pattern should be
 * Number of Ps, Number of RPs, <List of Prop Ids>
 * */
str 
SQLrdfScan(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	
	BAT **b = NULL;		//List of BATs (columns) to be returned
	int nRet = -1; 		//Number of BATs to return
	int *nP = NULL; 	//Number of props
	int *nRP = NULL; 	
	oid *lstProps = NULL; 
	
	//Constraints for o values
	oid *los = NULL; 
	oid *his = NULL; 

	int i;
	//int *lstbattypes = NULL; 
	clock_t sT; 
	clock_t eT; 

	(void) cntxt; 
	(void) mb; 
	(void) stk; 
	(void) pci; 
		
	sT = clock(); 

	nP = (int *) getArgReference(stk, pci, pci->retc + 0);
	nRP =  (int *) getArgReference(stk, pci, pci->retc + 1);
	nRet = 2 * (*nP); 
	
	(void) nRP;
	(void) nRet; 
	
	assert (pci->retc == nRet);
		
	b = (BAT **) GDKmalloc (sizeof (BAT*) * pci->retc); 
	lstProps = (oid *)GDKmalloc(sizeof(oid) * (*nP)); 
	los = (oid *)GDKmalloc(sizeof(oid) * (*nP));
	his = (oid *)GDKmalloc(sizeof(oid) * (*nP));

	for (i = 0; i < (*nP); i++){
		oid *tmpp = (oid *) getArgReference(stk, pci, pci->retc + 2 + i);
		oid *lo = (oid *) getArgReference(stk, pci, pci->retc + 2 + (*nP) + i);
		oid *hi = (oid *) getArgReference(stk, pci, pci->retc + 2 + 2 * (*nP) + i);

		lstProps[i] = *tmpp; 
		los[i] = *lo; 
		his[i] = *hi; 
	}
	
	#if PRINT_FOR_DEBUG
	for (i = 0; i < pci->retc; i++){
		//int tmp = -1; 
		//Get type from pci
		int bat_type = 	ATOMstorage(getColumnType(getArgType(mb,pci,i))); 
		
		if (bat_type == TYPE_str) printf("bat_type is string\n");
		else printf("bat_type is %d\n", bat_type); 

	}
	
	printf("There are %d props, among them %d RPs \n", *nP, *nRP);
	#endif
	
	//Step 1. "Full outer join" to get all the possible combination
	//of all props from PSO table
	{
		BAT *r_sbat, **r_obats; 
		BAT *m_sbat, **m_obats; 
		char *schema = "rdf"; 
		mvc *m = NULL;
		str msg; 
		BAT *pso_fullSbat = NULL, *pso_fullObat = NULL;
		clock_t sT1, eT1; 

		
		sT1 = clock(); 
		rethrow("sql.rdfShred", msg, getSQLContext(cntxt, mb, &m, NULL));

		pso_fullSbat = mvc_bind(m, schema, "pso", "s",0);
		pso_fullObat = mvc_bind(m, schema, "pso", "o",0);

		get_full_outerjoin_p_slices(lstProps, *nRP, *nP, los, his, pso_fullObat, pso_fullSbat, &r_sbat, &r_obats);

		eT1 = clock(); 
		printf("Step 1 in Handling exception took  %f seconds.\n", ((float)(eT1 - sT1))/CLOCKS_PER_SEC);
		
		/* {
		BUN testbun = BUN_NONE; 
		BUN testoid = 510173395288388;
		testbun = BUNfnd(r_sbat, &testoid); 
		if (testbun == BUN_NONE){
			printf("[DEBUG] That subject is not here\n");
		} else {
			printf("[DEBUG] The subject is found at " BUNFMT " position\n", testbun);
		}
		} */

		//Step 2. Merge exceptions with Tables
		sT1 = clock();
		
		combine_exception_and_regular_tables(m, &m_sbat, &m_obats, r_sbat, r_obats, lstProps, *nP, *nRP);


		#if PRINT_FOR_DEBUG
		printf("Combining exceptions and regular table returns "BUNFMT " rows\n", BATcount(m_sbat)); 

		BATprint_topn(m_sbat, 5); 
		for (i = 0; i < (*nP); i++){
			BATprint_topn(m_obats[i], 5); 
		}
		#endif

		//BATprint(m_sbat); 
		if (0)
		if ((*nP) > 2){
		
			oid *s_curs = (oid *) Tloc(m_sbat, BUNfirst(m_sbat));
			//oid *o_curs = (oid *) Tloc(m_obats[1], BUNfirst(m_obats[1])); 
			int j = 0; 
			for (j = 0; (oid)j < m_sbat->batCount; j++){
				if (s_curs[j] == 510173395288388){
					printf("[Debug] Having matching results\n"); 
				}
			}
		
		}

		for (i = 0; i < (*nP); i++){
			//BATprint(m_obats[i]);
			b[2*i] = COLcopy(m_sbat, m_sbat->ttype, FALSE, TRANSIENT);
			b[2*i+1] = COLcopy(m_obats[i], m_obats[i]->ttype, FALSE, TRANSIENT); 
		}
		
		eT1 = clock(); 
		printf("Step 2 in Handling exception took  %f seconds.\n", ((float)(eT1 - sT1))/CLOCKS_PER_SEC);

	}
	#if PRINT_FOR_DEBUG
	printf("Return the resusting BATs...");
	#endif
	bat2return(stk, pci, b);
	#if PRINT_FOR_DEBUG
	printf("... done\n"); 
	#endif
	GDKfree(b);

	eT = clock(); 
	printf("RDFscan for handling exception took  %f seconds.\n", ((float)(eT - sT))/CLOCKS_PER_SEC);

	return MAL_SUCCEED; 
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

static
void test_intersection(void){

	oid** listoid; 
	int* listcount; 
	oid* interlist; 
	int num = 4; 
	int internum = 0; 
	int i;
	oid list1[6] = {1, 3, 4, 5, 7, 11}; 
	oid list2[7] = {1, 3, 5, 7, 8, 9, 11};
	oid list3[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9 ,11};
	oid list4[5] = {2, 3, 7, 9, 11}; 

	listoid = (oid**)malloc(sizeof(oid*) * num); 
	listcount = (int *) malloc(sizeof(int) * num); 
	
	listoid[0] = list1; 
	listcount[0] = 6; 
	listoid[1] = list2;
	listcount[1] = 7;
	listoid[2] = list3; 
	listcount[2] = 10; 
	listoid[3] = list4;
	listcount[3] = 5;

	intersect_oidsets(listoid, listcount, num, &interlist, &internum);
	printf("Intersection list: \n"); 
	for (i = 0; i < internum; i++){
		printf(" " BUNFMT, interlist[i]);
	}
	printf("\n"); 
}

SimpleCSset 	*global_csset = NULL; 
PropStat 	*global_p_propstat = NULL; 
PropStat 	*global_c_propstat = NULL; 
BAT		*global_mbat = NULL;
BATiter 	global_mapi; 
PsoPropStat	*pso_propstat = NULL; /* Store the offsets of a prop 
					 in the exceptional PSO table */
int 		need_handling_exception = 1;

static
void getOffsets(PsoPropStat *pso_pstat, oid *p, BUN *start, BUN *end){

	BUN pos; 

	pos = BUNfnd(pso_pstat->pBat, p);
	if (pos == BUN_NONE){
		printf("The prop "BUNFMT " is not in PSO!\n", *p);
		*start = BUN_NONE; 
		*end = BUN_NONE; 
	}
	else{
		oid *tmpstart = (oid *) Tloc(pso_pstat->offsetBat, pos);
		*start = (*tmpstart); 
		if (pos == BATcount(pso_pstat->pBat) - 1){ //This is the last prop
			*end = BUN_NONE; 
		}
		else{
			oid *tmpend = (oid *) Tloc(pso_pstat->offsetBat, pos + 1);	
			*end = (*tmpend) - 1; 
		}

	}
}

/*
 * Get slide of PSO for an input P with lower bound constraint (lo_cst) and upper bound constraint (hi_cst)
 * */
void getSlides_per_P(PsoPropStat *pso_pstat, oid *p, oid lo_cst, oid hi_cst, BAT *obat, BAT *sbat, BAT **ret_oBat, BAT **ret_sBat){
	BUN l, h; 	
	getOffsets(pso_pstat, p, &l, &h); 

	(void) lo_cst;
	(void) hi_cst; 

	if (l != BUN_NONE){
		BAT *tmp_o = NULL, *tmp_s = NULL; 
		oid lo, hi; 
		BAT *tmpB = NULL;

		(void) lo; 
		(void) hi; 
		(void) tmpB; 

		tmp_o = BATslice(obat, l, h+1); 

		tmp_s = BATslice(sbat, l, h+1); 
		
		#if RDF_HANDLING_EXCEPTION_SELECTPUSHDOWN_OPT
			if (lo_cst == BUN_NONE && hi_cst == BUN_NONE){	//No constraint
				*ret_oBat =  tmp_o; 
				*ret_sBat = tmp_s; 
			} else {	
				if (lo_cst == BUN_NONE){
					lo = oid_nil; 
					hi = hi_cst; 
				} else if (hi_cst == BUN_NONE){
					lo = lo_cst; 
					hi = oid_nil; 
				
				} else {	//Have both lower and upper bounds
					lo = lo_cst; 
					hi = hi_cst; 
				}
				//BATselect(inputbat, <dont know yet>, lowValue, Highvalue, isIncludeLowValue, isIncludeHigh, <anti> 
				tmpB = BATselect(tmp_o, NULL, &lo, &hi, 1, 1, 0);
				*ret_oBat = BATproject(tmpB, tmp_o); 
				*ret_sBat = BATproject(tmpB, tmp_s); 
				BBPunfix(tmpB->batCacheid); 
			}
		#else
			*ret_oBat =  tmp_o; 
			*ret_sBat = tmp_s; 
		#endif
		(*ret_sBat)->tsorted = true;
	} else {
		//*ret_oBat = NULL;
		//*ret_sBat = NULL; 
		*ret_oBat = BATnew(TYPE_void, TYPE_oid, 0, TRANSIENT);
		*ret_sBat = BATnew(TYPE_void, TYPE_oid, 0, TRANSIENT);
	}

}

/*
 * From the full P BAT (column P) of the PSO
 * triple table, extract the offsets where each 
 * prop starts. 
 * */
static 
void build_PsoPropStat(BAT *full_pbat, int maxNumP, BAT *full_sbat, BAT *full_obat){
	BUN p, q; 
	oid *poid; 
	BATiter pi; 
	oid curP; 
	int batsize = 150000; 
	pi = bat_iterator(full_pbat); 

	(void) full_sbat; 
	(void) full_obat;

	batsize = maxNumP; 	//Can be smaller

	pso_propstat = (PsoPropStat *) GDKmalloc(sizeof(PsoPropStat)); 
	pso_propstat->pBat =  BATnew(TYPE_void, TYPE_oid, batsize, TRANSIENT);
	pso_propstat->pBat->tsorted = 1;
	pso_propstat->offsetBat = BATnew(TYPE_void, TYPE_oid, batsize, TRANSIENT); 
	
	curP = BUN_NONE; 
	BATloop(full_pbat, p, q){
		poid = (oid *) BUNtloc(pi, p); 
		if (*poid != curP){	//Start new poid
			oid tmpoffset = p; 
			BUNappend(pso_propstat->pBat, poid, TRUE); 
			BUNappend(pso_propstat->offsetBat, &tmpoffset, TRUE);

			curP = *poid; 	
		}	
	}	
	printf("Number of P in PSO is: "BUNFMT"\n", BATcount(pso_propstat->pBat)); 
	//BATprint(pso_propstat->pBat); 
	//BATprint(pso_propstat->offsetBat); 
	
}
				

str SQLrdfdeserialize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	
	SimpleCSset *csset; 

	printf("De-serialize simple CSset from Bats\n"); 

	csset = dumpBat_to_CSset(); 
		
	print_simpleCSset(csset); 

	test_intersection(); 

	free_simpleCSset(csset); 
	
	(void) cntxt; (void) mb; (void) stk; (void) pci;

	return MAL_SUCCEED; 
}

/*
 * Preparation for executing sparql queries
 * */

str SQLrdfprepare(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci){
	char *schema = "rdf"; 
	char *sschema = "sys";
	str msg; 
	mvc *m = NULL;
	int ret; 

	(void) ret;
	rethrow("sql.rdfShred", msg, getSQLContext(cntxt, mb, &m, NULL));
	
	//Load map bat
	printf("Load dictionary Bat\n");
	global_mbat = mvc_bind(m, schema, "map0", "lexical",0);
	global_mapi = bat_iterator(global_mbat);

	
	//Load ontologies
	
	printf("Load ontologies\n"); 
	{	
     		BAT *auri = mvc_bind(m, sschema, "ontattributes","muri",0);
     		BAT *aattr = mvc_bind(m, sschema, "ontattributes","mattr",0);
     		BAT *muri = mvc_bind(m, sschema, "ontmetadata","muri",0);
     		BAT *msuper = mvc_bind(m, sschema, "ontmetadata","msubclassof",0);
     		BAT *mlabel = mvc_bind(m, sschema, "ontmetadata","mlabel",0);

		RDFloadsqlontologies(&ret, &(auri->batCacheid), 
				&(aattr->batCacheid),
				&(muri->batCacheid),
				&(msuper->batCacheid),
				&(mlabel->batCacheid));

		BBPreclaim(auri);
		BBPreclaim(aattr);
		BBPreclaim(muri);
		BBPreclaim(msuper);
		BBPreclaim(mlabel); 

	}

	//Open Tokenizer
	printf("Open tokenizer with schema %s\n", schema); 
	if (TKNZRopen (NULL, &schema) != MAL_SUCCEED) {
		throw(RDF, "rdf.rdfschema",
		"could not open the tokenizer\n");
	}
	
	printf("Build global csset\n");
	//Build global cs set from persistent BATs
	global_csset = dumpBat_to_CSset(); 

	//Build propstat for props in final CSs
	global_p_propstat = getPropStat_P_simpleCSset(global_csset); 

	global_c_propstat = getPropStat_C_simpleCSset(global_csset); 

	//print_simpleCSset(global_csset);

	printf("Done preparation\n"); 
	
	#if HANDLING_EXCEPTION
	need_handling_exception = 1; 
	#else
	need_handling_exception = 0; 	
	#endif
	//Build pso_propstat to store offsets of each
	//P in PSO table
	{
		
		BAT *pso_fullPbat = mvc_bind(m, schema, "pso", "p",0);
		BAT *pso_fullSbat = mvc_bind(m, schema, "pso", "s",0);
		BAT *pso_fullObat = mvc_bind(m, schema, "pso", "o",0);
		build_PsoPropStat(pso_fullPbat, global_p_propstat->numAdded, pso_fullSbat, pso_fullObat); 
			
		if (BATcount(pso_fullSbat) == 0) {
			need_handling_exception = 0; 
			printf("Do not need to handle exception\n"); 
		}
		
	}
	

	/*
	{	//Test
		BAT *testBat = BATnew(TYPE_void, TYPE_str, 100, TRANSIENT); 
		str s1 = "a", s2 = "bbbb", s3 = "ccc"; 
		str sptr = NULL; 
		BATiter tmpiter; 
		int (*cmp)(const void *, const void *) = ATOMcompare(testBat->ttype);
		const void *nil = ATOMnilptr(testBat->ttype);

		BUNappend(testBat, s1, TRUE);
		BUNappend(testBat, s2, TRUE);	
		BUNappend(testBat, ATOMnilptr(testBat->ttype), TRUE);
		BUNappend(testBat, ATOMnilptr(testBat->ttype), TRUE);
		BUNappend(testBat, s3, TRUE);	
		BATprint(testBat); 
			
		tmpiter = bat_iterator(testBat);
		//Check data
		sptr = BUNtail(tmpiter, 3); 
		if (sptr == NULL) printf("NULL value \n");
		else if (sptr == str_nil) printf("STRING NULL\n"); 
		else printf("NONE OF THEM\n"); 
		
		if ((*cmp)(sptr, nil) == 0) printf("ATOM compare works\n"); 
	}
	*/

	(void) cntxt; (void) mb; (void) stk; (void) pci;
			
	return MAL_SUCCEED;
}

