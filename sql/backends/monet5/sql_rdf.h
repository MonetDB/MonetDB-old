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

/*
 * (author) M Kersten, N Nes
 * SQL support implementation
 * This module contains the wrappers around the SQL
 * multi-version-catalog and support routines.
 */
#ifndef _SQL_RDF_H
#define _SQL_RDF_H

#include <sql.h>

#ifdef HAVE_RAPTOR
# include <rdf.h>
# include <rdfschema.h>
#include <rdfretrieval.h>
#include <rdfscan.h>
#include <rdfdump.h>
#include "rdfcommon.h"
#endif

//sql5_export str SQLrdfdeserialize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfprepare(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfShred(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfreorganize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfRetrieveSubschema(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfdeserialize(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
//sql5_export str SQLrdfidtostr(str *ret, oid *id);

sql5_export str SQLrdfidtostr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfidtostr_bat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfstrtoid(oid *ret, str *s);


sql5_export str SQLrdf_convert_to_orig_oid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdf_convert_to_orig_oid_bat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

sql5_export str SQLrdftimetoid(oid *ret, str *dt);


//sql5_export str SQLrdfstrtoid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export void getTblSQLname(char *tmptbname, int tblIdx, int isExTbl, oid tblname, BATiter mapi, BAT *mbat);
sql5_export void getColSQLname(char *tmpcolname, int colIdx, int colType, oid propid, BATiter mapi, BAT *mbat);
sql5_export void getMvTblSQLname(char *tmpmvtbname, int tblIdx, int colIdx, oid tblname, oid propid, BATiter mapi, BAT *mbat);

sql5_export str SQLrdfScan_old(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str SQLrdfScan(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

sql5_export void getSlides_per_P(PsoPropStat *pso_pstat, oid *p, oid low, oid hi, BAT *obat, BAT *sbat, BAT **ret_oBat, BAT **ret_sBat); 

sql5_export void get_possible_matching_tbl_from_RPs(int **rettbId, int *num_match_tbl, oid *lstRP, int num, oid subj); 

extern SimpleCSset *global_csset; 
extern PropStat *global_p_propstat;
extern PropStat *global_c_propstat;
extern BAT *global_mbat;
extern BATiter global_mapi;
extern PsoPropStat *pso_propstat; 
extern int need_handling_exception; 

#define APPLY_OPTIMIZATION_FOR_OPTIONAL	1	/* Instead of using left join, we use a project with ifthenelse */
						/* on the set of optional columns */

#define HANDLING_EXCEPTION 1

#define RDF_HANDLING_EXCEPTION_MISSINGPROP_OPT 1

#define RDF_HANDLING_EXCEPTION_SELECTPUSHDOWN_OPT 1

#define RDF_HANDLING_EXCEPTION_POSSIBLE_TBL_OPT	1 /* Use the set of possible table for the set of required props to limit the number of matching subj Id */

#define PRINT_FOR_DEBUG 0

#define ONLY_COMPUTE_OPT_TIME 0

#endif /*_SQL_RDF_H */
