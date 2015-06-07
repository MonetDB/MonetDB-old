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

sql5_export void getTblSQLname(char *tmptbname, int tblIdx, int isExTbl, oid tblname, BATiter mapi, BAT *mbat);
sql5_export void getColSQLname(char *tmpcolname, int colIdx, int colType, oid propid, BATiter mapi, BAT *mbat);
sql5_export void getMvTblSQLname(char *tmpmvtbname, int tblIdx, int colIdx, oid tblname, oid propid, BATiter mapi, BAT *mbat);


extern SimpleCSset *global_csset; 
extern PropStat *global_p_propstat;
extern PropStat *global_c_propstat;
extern BAT *global_mbat;
extern BATiter global_mapi;

#define USINGRDFSCAN	0

#define APPLY_OPTIMIZATION_FOR_OPTIONAL	1	/* Instead of using left join, we use a project with ifthenelse */
						/* on the set of optional columns */

#endif /*_SQL_RDF_H */
