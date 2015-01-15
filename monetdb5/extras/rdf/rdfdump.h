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

#ifndef _RDFDUMP_H_
#define _RDFDUMP_H_

#include <sql_catalog.h>
#include "rdftypes.h"
#include "rdfschema.h"


typedef enum {
	csd_tblId,
	csd_tblname, 
	csd_csId, 
	csd_freq,
	csd_coverage, 
	csd_pOffset, 
	csd_fullP, 
	csd_cOffset,
	csd_fullC, 
	csd_cname
} csdumBatType;

#define N_CSDUM_BAT (csd_cname + 1)

typedef struct csdumBATdef{
	csdumBatType batType; 
	str name;
	int headType; 
	int tailType; 
} csdumBATdef; 

typedef struct CSDumpBATs{
	BAT** dumpBats; 
} CSDump; 
/*
typedef struct CSDumpBATs {	//TODO: Use array of BAT and a new CSDumpBatType struct for simplifying this struct
	BAT*    tblIdBat;
	BAT* 	tblnameBat; 	//SQLname of each table
	BAT* 	csIdBat; 
	BAT*	freqBat; 
	BAT*	coverageBat; 
	BAT*	pOffsetBat; 
	BAT*	fullPBat; 	//Store full list of properties for each CS
	BAT*	cOffsetBat; 
	BAT*	fullCBat; 	//Store list of columns in the table
	BAT* 	cnameBat; 	//SQL name of each column
} CSDump;
*/

//The simpleCS is similar to a CS, 
//but stores less amount of information 
//however, stores additional information for the table 
//(after removing infrequent props....)
typedef struct SimpleDumpCS {	
	int tblId; 
	int freqId; 
	int numP; 
	oid *lstProp; 
	int numC; 
	oid *lstCol; 
	int sup; 
	int cov; 

} SimpleCS; 

typedef struct SimpleCSset{
	int num; 
	SimpleCS **items; 
} SimpleCSset; 


extern str 	tblIdBatname, tblnameBatname, csIdBatname,
		freqBatname, coverageBatname,
		pOffsetBatname, fullPBatname,
		cOffsetBatname, fullCBatname, cnameBatname;


rdf_export 
void dumpFreqCSs(CStableStat* cstablestat, CSset *freqCSset); 

rdf_export
SimpleCSset *dumpBat_to_CSset(void);

rdf_export
void free_simpleCSset(SimpleCSset *csset); 

rdf_export
void print_simpleCSset(SimpleCSset *csset);

rdf_export 
PropStat* getPropStat_P_simpleCSset(SimpleCSset* csset);

rdf_export
PropStat* getPropStat_C_simpleCSset(SimpleCSset* csset);

rdf_export 
Postinglist get_p_postingList(PropStat *propStat, oid p);

#endif /* _RDFDUMP_H_ */
