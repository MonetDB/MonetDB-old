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

/* This contains algebra functions used for RDF store only */

#include "monetdb_config.h"
#include "rdf.h"
#include <gdk.h>
#include <hashmap/hashmap.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "rdfdump.h"
#include "bat5.h"
#include "rdfcommon.h"


static csdumBATdef csdumBatdefs[N_CSDUM_BAT] = {
	{csd_tblId, "tblIdBat_dump", TYPE_void, TYPE_int},
	{csd_tblname, "tblnameBat_dump", TYPE_void, TYPE_oid},
	{csd_csId, "csIdBat_dump", TYPE_void, TYPE_int},
	{csd_freq, "freqBat_dump", TYPE_void, TYPE_int},
	{csd_coverage, "coverageBat_dump", TYPE_void, TYPE_int},
	{csd_pOffset, "pOffsetBat_dump", TYPE_void, TYPE_oid},
	{csd_fullP, "fullPBat_dump", TYPE_void, TYPE_oid},
	{csd_cOffset, "cOffsetBat_dump", TYPE_void, TYPE_oid},
	{csd_fullC, "fullCBat_dump", TYPE_void, TYPE_oid},
	{csd_isMV, "isMVBat_dump", TYPE_void, TYPE_int},	// 0 indicating single-valued column, otherwise > 0
								// the value is the number of column in MVtable
	{csd_cname, "cIdxBat_dump", TYPE_void, TYPE_int}	//Index of the col in the table
};

static 
str initCSDump(CSDump *csdump, int *already_built){
	
	bat 	mapBatId;
	int 	ret; 
	int 	i = 0;

	*already_built = 0;
	
	mapBatId = BBPindex(csdumBatdefs[0].name);
	if (mapBatId != 0){
		printf("The dump of CSset has been built \n");
		*already_built = 1; 
		return MAL_SUCCEED; 
	}

	csdump->dumpBats = (BAT **) malloc(sizeof (BAT*) * N_CSDUM_BAT);
	for (i = 0; i < N_CSDUM_BAT; i++){
	
		csdump->dumpBats[i] = BATnew(csdumBatdefs[i].headType, csdumBatdefs[i].tailType 
					      , smallbatsz, PERSISTENT);

		BATseqbase(csdump->dumpBats[i], 0);
		
		if (csdump->dumpBats[i] == NULL) {
			throw(MAL, "initCSDump", OPERATION_FAILED);
		}

		if (BKCsetName(&ret, (int *) &(csdump->dumpBats[i]->batCacheid), (const char*const*) &csdumBatdefs[i].name) != MAL_SUCCEED)	
			throw(MAL, "initCSDump", OPERATION_FAILED);
		
		if (BKCsetPersistent(&ret, (int *) &(csdump->dumpBats[i]->batCacheid)) != MAL_SUCCEED)
			throw(MAL, "initCSDump", OPERATION_FAILED);


	}

	return MAL_SUCCEED; 
}

static 
void commitCSDump(CSDump *csdump){
	int i;
	bat *lstCommits; 
	lstCommits = GDKmalloc(sizeof(bat) * (N_CSDUM_BAT+1)); 
	lstCommits[0] = 0;
	for (i = 1; i < (N_CSDUM_BAT+1); i++){
		lstCommits[i] = csdump->dumpBats[i-1]->batCacheid;
	}

	TMsubcommit_list(lstCommits,(N_CSDUM_BAT+1));

	GDKfree(lstCommits);
}

static 
void dumpCS(CSDump *csdump, int _freqId, int _tblId, CS cs, CStable cstbl){
	BUN	offset, offsetc; 
	int tblId, freqId, freq, cov; 
	oid tblname; 
	int *lstIsMV;
	int i;
	
	tblId = _tblId; 
	freqId = _freqId; 
	freq = cs.support; 
	cov = cs.coverage; 
	tblname = cstbl.tblname; 

	assert(tblId == (int)BATcount(csdump->dumpBats[csd_tblId]));
	BUNappend(csdump->dumpBats[csd_tblId], &tblId, TRUE);

	BUNappend(csdump->dumpBats[csd_tblname], &tblname, TRUE);

	BUNappend(csdump->dumpBats[csd_csId], &freqId, TRUE);
	
	BUNappend(csdump->dumpBats[csd_freq], &freq, TRUE); 
	BUNappend(csdump->dumpBats[csd_coverage], &cov, TRUE); 

	offset = BUNlast(csdump->dumpBats[csd_fullP]);

	/* Add list of p to fullPBat and dumpBats[csd_pOffset]*/
	BUNappend(csdump->dumpBats[csd_pOffset], &offset , TRUE);
	appendArrayToBat(csdump->dumpBats[csd_fullP], cs.lstProp, cs.numProp);

	offsetc = BUNlast(csdump->dumpBats[csd_fullC]);
	/* Add list of columns to dumpBats[csd_fullC] and dumpBats[csd_cOffset]*/
	BUNappend(csdump->dumpBats[csd_cOffset], &offsetc , TRUE);
	appendArrayToBat(csdump->dumpBats[csd_fullC], cstbl.lstProp, cstbl.numCol);
	/* Add list of multi-valued indication to csd_isMV bat*/

	lstIsMV = (int *) malloc(sizeof(int) * cstbl.numCol);
	for (i = 0; i < cstbl.numCol; i++){
		lstIsMV[i] = cstbl.lstMVTables[i].numCol; 
	}
	
	appendIntArrayToBat(csdump->dumpBats[csd_isMV], lstIsMV, cstbl.numCol);

	free(lstIsMV); 

}

void dumpFreqCSs(CStableStat* cstablestat, CSset *freqCSset){
	int i, numTables;
	int freqId; 
	int is_already_built = 0; 

	CSDump *csdump = NULL; 
	
	csdump = (CSDump *) malloc(sizeof(CSDump));
	csdump->dumpBats = NULL; 

	initCSDump(csdump, &is_already_built); 
	if (is_already_built){
		free(csdump); 
	}
	else{
		printf("Dumping CSset to BATs\n");
		numTables = cstablestat->numTables;
		
		for (i = 0; i < numTables; i++){
			freqId = cstablestat->lstfreqId[i];
			assert(freqId != -1); 
			dumpCS(csdump, freqId, i, freqCSset->items[freqId], cstablestat->lstcstable[i]); 			
		}

		commitCSDump(csdump); 	
	}
}

static
str read_BATs_from_dump(CSDump *csdump){
	
	int i; 
	
	csdump->dumpBats = (BAT **) malloc(sizeof(BAT *) * N_CSDUM_BAT); 
		
	for (i = 0; i < N_CSDUM_BAT; i++){
		bat tmpId = BBPindex(csdumBatdefs[i].name);

		if (tmpId == 0) 
			throw(MAL, "read_BATs_from_dump", "The dump Bats should be built already");

		if ((csdump->dumpBats[i]= BATdescriptor(tmpId)) == NULL) {
			throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
		}

	}

	return MAL_SUCCEED; 
}

static 
void freeCSDump(CSDump *csdump){
	int i; 
	if (csdump->dumpBats){
		for (i = 0; i < N_CSDUM_BAT; i++){
			if (csdump->dumpBats[i])
				BBPunfix(csdump->dumpBats[i]->batCacheid);
		}
		free(csdump->dumpBats);
	}
	
	free(csdump);
}


static
SimpleCS *create_simpleCS(int tblId, oid tblname, int freqId, int numP, oid* lstProp, int numC, oid* lstCol, int* lstIsMV,  int sup, int cov){
	SimpleCS *cs;  
	cs = (SimpleCS *) malloc(sizeof(SimpleCS)); 
	cs->tblId = tblId; 
	cs->tblname = tblname;
	cs->freqId = freqId; 
	
	cs->numP = numP; 
	cs->lstProp = (oid *) malloc(sizeof(oid) * numP); 
	copyOidSet(cs->lstProp, lstProp, numP);

	cs->numC = numC; 
	cs->lstCol = (oid *) malloc(sizeof(oid) * numC);
	copyOidSet(cs->lstCol, lstCol, numC);

	cs->lstIsMV = (int *) malloc(sizeof(int) * numC); 
	copyIntSet(cs->lstIsMV, lstIsMV, numC); 

	cs->sup = sup; 
	cs->cov = cov; 

	return cs; 
}
static 
void free_simpleCS(SimpleCS *cs){
	if (cs->lstProp) free(cs->lstProp);
	if (cs->lstCol) free(cs->lstCol); 
	if (cs->lstIsMV) free(cs->lstIsMV);
	free(cs); 
}	


static
SimpleCS* read_a_cs_from_csdump(int pos, CSDump *csdump){

	BUN *offsetP, *offsetP2, *offsetC, *offsetC2;
	int numP, numC; 
	int *tblId, *freqId, *freq, *coverage;
	oid *tblname; 
	oid *lstProp = NULL, *lstCol = NULL; 
	int *lstIsMV = NULL; 

	SimpleCS *cs; 

	
	tblId = (int *) Tloc(csdump->dumpBats[csd_tblId], pos); 
	assert(*tblId == pos); 

	tblname = (oid *) Tloc(csdump->dumpBats[csd_tblname], pos);
	
	freqId = (int *) Tloc(csdump->dumpBats[csd_csId], pos); 

	freq = (int *) Tloc(csdump->dumpBats[csd_freq], pos); 

	coverage = (int *) Tloc(csdump->dumpBats[csd_coverage], pos); 
	
	//Get number of Properties
	offsetP = (oid *) Tloc(csdump->dumpBats[csd_pOffset], pos);

	if ((pos + 1) < (int)csdump->dumpBats[csd_pOffset]->batCount){
		offsetP2 = (oid *)Tloc(csdump->dumpBats[csd_pOffset], pos + 1);
		numP = *offsetP2 - *offsetP;
	}
	else
		numP = BUNlast(csdump->dumpBats[csd_fullP]) - *offsetP;
	
	lstProp = (oid *)Tloc(csdump->dumpBats[csd_fullP], *offsetP);	

	//Get number of columns 

	offsetC = (oid *) Tloc(csdump->dumpBats[csd_cOffset], pos);

	if ((pos + 1) < (int)csdump->dumpBats[csd_cOffset]->batCount){
		offsetC2 = (oid *)Tloc(csdump->dumpBats[csd_cOffset], pos + 1);
		numC = *offsetC2 - *offsetC;
	}
	else
		numC = BUNlast(csdump->dumpBats[csd_fullC]) - *offsetC;

	lstCol = (oid *)Tloc(csdump->dumpBats[csd_fullC], *offsetC);	

	lstIsMV = (int *)Tloc(csdump->dumpBats[csd_isMV], *offsetC); 

	cs = create_simpleCS(*tblId, *tblname, *freqId, numP, lstProp, numC, lstCol, lstIsMV, *freq, *coverage);

	return cs; 
}

static
SimpleCSset *init_simpleCSset(int numCS){
	SimpleCSset *csset; 
	int i = 0;
	csset = (SimpleCSset *)malloc(sizeof(SimpleCSset)); 
	csset->num = numCS; 
	csset->items = (SimpleCS **) malloc(sizeof(SimpleCS *) * numCS); 
	for (i = 0; i < numCS; i++){
		csset->items[i] = NULL; 
	}

	return csset;

}

void free_simpleCSset(SimpleCSset *csset){
	int i; 
	for (i = 0; i < csset->num; i++){
		if (csset->items[i]) free_simpleCS(csset->items[i]); 
	}
	free(csset->items); 
	free(csset); 
}

void print_simpleCSset(SimpleCSset *csset){
	int i, j;
	int num = csset->num; 

	for (i = 0; i < num; i++){
		SimpleCS *cs = csset->items[i]; 
		printf("Simple CS: %d [TblId: %d] [FreqId: %d] [Support: %d] [Coverage: %d]\n", i, cs->tblId, cs->freqId, cs->sup, cs->cov);  
		printf("              Props: "); 
		for (j = 0; j < cs->numP; j++){
			printf(" " BUNFMT, cs->lstProp[j]); 
		}	
		printf("\n"); 

		printf("              Cols: "); 
		for (j = 0; j < cs->numC; j++){
			printf(" " BUNFMT "  (isMV: %d) ", cs->lstCol[j],cs->lstIsMV[j]); 
		}	
		printf("\n"); 
	}
}

SimpleCSset *dumpBat_to_CSset(void){
	int numTbl =  0; 
	int i = 0; 
	SimpleCSset *csset = NULL; 

	CSDump *csdump = (CSDump *) malloc(sizeof(CSDump)); 
	if (read_BATs_from_dump(csdump) != MAL_SUCCEED){
		freeCSDump(csdump); 
		fprintf(stderr, "Fail while de-serializing data from dumpBat\n");
	}	


	numTbl = BATcount(csdump->dumpBats[csd_tblId]); 

	//BATprint(csdump->dumpBats[csd_pOffset]); 
	//BATprint(csdump->fullPBat); 

	csset = init_simpleCSset(numTbl); 

	for (i = 0; i < numTbl; i++){
		SimpleCS *tmpcs = read_a_cs_from_csdump(i, csdump); 
		csset->items[i] = tmpcs; 
	}

	freeCSDump(csdump); 

	return csset; 
}

int getColIdx_from_oid(int tblId, SimpleCSset *csset, oid coloid){
	
	int i = 0; 

	SimpleCS *cs = csset->items[tblId];

	for (i = 0; i < cs->numC; i++){
		if (cs->lstCol[i] == coloid) return i; 
	}
	
	if (i == cs->numC) return -1; 

	return -1; 
}

int isMVCol(int tblId, int colIdx, SimpleCSset *csset){
	return (csset->items[tblId])->lstIsMV[colIdx]; 
}

PropStat* getPropStat_P_simpleCSset(SimpleCSset* csset){

	int i, j; 

	PropStat* propStat; 

	int num = csset->num; 
	
	propStat = initPropStat(); 

	for (i = 0; i < num; i++){
		SimpleCS *cs = csset->items[i];
		for (j = 0; j < cs->numP; j++){
			addaProp(propStat, cs->lstProp[j], i, j);
		}

		if (cs->numP > propStat->maxNumPPerCS)
			propStat->maxNumPPerCS = cs->numP;
	}
	
	//Donot need to compute tfidf
	
	return propStat; 
}


PropStat* getPropStat_C_simpleCSset(SimpleCSset* csset){

	int i, j; 

	PropStat* propStat; 

	int num = csset->num;
	
	propStat = initPropStat(); 

	for (i = 0; i < num; i++){
		SimpleCS *cs = csset->items[i];
		for (j = 0; j < cs->numC; j++){
			addaProp(propStat, cs->lstCol[j], i, j);
		}

		if (cs->numC > propStat->maxNumPPerCS)
			propStat->maxNumPPerCS = cs->numC;
	}
	
	//Donot need to compute tfidf
	
	return propStat; 
}

/*
 * Given a property, get the postinglist for that p from
 * PropStat. The postinglist of p contains list of Table Ids
 * where p belongs to its corresponding CS.
 * Note that p may be in the list of column of that table as 
 * it may be refined while removing infrequent p. 
 * */
Postinglist get_p_postingList(PropStat *propStat, oid p){

	BUN ppos = BUN_NONE;
	Postinglist ptl;

	//Get number of BATs for this p
	ppos = BUNfnd(propStat->pBat, &p);

	if (ppos == BUN_NONE){
		fprintf(stderr, "The prop "BUNFMT" must be in propStat bat\n", p);
	}

	ptl =  propStat->plCSidx[ppos];
	
	return ptl; 
}

/*
void mergePostingList(){

	

}*/
