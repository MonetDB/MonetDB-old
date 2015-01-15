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


str 	tblIdBatname = "tblIdBat_dump", csIdBatname = "csIdBat_dump",
	freqBatname = "freqBat_dump", coverageBatname = "coverageBat_dump",
	pOffsetBatname = "pOffsetBat_dump", fullPBatname = "fullPBat_dump",
	cOffsetBatname = "cOffsetBat_dump", fullCBatname = "fullCBat_dump";


static 
str initCSDump(CSDump *csdump, int *already_built){
	
	bat 	mapBatId;
	int 	ret; 

	*already_built = 0;
	
	mapBatId = BBPindex(tblIdBatname);
	if (mapBatId != 0){
		printf("The dump of CSset has been built \n");
		*already_built = 1; 
		return MAL_SUCCEED; 
	}

	
	csdump->tblIdBat = BATnew(TYPE_void, TYPE_int, smallbatsz, PERSISTENT);


	BATseqbase(csdump->tblIdBat, 0);
	
	if (csdump->tblIdBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);
	}

	if (BKCsetName(&ret, (int *) &(csdump->tblIdBat->batCacheid), (const char*const*) &tblIdBatname) != MAL_SUCCEED)	
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->tblIdBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);



	csdump->csIdBat = BATnew(TYPE_void, TYPE_int, smallbatsz, PERSISTENT);

	if (csdump->csIdBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);
	}

	if (BKCsetName(&ret, (int *) &(csdump->csIdBat->batCacheid), (const char*const*) &csIdBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->csIdBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);


	csdump->freqBat = BATnew(TYPE_void, TYPE_int, smallbatsz, PERSISTENT);

	if (csdump->freqBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);
	}
	
	if (BKCsetName(&ret, (int *) &(csdump->freqBat->batCacheid), (const char*const*) &freqBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->freqBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);


	csdump->coverageBat = BATnew(TYPE_void, TYPE_int, smallbatsz, PERSISTENT);
	
	if (csdump->coverageBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);
	}

	if (BKCsetName(&ret, (int *) &(csdump->coverageBat->batCacheid), (const char*const*) &coverageBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->coverageBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);


	csdump->pOffsetBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, PERSISTENT);
	
	if (csdump->pOffsetBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);
	}

	if (BKCsetName(&ret, (int *) &(csdump->pOffsetBat->batCacheid), (const char*const*) &pOffsetBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->pOffsetBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);


	csdump->fullPBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, PERSISTENT);
	
	if (csdump->fullPBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);	
	}

	if (BKCsetName(&ret, (int *) &(csdump->fullPBat->batCacheid), (const char*const*) &fullPBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->fullPBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	

	csdump->cOffsetBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, PERSISTENT);
	
	if (csdump->cOffsetBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);	
	}

	if (BKCsetName(&ret, (int *) &(csdump->cOffsetBat->batCacheid), (const char*const*) &cOffsetBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->cOffsetBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);


	csdump->fullCBat = BATnew(TYPE_void, TYPE_oid, smallbatsz, PERSISTENT);
	
	if (csdump->fullCBat == NULL) {
		throw(MAL, "initCSDump", OPERATION_FAILED);
	}

	if (BKCsetName(&ret, (int *) &(csdump->fullCBat->batCacheid), (const char*const*) &fullCBatname) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);
	
	if (BKCsetPersistent(&ret, (int *) &(csdump->fullCBat->batCacheid)) != MAL_SUCCEED)
		throw(MAL, "initCSDump", OPERATION_FAILED);

	return MAL_SUCCEED; 
}

static 
void commitCSDump(CSDump *csdump){

	bat *lstCommits; 
	lstCommits = GDKmalloc(sizeof(bat) * 9); 
	lstCommits[0] = 0;
	lstCommits[1] = csdump->tblIdBat->batCacheid;
	lstCommits[2] = csdump->csIdBat->batCacheid;
	lstCommits[3] = csdump->freqBat->batCacheid;
	lstCommits[4] = csdump->coverageBat->batCacheid;
	lstCommits[5] = csdump->pOffsetBat->batCacheid;
	lstCommits[6] = csdump->fullPBat->batCacheid;
	lstCommits[7] = csdump->cOffsetBat->batCacheid;
	lstCommits[8] = csdump->fullCBat->batCacheid;

	TMsubcommit_list(lstCommits,9);

	GDKfree(lstCommits);
}

static 
void dumpCS(CSDump *csdump, int _freqId, int _tblId, CS cs, CStable cstbl){
	BUN	offset, offsetc; 
	int tblId, freqId, freq, cov; 
	
	tblId = _tblId; 
	freqId = _freqId; 
	freq = cs.support; 
	cov = cs.coverage; 

	assert(tblId == (int)BATcount(csdump->tblIdBat));
	BUNappend(csdump->tblIdBat, &tblId, TRUE);

	BUNappend(csdump->csIdBat, &freqId, TRUE);
	
	BUNappend(csdump->freqBat, &freq, TRUE); 
	BUNappend(csdump->coverageBat, &cov, TRUE); 

	offset = BUNlast(csdump->fullPBat);

	/* Add list of p to fullPBat and pOffsetBat*/
	BUNappend(csdump->pOffsetBat, &offset , TRUE);
	appendArrayToBat(csdump->fullPBat, cs.lstProp, cs.numProp);

	offsetc = BUNlast(csdump->fullCBat);
	/* Add list of columns to fullCBat and cOffsetBat*/
	BUNappend(csdump->cOffsetBat, &offsetc , TRUE);
	appendArrayToBat(csdump->fullCBat, cstbl.lstProp, cstbl.numCol);


}

void dumpFreqCSs(CStableStat* cstablestat, CSset *freqCSset){
	int i, numTables;
	int freqId; 
	int is_already_built = 0; 

	CSDump *csdump = NULL; 
	
	csdump = (CSDump *) malloc(sizeof(CSDump));
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

	bat	tblIdBatid, csIdBatid, freqBatid, coverageBatid, pOffsetBatid, 
		fullPBatid, cOffsetBatid, fullCBatid; 

	tblIdBatid = BBPindex(tblIdBatname);
	csIdBatid = BBPindex(csIdBatname); 
	freqBatid = BBPindex(freqBatname); 
	coverageBatid = BBPindex(coverageBatname); 
	pOffsetBatid = BBPindex(pOffsetBatname); 
	fullPBatid = BBPindex(fullPBatname); 
	cOffsetBatid = BBPindex(cOffsetBatname); 
	fullCBatid = BBPindex(fullCBatname); 
	
	if (tblIdBatid == 0 ||  csIdBatid == 0 || freqBatid == 0 || coverageBatid == 0  || 
 	    pOffsetBatid == 0 || fullPBatid == 0 || cOffsetBatid == 0 || fullCBatid == 0){
		throw(MAL, "read_BATs_from_dump", "The dump Bats should be built already");
	}

	if ((csdump->tblIdBat= BATdescriptor(tblIdBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}

	if ((csdump->csIdBat= BATdescriptor(csIdBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}
	
	if ((csdump->freqBat= BATdescriptor(freqBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}

	if ((csdump->coverageBat= BATdescriptor(coverageBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}
	
	if ((csdump->pOffsetBat= BATdescriptor(pOffsetBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}

	if ((csdump->fullPBat= BATdescriptor(fullPBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}
	
	if ((csdump->cOffsetBat = BATdescriptor(cOffsetBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}

	if ((csdump->fullCBat= BATdescriptor(fullCBatid)) == NULL) {
		throw(MAL, "read_BATs_from_dump", RUNTIME_OBJECT_MISSING);
	}
	
	return MAL_SUCCEED; 
}

static 
void freeCSDump(CSDump *csdump){
	BBPunfix(csdump->tblIdBat->batCacheid); 
	BBPunfix(csdump->csIdBat->batCacheid); 
	BBPunfix(csdump->freqBat->batCacheid); 
	BBPunfix(csdump->coverageBat->batCacheid); 
	BBPunfix(csdump->pOffsetBat->batCacheid); 
	BBPunfix(csdump->fullPBat->batCacheid); 
	BBPunfix(csdump->cOffsetBat->batCacheid); 
	BBPunfix(csdump->fullCBat->batCacheid); 
	
	free(csdump);
}


static
SimpleCS *create_simpleCS(int tblId, int freqId, int numP, oid* lstProp, int numC, oid* lstCol, int sup, int cov){
	SimpleCS *cs;  
	cs = (SimpleCS *) malloc(sizeof(SimpleCS)); 
	cs->tblId = tblId; 
	cs->freqId = freqId; 
	
	cs->numP = numP; 
	cs->lstProp = (oid *) malloc(sizeof(oid) * numP); 
	copyOidSet(cs->lstProp, lstProp, numP);

	cs->numC = numC; 
	cs->lstCol = (oid *) malloc(sizeof(oid) * numC);
	copyOidSet(cs->lstCol, lstCol, numC);

	cs->sup = sup; 
	cs->cov = cov; 

	return cs; 
}
static 
void free_simpleCS(SimpleCS *cs){
	if (cs->lstProp) free(cs->lstProp);
	if (cs->lstCol) free(cs->lstCol); 
	free(cs); 
}	


static
SimpleCS* read_a_cs_from_csdump(int pos, CSDump *csdump){

	BUN *offsetP, *offsetP2, *offsetC, *offsetC2;
	int numP, numC; 
	int *tblId, *freqId, *freq, *coverage;
	oid *lstProp = NULL, *lstCol = NULL; 

	SimpleCS *cs; 

	
	tblId = (int *) Tloc(csdump->tblIdBat, pos); 
	assert(*tblId == pos); 

	freqId = (int *) Tloc(csdump->csIdBat, pos); 

	freq = (int *) Tloc(csdump->freqBat, pos); 

	coverage = (int *) Tloc(csdump->coverageBat, pos); 
	
	//Get number of Properties
	offsetP = (oid *) Tloc(csdump->pOffsetBat, pos);

	if ((pos + 1) < (int)csdump->pOffsetBat->batCount){
		offsetP2 = (oid *)Tloc(csdump->pOffsetBat, pos + 1);
		numP = *offsetP2 - *offsetP;
	}
	else
		numP = BUNlast(csdump->fullPBat) - *offsetP;
	
	lstProp = (oid *)Tloc(csdump->fullPBat, *offsetP);	

	//Get number of columns 

	offsetC = (oid *) Tloc(csdump->cOffsetBat, pos);

	if ((pos + 1) < (int)csdump->cOffsetBat->batCount){
		offsetC2 = (oid *)Tloc(csdump->cOffsetBat, pos + 1);
		numC = *offsetC2 - *offsetC;
	}
	else
		numC = BUNlast(csdump->fullCBat) - *offsetC;

	lstCol = (oid *)Tloc(csdump->fullCBat, *offsetC);	

	cs = create_simpleCS(*tblId, *freqId, numP, lstProp, numC, lstCol, *freq, *coverage);

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
			printf(" " BUNFMT, cs->lstCol[j]); 
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
		free(csdump); 
		fprintf(stderr, "Fail while de-serializing data from dumpBat\n");
	}	


	numTbl = BATcount(csdump->tblIdBat); 

	//BATprint(csdump->pOffsetBat); 
	//BATprint(csdump->fullPBat); 

	csset = init_simpleCSset(numTbl); 

	for (i = 0; i < numTbl; i++){
		SimpleCS *tmpcs = read_a_cs_from_csdump(i, csdump); 
		csset->items[i] = tmpcs; 
	}

	freeCSDump(csdump); 

	return csset; 
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
