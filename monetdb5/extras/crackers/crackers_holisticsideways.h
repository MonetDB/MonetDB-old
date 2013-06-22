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

#ifndef _CRACKERS_HOLISTICSIDEWAYS_H_
#define _CRACKERS_HOLISTICSIDEWAYS_H_

#include "mal.h"
#include "mal_client.h"

typedef struct frequencysideways{

	int 		bid_1;   		/* The head bid in cracker map (selection attribute) */
	int 		bid_2;   		/* The tail bid in cracker map */
	int 		c;		/*number of pieces in the index*/
	int 		f1;		/*number of queries that triggered cracking*/
	int 		f2;		/*number of queries that did not trigger cracking(because the value already existed in the index)*/
	double 		weight;
	struct frequencysideways *next;

}FrequencyNodeSideways;

crackers_export int isIdleQuerySideways;

crackers_export FrequencyNodeSideways *getFrequencyStructSideways(char which);
crackers_export void pushSideways(int bat_id1,int bat_id2,FrequencyNodeSideways* head);
crackers_export double changeWeightSideways(FrequencyNodeSideways* node,int N,int L1);
crackers_export FrequencyNodeSideways* findMaxSideways(FrequencyNodeSideways* head);
crackers_export FrequencyNodeSideways* searchBATSideways(FrequencyNodeSideways* head,int bat_id1,int bat_id2);
crackers_export void printFrequencyStructSideways(FrequencyNodeSideways* head);
crackers_export void AlignInformation(FrequencyNodeSideways* head,FrequencyNodeSideways* node);

crackers_export str CRKinitHolisticSideways(int *ret);
crackers_export str CRKinitFrequencyStructSideways(int *vid,int *bid_1, int *bid_2);
crackers_export str CRKrandomCrackSideways(int *ret);

#endif /*crackers_holistic*/ 
