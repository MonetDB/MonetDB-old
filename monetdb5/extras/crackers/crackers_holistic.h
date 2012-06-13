#ifndef _CRACKERS_HOLISTIC_H_
#define _CRACKERS_HOLISTIC_H_

#include <mal.h>


#ifdef WIN32
#ifndef LIBCRACKERS
#define crackers_export extern __declspec(dllimport)
#else
#define crackers_export extern __declspec(dllexport)
#endif
#else
#define crackers_export extern
#endif


typedef struct frequency{

	int 		bid;   		/* The base BAT */
	int 		c;		/*number of times the column was cracked*/
	int 		f1;		/*number of queries that triggered cracking*/
	int 		f2;		/*number of queries that did not trigger cracking(because the value already existed in the index)*/
	double 		weight;
	struct frequency *next;

}FrequencyNode;

crackers_export FrequencyNode *getFrequencyStruct(char which);
crackers_export FrequencyNode* push(int bat_id,FrequencyNode* head);
crackers_export FrequencyNode* pop(FrequencyNode* head);
crackers_export void printFrequencyStruct(FrequencyNode* head);
crackers_export FrequencyNode* searchBAT(FrequencyNode* head,int bat_id);
crackers_export double changeWeight(FrequencyNode* node,int N,int L1);
crackers_export int findMax(FrequencyNode* head);

crackers_export str CRKinitHolistic(int *ret);
crackers_export str CRKinitFrequencyStruct(int *vid,int *bid);
crackers_export str CRKrandomCrack(int *ret);

#endif /*crackers_holistic*/ 
