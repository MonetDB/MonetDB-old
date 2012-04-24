#ifndef _CRACKERS_HOLISTIC_H_
#define _CRACKERS_HOLISTIC_H_

typedef struct frequency{

	int 		bid;   		/* The base BAT */

	int 		c;		/*number of times the column was cracked*/
	int 		f1;		/*number of queries that triggered cracking*/
	int 		f2;		/*number of queries that did not trigger cracking(because the value already existed in the index)*/
	double 		weight;
	struct frequency *next;
}FrequencyNode;

int existsFrequencyStruct(FrequencyNode* head);
FrequencyNode* createFrequencyStruct();
FrequencyNode* push(int bat_id,FrequencyNode* head);
FrequencyNode* pop(FrequencyNode* head);
void printFrequencyStruct(FrequencyNode* head);
FrequencyNode* searchBAT(FrequencyNode* head,int bat_id);
double changeWeight(FrequencyNode* node,int N,int L1);
#endif /*crackers_holistic*/ 
