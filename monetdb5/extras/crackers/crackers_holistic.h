#ifndef _CRACKERS_HOLISTIC_H_
#define _CRACKERS_HOLISTIC_H_



#ifdef WIN32
#ifndef LIBCRACKERS
#define holistic_export extern __declspec(dllimport)
#else
#define holistic_export extern __declspec(dllexport)
#endif
#else
#define holistic_export extern
#endif


typedef struct frequency{

	int 		bid;   		/* The base BAT */
	int 		c;		/*number of times the column was cracked*/
	int 		f1;		/*number of queries that triggered cracking*/
	int 		f2;		/*number of queries that did not trigger cracking(because the value already existed in the index)*/
	double 		weight;
	struct frequency *next;

}FrequencyNode;

holistic_export int existsFrequencyStruct(FrequencyNode* head);
holistic_export FrequencyNode* createFrequencyStruct(void);
holistic_export FrequencyNode* push(int bat_id,FrequencyNode* head);
holistic_export FrequencyNode* pop(FrequencyNode* head);
holistic_export void printFrequencyStruct(FrequencyNode* head);
holistic_export FrequencyNode* searchBAT(FrequencyNode* head,int bat_id);
holistic_export double changeWeight(FrequencyNode* node,int N,int L1);
holistic_export int findMax(FrequencyNode* head);

#endif /*crackers_holistic*/ 
