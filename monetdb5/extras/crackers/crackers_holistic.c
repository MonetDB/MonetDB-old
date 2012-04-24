#include "monetdb_config.h"
#include "crackers_holistic.h"
#include "gdk.h"


int 
existsFrequencyStruct(FrequencyNode* head)
{
	if(head == NULL)
		return 0;
	else
		return 1;
}
FrequencyNode* 
createFrequencyStruct()
{
	FrequencyNode *x;
	x=(FrequencyNode *) GDKmalloc(sizeof(FrequencyNode)); 
	x->bid=0;	
	x->c=0;
	x->f1=0;
	x->f2=0;
	x->weight=0.0;
	x->next=NULL;
	return x;
}
FrequencyNode* 
push(int bat_id,FrequencyNode* head)
{
	FrequencyNode* new_node;
	new_node=(FrequencyNode *) GDKmalloc(sizeof(FrequencyNode));
	new_node->bid=bat_id;
	new_node->c=0;
	new_node->f1=0;
	new_node->f2=0;
	new_node->weight=0.0;
	new_node->next=head->next;
	head->next=new_node;
	return new_node; 
}
FrequencyNode*
pop(FrequencyNode* head)
{
	FrequencyNode* dummy;
	dummy=head->next;
	head->next=head->next->next;
	GDKfree(dummy);
	return head;
}
void 
printFrequencyStruct(FrequencyNode* head)
{
	int i=0;
	FrequencyNode* temp;
	temp=head;
	while(temp != NULL)
	{
		fprintf(stderr,"Item No. %d:Bid=%d c=%d f1=%d f2=%d W=%lf  \n",i++,temp->bid,temp->c,temp->f1,temp->f2,temp->weight);
		temp=temp->next;
	}
}
FrequencyNode* 
searchBAT(FrequencyNode* head,int bat_id)
{
	FrequencyNode* temp;
	temp=head;
	while((temp->bid != bat_id))
	{
		temp=temp->next;
	}
	return temp;
}
double
changeWeight(FrequencyNode* node,int N,int L1)
{
	int p; /*number of pieces in the index*/
	double Sp; /*average size of each piece*/
	double d; /*distance from optimal piece(L1)*/
	p = 2 * (node->c);
	Sp =((double)N)/p;	
	d = ABS(Sp - L1);
	node->weight = ((node->f2)/(node->f1)) * d;

	return node->weight;

}
