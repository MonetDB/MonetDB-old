#include "monetdb_config.h"
#include "crackers_holistic.h"
#include "crackers.h"
#include "gdk.h"
#include "mal_exception.h"
#include "mutils.h"

static FrequencyNode *_InternalFrequencyStruct = NULL;
static MT_Lock frequencylock;

str
CRKinitHolistic(int *ret)
{
	MT_lock_init(&frequencylock, "FrequencyStruct");
	*ret = 0;
	return MAL_SUCCEED;
}

FrequencyNode *
getFrequencyStruct(void)
{
	mal_set_lock(frequencylock, "getFrequencyStruct");
	if (_InternalFrequencyStruct == NULL) {
		_InternalFrequencyStruct = GDKmalloc(sizeof(FrequencyNode)); 
		_InternalFrequencyStruct->bid=0;
		_InternalFrequencyStruct->c=0;
		_InternalFrequencyStruct->f1=0;
		_InternalFrequencyStruct->f2=0;
		_InternalFrequencyStruct->weight=0.0;
		_InternalFrequencyStruct->next=NULL;
	}
	mal_unset_lock(frequencylock, "getFrequencyStruct");
	
	return _InternalFrequencyStruct;
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
int
findMax(FrequencyNode* head)
{
	FrequencyNode* temp;
	double tmpW;
	int bat;
	temp=head->next;
	tmpW=temp->weight;
	bat=temp->bid;
	while(temp!=NULL)
	{
		if(temp->weight > tmpW)
		{
			tmpW=temp->weight;
			bat=temp->bid;
		}
		temp=temp->next;
	}
	return bat;
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
	/*fprintf(stderr,"p=%d Sp=%lf d=%lf\n",p,Sp,d);*/
	if (node->f2!=0)
		node->weight = ((double)(node->f1)/(double)(node->f2)) * d;
	/*fprintf(stderr,"W=%lf\n",node->weight);*/
	return node->weight;

}

str 
CRKinitFrequencyStruct(int *vid,int *bid)
{
	FrequencyNode* new_node;
	FrequencyNode *fs = getFrequencyStruct();

	/*fprintf(stderr,"BAT_ID=%d\n",*bid);*/
	
	new_node=push(*bid,fs);
        fprintf(stderr,"Bid=%d c=%d f1=%d f2=%d weight=%lf \n",new_node->bid,new_node->c,new_node->f1,new_node->f2,new_node->weight);

	*vid = 0;
	return MAL_SUCCEED;
}

str
CRKrandomCrack(int *ret)
{
	int bid=0;
	BAT *b;
	int low=0, hgh=0;
	int *t;
	int temp=0;
	oid posl,posh,p;
	bit inclusive=TRUE;
	FrequencyNode *fs = getFrequencyStruct();

	bid=findMax(fs);
	b=BATdescriptor(bid);
	t=(int*)Tloc(b,BUNfirst(b));
	posl=BUNfirst(b);
	posh=BUNlast(b) - 1;
	p=(rand()%(posh-posl+1))+posl;
	low=t[p];
	p=(rand()%(posh-posl+1))+posl;
	hgh=t[p];
	if(hgh < low)
	{
		temp=low;
		low=hgh;
		hgh=temp;
	}
	/*fprintf(stderr,"posl = "OIDFMT" posh = "OIDFMT" low = %d hgh = %d inclusive = %d", posl,posh,low,hgh,inclusive );*/

	CRKselectholBounds_int(ret, &bid, &low, &hgh, &inclusive, &inclusive);
	*ret = 0;
	return MAL_SUCCEED;
}

