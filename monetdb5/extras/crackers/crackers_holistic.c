#include "monetdb_config.h"
#include "crackers_holistic.h"
#include "crackers.h"
#include "gdk.h"
#include "mal_exception.h"
#include "mutils.h"

static FrequencyNode *_InternalFrequencyStructA = NULL;
static FrequencyNode *_InternalFrequencyStructB = NULL;
static MT_Lock frequencylock;

str
CRKinitHolistic(int *ret)
{
	MT_lock_init(&frequencylock, "FrequencyStruct");
	*ret = 0;
	return MAL_SUCCEED;
}

/*singleton pattern*/
FrequencyNode *
getFrequencyStruct(char which)
{
	FrequencyNode **theNode = NULL;

	mal_set_lock(frequencylock, "getFrequencyStruct");
	switch (which) {
                case 'A':
                        theNode = &_InternalFrequencyStructA;
                        break;
                case 'B':
                        theNode = &_InternalFrequencyStructB;
                        break;
                default:
                        assert(0);
         }

        /* GDKzalloc = calloc = malloc + memset(0) */
        if (*theNode == NULL)
                *theNode = GDKzalloc(sizeof(FrequencyNode));
	mal_unset_lock(frequencylock, "getFrequencyStruct");
	
	return *theNode;
}

void 
push(int bat_id,FrequencyNode* head)
{
	FrequencyNode* new_node;
	new_node=(FrequencyNode *) GDKmalloc(sizeof(FrequencyNode));
	new_node->bid=bat_id;
	new_node->c=1;
	new_node->f1=0;
	new_node->f2=0;
	new_node->weight=0.0; /*weight=f1*((N/c)-L1)*/
	new_node->next=head->next;
	head->next=new_node; 
}

/*this function pushes nodes in the list in the first and the second experiment (1st & 2nd cost model)*/
void 
push_2(int bat_id,FrequencyNode* head,int N,int L1)
{
	FrequencyNode* new_node;
	new_node=(FrequencyNode *) GDKmalloc(sizeof(FrequencyNode));
	new_node->bid=bat_id;
	new_node->c=1;
	new_node->f1=0;
	new_node->f2=0;
	new_node->weight=N-L1;
	new_node->next=head->next;
	head->next=new_node; 
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

FrequencyNode*
findMax(FrequencyNode* head)
{
	FrequencyNode* temp;
	FrequencyNode* ret_node=NULL;
	double tmpW;
	//int bat;
	temp=head->next;
	tmpW=temp->weight;
	//bat=temp->bid;
	while(temp!=NULL)
	{
		if(temp->weight >= tmpW)
		{
			tmpW=temp->weight;
			//bat=temp->bid;
			ret_node=temp;
		}
		temp=temp->next;
	}
	return ret_node;
}

double
changeWeight(FrequencyNode* node,int N,int L1)
{
	int p; /*number of pieces in the index*/
	double Sp; /*average size of each piece*/
	double d; /*distance from optimal piece(L1)*/
	p = node->c;
	Sp =((double)N)/p;	
	d = Sp - L1;
	/*fprintf(stderr,"p=%d Sp=%lf d=%lf\n",p,Sp,d);*/
	if (node->f2!=0)
		node->weight = ((double)(node->f1)/(double)(node->f2)) * d;
	else
		node->weight = (double)(node->f1) * d;

	/*fprintf(stderr,"W=%lf\n",node->weight);*/
	return node->weight;

}

/*this function updates the weights in the list in the first experiment (1st cost model)*/
double
changeWeight_1(FrequencyNode* node,int N,int L1)
{
	int p; /*number of pieces in the index*/
	double Sp; /*average size of each piece*/
	double d; /*distance from optimal piece(L1)*/
	p = node->c;
	Sp =((double)N)/p;	
	d = Sp - L1;
	/*fprintf(stderr,"p=%d Sp=%lf d=%lf\n",p,Sp,d);*/
	node->weight = d;
	/*fprintf(stderr,"W=%lf\n",node->weight);*/
	return node->weight;
}

/*this function updates the weights in the list in the second experiment (2nd cost model)*/
double
changeWeight_2(FrequencyNode* node,int N,int L1)
{
	int p; /*number of pieces in the index*/
	double Sp; /*average size of each piece*/
	double d; /*distance from optimal piece(L1)*/
	p = node->c;
	Sp =((double)N)/p;	
	d = Sp - L1;
	/*fprintf(stderr,"p=%d Sp=%lf d=%lf\n",p,Sp,d);*/
	node->weight = (double)(node->f1) * d;
	/*fprintf(stderr,"W=%lf\n",node->weight);*/
	return node->weight;
}

void
deleteNode(FrequencyNode* head,int bat_id)
{
	FrequencyNode* temp;
	temp=head;
	while((temp->next != NULL))
	{
		if(temp->next->bid == bat_id)
		{
			temp->next=temp->next->next;	
			break;
		}
		temp=temp->next;
	}

}

str 
CRKinitFrequencyStruct(int *vid,int *bid)
{
	FrequencyNode *fs = getFrequencyStruct('A');
	/*fprintf(stderr,"BAT_ID=%d\n",*bid);*/
	push(*bid,fs);
	*vid = 0;
	return MAL_SUCCEED;
}


/*this function initializes the list in the first & second experiment(1st & 2nd cost model)*/
str 
CRKinitFrequencyStruct_2(int *vid,int *bid,int* N,int* L1)
{
	FrequencyNode *fs = getFrequencyStruct('A');
	/*fprintf(stderr,"BAT_ID=%d\n",*bid);*/
	push_2(*bid,fs,*N,*L1);
	*vid = 0;
	return MAL_SUCCEED;
}


str
CRKrandomCrack(int *ret)
{
	int bid=0;
	FrequencyNode* max_node;
	BAT *b;
	int low=0, hgh=0;
	int *t;
	int temp=0;
	oid posl,posh,p;
	bit inclusive=TRUE;
	FrequencyNode *fs = getFrequencyStruct('A');

	max_node=findMax(fs);
	if(max_node->weight > 0)
	{
		bid=max_node->bid;
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
		max_node->f1=max_node->f1-1; /*increase frequency only when the column is refined during workload executuion and not during idle time*/
	}
	*ret = 0;
	return MAL_SUCCEED;
}

