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
 * Copyright August 2008-2012 MonetDB B.V.
 * All Rights Reserved.
 */

#include "monetdb_config.h"
#include "crackers.h"
#include "gdk.h"
#include "mal_exception.h"
#include "opt_pipes.h"
#include "mutils.h"

static FrequencyNodeSideways *_InternalFrequencyStructSidewaysA = NULL;
static FrequencyNodeSideways *_InternalFrequencyStructSidewaysB = NULL;
static MT_Lock frequencylocksideways;

int isIdleQuerySideways = 0;

str
CRKinitHolisticSideways(int *ret)
{
	MT_lock_init(&frequencylocksideways, "FrequencyStructSideways");

	*ret = 0;
	return MAL_SUCCEED;
}

/*singleton pattern*/
FrequencyNodeSideways *
getFrequencyStructSideways(char which)
{
	FrequencyNodeSideways **theNode = NULL;

	MT_lock_set(&frequencylocksideways, "getFrequencyStructSideways");
	switch (which) {
                case 'A':
                        theNode = &_InternalFrequencyStructSidewaysA;
                        break;
                case 'B':
                        theNode = &_InternalFrequencyStructSidewaysB;
                        break;
                default:
                        assert(0);
         }

        /* GDKzalloc = calloc = malloc + memset(0) */
        if (*theNode == NULL)
                *theNode = GDKzalloc(sizeof(FrequencyNodeSideways));
	MT_lock_unset(&frequencylocksideways, "getFrequencyStructSideways");
	
	return *theNode;
}
/*this function pushes nodes in the list and is used in cost models: 2,4,6,8,10*/
void 
pushSideways(int bat_id1, int bat_id2, FrequencyNodeSideways* head)
{
	FrequencyNodeSideways* new_node;
	new_node=(FrequencyNodeSideways *) GDKmalloc(sizeof(FrequencyNodeSideways));
	new_node->bid_1=bat_id1;
	new_node->bid_2=bat_id2;
	new_node->c=1;
	new_node->f1=0;
	new_node->f2=0;
	new_node->weight=0.0; /*weight=f1*((N/c)-L1)*/
	new_node->next=head->next;
	head->next=new_node;

}
FrequencyNodeSideways*
findMaxSideways(FrequencyNodeSideways* head)
{
	FrequencyNodeSideways* temp;
	FrequencyNodeSideways* ret_node=NULL;
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

/*The following function updates the weights in the list*/
/*This cost model takes into consideration both the frequency of the queries that use the index and the distance from the optimal index.*/
/*The initial weights are initialized to 0 (ZERO)*/
double
changeWeightSideways(FrequencyNodeSideways* node,int N,int L1)
{
	int p; /*number of pieces in the index*/
	double Sp; /*average size of each piece*/
	double d; /*distance from optimal piece(L1)*/
	p = node->c;
	Sp =((double)N)/p;	
	d = Sp - L1;
	if (node->f1==0)
	{
		node->weight = 0;
	}
	else
	{
		node->weight =  d;
	}
	fprintf(stderr,"bid1=%d bid2=%d f1=%d f2=%d p=%d Sp=%lf d=%lf W=%lf\n",node->bid_1,node->bid_2,node->f1,node->f2,p,Sp,d,node->weight);
	return node->weight;
}


str 
CRKinitFrequencyStructSideways(int *vid,int *bid1, int *bid2)
{
	FrequencyNodeSideways *fs = getFrequencyStructSideways('A');
	/*fprintf(stderr,"BAT_ID=%d\n",*bid);*/
	pushSideways(*bid1,*bid2,fs);
	*vid = 0;
	return MAL_SUCCEED;
}
FrequencyNodeSideways* 
searchBATSideways(FrequencyNodeSideways* head,int bat_id1,int bat_id2)
{
	FrequencyNodeSideways* temp;
	temp=head;
	while(temp!=NULL)
	{
		if((temp->bid_1 == bat_id1) && (temp->bid_2 == bat_id2))
			break;
		temp=temp->next;
	}
	return temp;
}
void 
printFrequencyStructSideways(FrequencyNodeSideways* head)
{
	FrequencyNodeSideways* temp;
	temp=head;
	while(temp != NULL)
	{
		fprintf(stderr,"Bid1=%d Bid2=%d c=%d f1=%d f2=%d W=%lf  \n",temp->bid_1,temp->bid_2,temp->c,temp->f1,temp->f2,temp->weight);
		temp=temp->next;
	}
}

void 
AlignInformation(FrequencyNodeSideways* head,FrequencyNodeSideways* node)
{

	FrequencyNodeSideways* temp;
	double Weight=0.0;
	int maxPieces;
	temp=head;
	maxPieces=temp->c;
	while(temp!=NULL)
	{
		if(temp->bid_1 == node->bid_1)
		{
			if(temp->c >= maxPieces)
			{
				maxPieces=temp->c; 
				Weight=temp->weight;
			}
		}
		temp=temp->next;
	}
	temp=head;
	while(temp!=NULL)
	{
		if(temp->bid_1 == node->bid_1)
		{
			temp->f1=node->f1;
			temp->c=maxPieces;
			temp->weight=Weight;
		}
		temp=temp->next;
	}
}
/*This function is used during idle time for all the cost models*/
str
CRKrandomCrackSideways(int *ret)
{
	int bid=0,pbid=0;
	FrequencyNodeSideways* max_node;
	BAT *b1,*b2;
	int *t;
	oid posl,posh,p;
	bit inclusive=TRUE;
	FrequencyNodeSideways *fs = getFrequencyStructSideways('A');	
	isIdleQuerySideways=1;
	max_node=findMaxSideways(fs);
	bid=max_node->bid_1;
	pbid=max_node->bid_2;
	b1=BATdescriptor(bid);
	b2=BATdescriptor(pbid);

	if(max_node!=NULL)
	{

	switch ( ATOMtype(b1->ttype)) {
	case TYPE_int:
		{
			int low,hgh,temp;
			t=(int*)Tloc(b1,BUNfirst(b1));
			posl=BUNfirst(b1);
			posh=BUNlast(b1) - 1;
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
			switch (ATOMtype(b2->ttype)) {
				case TYPE_lng:{CRKtselectBoundsSidewayshol_int_lng(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_int:{CRKtselectBoundsSidewayshol_int_int(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_str:{CRKtselectBoundsSidewayshol_int_str(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_oid:{CRKtselectBoundsSidewayshol_int_oid(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				default: 
					{
					if(ATOMtype(b2->ttype)==TYPE_date)
						CRKtselectBoundsSidewayshol_int_date(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);
					else
						fprintf(stderr,"Not supported type for b2\n");
					}
			}
		}break;
	case TYPE_lng:
		{
			lng low,hgh,temp;
			t=(int*)Tloc(b1,BUNfirst(b1));
			posl=BUNfirst(b1);
			posh=BUNlast(b1) - 1;
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
			switch (ATOMtype(b2->ttype)) {
				case TYPE_lng:{CRKtselectBoundsSidewayshol_lng_lng(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				default: 
					{
					if(ATOMtype(b2->ttype)==TYPE_date)
						CRKtselectBoundsSidewayshol_lng_date(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);
					else
						fprintf(stderr,"Not supported type for b2\n");
					}
			}
		}break;
	case TYPE_dbl:
		{
			dbl low,hgh,temp;
			t=(int*)Tloc(b1,BUNfirst(b1));
			posl=BUNfirst(b1);
			posh=BUNlast(b1) - 1;
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
			switch ( ATOMtype(b2->ttype)) {
				case TYPE_dbl:{CRKtselectBoundsSidewayshol_dbl_dbl(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_int:{CRKtselectBoundsSidewayshol_dbl_int(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_str:{CRKtselectBoundsSidewayshol_dbl_str(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				default: 
					{
					if(ATOMtype(b2->ttype)==TYPE_date)
						CRKtselectBoundsSidewayshol_dbl_date(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);
					else
						fprintf(stderr,"Not supported type for b2\n");
					}
			}
		}break;
	default: 
		{
		if(ATOMtype(b1->ttype)==TYPE_date)
		{
			date low,hgh,temp;
			t=(int*)Tloc(b1,BUNfirst(b1));
			posl=BUNfirst(b1);
			posh=BUNlast(b1) - 1;
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
			switch ( ATOMtype(b2->ttype)) {
				case TYPE_lng:{CRKtselectBoundsSidewayshol_date_lng(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_int:{CRKtselectBoundsSidewayshol_date_int(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_str:{CRKtselectBoundsSidewayshol_date_str(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				case TYPE_dbl:{CRKtselectBoundsSidewayshol_date_dbl(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);}break;
				default: 
					{
					if(ATOMtype(b2->ttype)==TYPE_date)
						CRKtselectBoundsSidewayshol_date_date(ret, &bid, &pbid, &low, &hgh, &inclusive, &inclusive);
					else
						fprintf(stderr,"Not supported type for b2\n");
					}
			}
		}	
		else
			fprintf(stderr,"Not supported type for b2\n");
		}
    	}
	
	}
	isIdleQuerySideways=0;

	*ret = 0;
	return MAL_SUCCEED;
}

