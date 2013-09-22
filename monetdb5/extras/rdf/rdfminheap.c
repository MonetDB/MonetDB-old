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

#include "monetdb_config.h"
#include "rdf.h"
#include "tokenizer.h"
#include <math.h>
#include "rdfminheap.h"


//to get index of left child of node at index i
int leftchild(int i) { 
	return (2*i + 1); 
}

//to get index of right child of node at index i
int rightchild(int i) { 
	return (2*i + 2); 
}

//to get the root
MinHeapNode getMin(MinHeap *minHeap) { 
	return minHeap->harr[0];
}

// A utility function to swap two elements
void swap(MinHeapNode *x, MinHeapNode *y){
	MinHeapNode temp = *x;  *x = *y;  *y = temp;
}


//to replace root with new node x and heapify() new root
void replaceMin(MinHeap *minHeap, MinHeapNode x){ 
	minHeap->harr[0] = x;  
	MinHeapify(minHeap, 0); 
}

// This method assumes that the subtrees are already heapified
void MinHeapify(MinHeap *minHeap, int i)
{
	int l = leftchild(i);
	int r = rightchild(i);
	int smallest = i;
	if (l < minHeap->heap_size && minHeap->harr[l].element < minHeap->harr[i].element)
		smallest = l;
	if (r < minHeap->heap_size && minHeap->harr[r].element < minHeap->harr[smallest].element)
		smallest = r;
	if (smallest != i){
		swap(&minHeap->harr[i], &minHeap->harr[smallest]);
		MinHeapify(minHeap, smallest);
	}
}

void initMinHeap(MinHeap *minHeap, MinHeapNode *a, int size)
{
	int i; 
	minHeap->heap_size = size;
	minHeap->harr = a;  // store address of array
	i = (minHeap->heap_size - 1)/2;
	while (i >= 0)
	{
		MinHeapify(minHeap, i);
		i--;
	}
}
 

/* This function takes an array of arrays as an argument and
All arrays are assumed to be sorted. It merges them together
and prints the final sorted output. */

int *mergeKArrays(int **arr, int k)
{
	int i;
	int count; 
	int n = 4; 
	int *output = (int*) malloc (sizeof(int) *n*k); //To store output array
	MinHeap *hp;

	//Create a min heap with k heap nodes.  Every heap node
	//has first element of an array
	MinHeapNode *harr = (MinHeapNode*)malloc(sizeof(MinHeapNode) * k);
	for (i = 0; i < k; i++)
	{
		harr[i].element = arr[i][0]; //Store the first element
		harr[i].i = i; //index of array
		harr[i].j = 1; //Index of next element to be stored from array
	}

	hp = (MinHeap *) malloc(sizeof(MinHeap)); 
	initMinHeap(hp, harr, k);  //Create the heap

	//Now one by one get the minimum element from min
	//heap and replace it with next element of its array
	for (count = 0; count < n*k; count++)
	{
		//Get the minimum element and store it in output
		MinHeapNode root = getMin(hp);
		output[count] = root.element;

		//Find the next elelement that will replace current
		//root of heap. The next element belongs to same
		//array as the current root.
		if (root.j < n)
		{
			root.element = arr[root.i][root.j];
			root.j += 1;
		}
		//If root was the last element of its array
		else root.element =  INT_MAX; //INT_MAX is for infinite

		//Replace root with next element of array
		replaceMin(hp, root);
	}

	return output;
}

/* Merge multi property lists in freqCSset. This is used for forming mergeCS */
#define INIT_MERGELIST_SIZE 100 

oid* mergeMultiPropList(CSset *freqCSset, int *freqIdList, int k, int *numCombinedP)
{
	int i;
	//int j;
	MinHeap *hp;
	int freqIdx; 
	MinHeapNode *harr;
	int numAllocation = INIT_MERGELIST_SIZE;
	oid *tmp;

	oid *output = (oid *) malloc(sizeof(oid) * numAllocation); // Output array

	/*
	printf("Input list: \n");
	for (i = 0; i < k; i++){
		printf(" List %d: ",i);
		freqIdx = freqIdList[i];
		for (j = 0; j < freqCSset->items[freqIdx].numProp; j++){
			printf("  "BUNFMT, freqCSset->items[freqIdx].lstProp[j]);
		}
		printf("\n");
	}
	*/

	//Create a min heap with k heap nodes.  Every heap node
	//has first element of an array
	harr = (MinHeapNode*)malloc(sizeof(MinHeapNode) * k);
	for (i = 0; i < k; i++)
	{
		harr[i].element = freqCSset->items[freqIdList[i]].lstProp[0]; //Store the first element
		harr[i].i = i; //index of array
		harr[i].j = 1; //Index of next element to be stored from array
	}

	hp = (MinHeap *) malloc(sizeof(MinHeap)); 
	initMinHeap(hp, harr, k);  //Create the heap

	//Now one by one get the minimum element from min
	//heap and replace it with next element of its array
	*numCombinedP = 0;
	while (1)
	{
		//Get the minimum element and store it in output
		MinHeapNode root = getMin(hp);
		if (root.element == INT_MAX) break; 
		
		if (output[*numCombinedP - 1] != root.element){		//Only append the distinct prop to combined list
			if (*numCombinedP == numAllocation){
				numAllocation += INIT_MERGELIST_SIZE;		
				tmp = realloc(output, sizeof(oid) * numAllocation); 
				output = (oid*)tmp; 
			}
			output[*numCombinedP] = root.element;
			(*numCombinedP)++;
		}

		//Find the next elelement that will replace current
		//root of heap. The next element belongs to same
		//array as the current root.
		freqIdx = freqIdList[root.i]; 
		if (root.j < freqCSset->items[freqIdx].numProp)
		{
			root.element = freqCSset->items[freqIdx].lstProp[root.j];
			root.j += 1;
		}
		//If root was the last element of its array
		else root.element =  INT_MAX; //INT_MAX is for infinite

		//Replace root with next element of array
		replaceMin(hp, root);
	}
	
	/*
	printf("Output merge propList (%d) : \n", *numCombinedP);
	for (i = 0; i < *numCombinedP; i++){
		printf(BUNFMT "  ",output[i]);
	}
	printf("\n");
	*/
	return output;
}
