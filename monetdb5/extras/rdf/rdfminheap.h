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

#ifndef _MINHEAP_H_
#define _MINHEAP_H_

#include "rdfschema.h"

//A min heap node
typedef struct MinHeapNode
{
	oid element;    //The element to be stored
	int i;  	//index of the array from which the element is taken
	int j;  	//index of the next element to be picked from array
} MinHeapNode;


//A struct for Min Heap
typedef struct MinHeap
{
	MinHeapNode *harr;  //pointer to array of elements in heap
	int heap_size;  //size of min heap
} MinHeap; 

//to get index of left child of node at index i
rdf_export int leftchild(int i);

//to get index of right child of node at index i
rdf_export int rightchild(int i);

//to get the root
rdf_export MinHeapNode getMin(MinHeap *minHeap);

// A utility function to swap two elements
rdf_export void swap(MinHeapNode *x, MinHeapNode *y);


// This method assumes that the subtrees are already heapified
rdf_export void MinHeapify(MinHeap *minHeap, int i);

//to replace root with new node x and heapify() new root
rdf_export void replaceMin(MinHeap *minHeap, MinHeapNode x);


rdf_export void initMinHeap(MinHeap *minHeap, MinHeapNode *a, int size);
 
rdf_export int* mergeKArrays(int **arr, int k);

rdf_export oid* mergeMultiPropList(CSset *freqCSset, int *freqIdList, int num, int *numCombinedP);  /* Merge the lstProp from multi freqCSs */

#endif /* _MINHEAP_H_ */
