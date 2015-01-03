/*
 * The elements of this file are subject to the MonetDB Public License
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
#include <gdk.h>
#include "rdfstack.h"       

//As this stack is currently used for the join graph
//The maxSize shoule be the MAX_JGRAPH_NODENUMBER
//
stackT* initStack(int maxSize)
{
	stackT *stackP = NULL; 
	stElementT *elements;

	stackP = (stackT *) malloc(sizeof(stackT)); 

	elements = (stElementT *)malloc(sizeof(stElementT) * maxSize);

	if (elements == NULL) {
		fprintf(stderr, "Failed to initialize stack.\n");
		exit(1);  
	}

	stackP->elements = elements;
	stackP->maxSize = maxSize;
	stackP->top = -1;  /* I.e., empty */

	return stackP; 
}

void stFree(stackT *stackP)
{
	free(stackP->elements);

	stackP->elements = NULL;
	stackP->maxSize = 0;
	stackP->top = -1;  

	free(stackP); 
}

void stPush(stackT *stackP, stElementT element)
{
	if (isFullStack(stackP)) {
		fprintf(stderr, "Stack is full. Cannot push more element. \n");
		exit(1);  /* Exit, returning error code. */
	}


	stackP->elements[++stackP->top] = element;
}

stElementT stPop(stackT *stackP)
{
	if (isEmptyStack(stackP)) {
		fprintf(stderr, "Empty stack!!!.\n");
		exit(1);  /* Exit, returning error code. */
	}

	return stackP->elements[stackP->top--];
}

int isEmptyStack(stackT *stackP)
{
	return stackP->top < 0;
}

int isFullStack(stackT *stackP)
{
	return stackP->top >= stackP->maxSize - 1;
}

int isInStack(stackT *stackP, stElementT element){
	int i = 0; 
	for (i = 0; i <= stackP->top; i++){
		if (stackP->elements[i] == element) return 1; 	
	}

	return 0; 
}

