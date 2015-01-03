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

#ifndef _RDFSTACK_H
#define _RDFSTACK_H

#include <gdk.h>

#ifdef WIN32
#ifndef LIBRDF
#define rdf_export extern __declspec(dllimport)
#else
#define rdf_export extern __declspec(dllexport)
#endif
#else
#define rdf_export extern
#endif

typedef int stElementT;	//stack element type (vertex Id)

typedef struct {
  stElementT *elements;
  int maxSize;
  int top;
} stackT;

rdf_export
stackT* initStack(int maxSize);

rdf_export
void stFree(stackT *stackP);

rdf_export
void stPush(stackT *stackP, stElementT element);

rdf_export
stElementT stPop(stackT *stackP);

rdf_export
int isEmptyStack(stackT *stackP);

rdf_export
int isFullStack(stackT *stackP);

rdf_export
int isInStack(stackT *stackP, stElementT element);

#endif  /* not defined _STACK_H */

