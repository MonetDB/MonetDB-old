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

/* This contains graph algorithms for the graph formed by CS's and their relationships */

#include "monetdb_config.h"
#include "mal_exception.h"
#include "url.h"
#include "tokenizer.h"
#include <gdk.h>
#include <rdfparams.h>
#include <string.h>

int dimensionFactor; 
int upperboundNumTables;
float generalityThreshold; 
float simTfidfThreshold;

void createDefaultParamsFile(void){
	
	FILE *paramFile;
	
	paramFile = fopen("params.ini", "wt");
	
	fprintf(paramFile, "dimensionFactor 1000\n");
	fprintf(paramFile, "upperboundNumTables 1000");
	fprintf(paramFile, "simTfidfThreshold 0.75");

	fclose(paramFile); 
}

void readParamsInput(void){
	FILE *pf;
	char variable[80];
	char value[80];

	pf = fopen("params.ini","r");
	
	if (pf == NULL){
		printf("No input parameter file found!");
		return; 
	}

	while (!feof(pf)){
		if(fscanf(pf, "%s %s", variable, value) == 2){
			if (strcmp(variable, "dimensionFactor") == 0){
				dimensionFactor = atoi(value);
				printf("dimensionFactor = %d\n",dimensionFactor);
			}
			else if (strcmp(variable, "upperboundNumTables") == 0){
				upperboundNumTables = atoi(value);
				printf("upperboundNumTables = %d\n", upperboundNumTables);
			}
			else if (strcmp(variable, "simTfidfThreshold") == 0){
				simTfidfThreshold = atof(value);
				printf("simTfidfThreshold = %f\n", simTfidfThreshold);
			}
		}
	}

	
	if (upperboundNumTables != 0){
		generalityThreshold = (float) 1 / (float)upperboundNumTables; 
		printf("generalityThreshold = %f\n",generalityThreshold);
	}
	else{ //default
		generalityThreshold = 0.001; 
	}


}
