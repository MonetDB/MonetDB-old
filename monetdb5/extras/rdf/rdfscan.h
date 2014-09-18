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
 * Copyright August 2008-2014 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * @f rdf
 * @a Minh-Duc Pham
 *
 * @* The RDFscan operator for MonetDB/RDF
 *
 */


#ifndef _RDF_SCAN_
#define _RDF_SCAN_

typedef struct RdfScanParams {
	char	schema[20]; 
	char** 	lstRPs; 	/* List of required params */	
	char**	lstLow;	/* Low bound of RP*/
	char**	lstHi; 	/* High bound of RP*/
	int	numRP; 	/* Number of required params */
	char** 	lstOPs;	/* List of optional params */
	int	numOP;
	
} RdfScanParams; 


rdf_export str
RDFscan(str query, str schema); 

#endif /* RDFSCAN */
