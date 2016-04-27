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
 * Copyright August 2008-2016 MonetDB B.V.
 * All Rights Reserved.
 */

#ifndef _BASKETS_
#define _BASKETS_

#include "monetdb_config.h"
#include "mal.h"
#include "mal_interpreter.h"
#include "sql.h"

#ifdef WIN32
#ifndef LIBDATACELL
#define iot_export extern __declspec(dllimport)
#else
#define iot_export extern __declspec(dllexport)
#endif
#else
#define iot_export extern
#endif

/* #define _DEBUG_DATACELL     debug this module */
#define BSKTout GDKout
#define MAXBSKT 64

typedef struct{
	str schema_name;	/* schema for the basket */
	str table_name;	/* table that represents the basket */
	sql_schema *schema;
	sql_table *table;
	str *cols;

	int threshold ; /* bound to determine scheduling eligibility */
	int winsize, winstride; /* sliding window operations */
	lng timeslice, timestride; /* temporal sliding window, determined by first temporal component */
	lng beat;	/* milliseconds delay */
	int count;	/* number of events available in basket */

	/* statistics */
	int status;		/* (DE)ACTIVATE */
	timestamp seen;
	int events; /* total number of events grabbed */
	int cycles; 
	/* collected errors */
	BAT *errors;
} *Basket, BasketRec;


/* individual streams can be paused and restarted */
#define BSKTAVAILABLE   1      
#define BSKTPAUSED    2
#define BSKTLOCKED    3

iot_export BasketRec *baskets;

iot_export str BSKTbind(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTreset(void *ret);
iot_export int BSKTlocate(str sch, str tbl);
iot_export str BSKTdump(void *ret);

iot_export str BSKTactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTdeactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

iot_export str BSKTthreshold(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTbeat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTwindow(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

iot_export str BSKTtable( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTtableerrors(bat *nmeId, bat *errorId);
iot_export str BSKTfinish( Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

//iot_export str BSKTnewbasket(sql_schema *s, sql_table *t);
iot_export str BSKTdrop(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

iot_export int BSKTlocate(str sch, str tbl);
iot_export int BSKTunlocate(str sch, str tbl);
iot_export int BSKTlocate(str sch, str tbl);
iot_export str BSKTappend(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTdelete(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTclear(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTcommit(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTpushBasket(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str BSKTupdate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

#endif
