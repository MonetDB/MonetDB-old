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

#ifndef _PETRINET_
#define _PETRINET_
#include "mal_interpreter.h"
#include "sql_scenario.h"
#include "basket.h"

#define _DEBUG_PETRINET_ if(1)

#define PNout mal_clients[1].fdout
/*#define  _BASKET_SIZE_*/

#ifdef WIN32
#ifndef LIBDATACELL
#define iot_export extern __declspec(dllimport)
#else
#define iot_export extern __declspec(dllexport)
#endif
#else
#define iot_export extern
#endif

#define PNINIT 0
#define PNRUNNING 1
#define PNREADY 2       /* not active now */

#define PAUSEDEFAULT 1000

iot_export str PNregisterInternal(Client cntxt, MalBlkPtr mb);
iot_export str PNregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str PNactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str PNdeactivate(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str PNcycles(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str PNdrop(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str PNdump(void *ret);

iot_export str PNperiod(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

iot_export str PNanalysis(Client cntxt, MalBlkPtr mb, int pn);
iot_export str PNanalyseWrapper(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
iot_export str PNtable(bat *modnameId, bat *fcnnameId, bat *statusId, bat *seenId, bat *cyclesId, bat *eventsId, bat *timeId, bat * errorId);
iot_export str PNinputplaces(bat *schemaId, bat *streamId, bat *modnameId, bat *fcnnameId);
iot_export str PNoutputplaces(bat *schemaId, bat *streamId, bat *modnameId, bat *fcnnameId);
#endif

