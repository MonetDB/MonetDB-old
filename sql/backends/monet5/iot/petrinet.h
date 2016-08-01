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
#include "iot.h"
#include "basket.h"

//#define _DEBUG_PETRINET_ 

#define PNINIT 0
#define PNRUNNING 1	   /* query is running */
#define PNWAIT 2       /* wait for data */
#define PNPAUSED 3     /* not active now */
#define PNSTOP 4	  /* stop all activity */

#define PAUSEDEFAULT 1000

sql5_export str PNregisterInternal(Client cntxt, MalBlkPtr mb, int calls);
sql5_export str PNregister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNderegister(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNresume(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNpause(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNwait(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNcycles(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNshow(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
sql5_export str PNstop(void);
sql5_export str PNdump(void *ret);
sql5_export str PNheartbeat(str mod, str fcn, lng ticks);

sql5_export str PNperiod(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);

sql5_export str PNanalysis(Client cntxt, MalBlkPtr mb, int pn);
sql5_export str PNtable(bat *modnameId, bat *fcnnameId, bat *statusId, bat *seenId, bat *cyclesId, bat *eventsId, bat *timeId, bat * errorId);
sql5_export str PNinputplaces(bat *schemaId, bat *streamId, bat *modnameId, bat *fcnnameId);
sql5_export str PNoutputplaces(bat *schemaId, bat *streamId, bat *modnameId, bat *fcnnameId);
#endif

