/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2016 MonetDB B.V.
 */
/*
 * Authors: R. Goncalves
 *
 * This module contains primitives for accessing data in GADGET file format.
 */

#ifndef _GADGET_
#define _GADGET_
#include "sql.h"
#include "mal.h"
#include "mal_client.h"
#include <sys/stat.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include <sys/mman.h>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "vault.h"
#include "peano.h"

#ifdef WIN32
#ifndef LIBGADGET
#define gadget_export extern __declspec(dllimport)
#else
#define gadget_export extern __declspec(dllexport)
#endif
#else
#define gadget_export extern
#endif

#define MRSNAP_POS_OFFSET 268
#define MRSNAP_VEL_OFFSET 276  // +sum(npart)*12
#define MRSNAP_ID_OFFSET 284 // +sum(npart)*24

/* next assumes there are only positions, velocities and identifiers in a file */
#define MRSNAP_HASTHTABLE_OFFSET 308 // +sum(npart)*32

/*Default number of PHBins*/
#define MRS_PHBin_DEFAULT 5000
#define MRS_PARTICLE_DEFAULT 5000

gadget_export str gadgetTest(int *res, str *fname);
gadget_export str gadgetListDir(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetListDirAll(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetListDirPat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetAttach(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetAttachAll(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetLoadTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetLoadTableAll(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetLinksTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetExportTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetCheckTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetAnalyzeTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetPHkeyConvert(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
gadget_export str gadgetPHkeyInvert(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
#endif

typedef unsigned char uchar;
