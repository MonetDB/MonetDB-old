/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

/*
 * @-
 * @+ Implementation
 */
#ifndef _VAULT_H
#define _VAULT_H
#include "mal.h"
#include "mtime.h"
#include "clients.h"

#ifdef WIN32
#ifndef LIBVAULT
#define vault_export extern __declspec(dllimport)
#else
#define vault_export extern __declspec(dllexport)
#endif
#else
#define vault_export extern
#endif

#define _VAULT_DEBUG_

//vault_export str VLTprelude(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
vault_export str VLTprelude(void *ret);
vault_export str VLTimport(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
vault_export str VLTsetLocation(str *ret, str *src);
vault_export str VLTgetLocation(str *ret);
vault_export str VLTremove(timestamp *ret, str *t);
vault_export str VLTbasename(str *ret, str *fnme, str *splot);
vault_export str VLTepilogue(void *ret);

vault_export str VLTcheckTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
vault_export str VLTanalyzeTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
vault_export str mvc_VLT_bind_wrap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
vault_export str VLTTid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);


vault_export char vaultpath[BUFSIZ];

typedef enum {
    VAULT_TABLE_UNLOADED,
    VAULT_TABLE_LOADED,
    VAULT_TABLE_ANALYZE,
    VAULT_TABLE_DONE
} VAULT_STATUS;

typedef enum {
    VAULT_LIDAR_READER = 1,
    VAULT_GADGET_READER = 2
} VAULT_READER;

#endif /* _VAULT_H */
