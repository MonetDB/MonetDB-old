/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _MAL_LINKER_H
#define _MAL_LINKER_H

#define MAL_EXT ".mal"
#define SQL_EXT ".sql"

#ifndef NATIVE_WIN32
#include <dlfcn.h>
#else
#define RTLD_LOCAL  0
#define RTLD_LAZY   1
#define RTLD_NOW    2
#define RTLD_GLOBAL 4
#endif

mal_export str initLinker(const char* path);
mal5_export MALfcn getAddress(str fcnname);
mal5_export char *MSP_locate_sqlscript(const char *mod_name, bit recurse);
mal5_export str loadLibrary(str modulename, int flag);
mal5_export char *locate_file(const char *basename, const char *ext, bit recurse);
mal5_export int malLibraryEnabled(str name);
mal5_export char* malLibraryHowToEnable(str name);
#endif /* _MAL_LINKER_H */
