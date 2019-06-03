/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _MAL_PROFILER_H
#define _MAL_PROFILER_H

#include "mal_client.h"

#ifndef NATIVE_WIN32
# include <sys/times.h>
# include <sys/resource.h>
typedef struct rusage Rusage;
#endif

mal5_export int malProfileMode;

mal5_export void initProfiler(void);
mal5_export str openProfilerStream(stream *fd, int mode);
mal5_export str closeProfilerStream(void);

mal5_export void profilerEvent(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, int start, str usrname);

mal5_export str startProfiler(void);
mal5_export str stopProfiler(void);
mal5_export str startTrace(str path);
mal5_export str stopTrace(str path);
mal5_export void setHeartbeat(int delay);
mal5_export void initHeartbeat(void);
mal5_export void profilerHeartbeatEvent(char *alter);
mal5_export int getprofilerlimit(void);
mal5_export void setprofilerlimit(int limit);

mal5_export void MPresetProfiler(stream *fdout);

mal5_export void clearTrace(void);
mal5_export int TRACEtable(BAT **r);
mal5_export int initTrace(void);
mal5_export str cleanupTraces(void);
mal5_export BAT *getTrace(const char *ev);

mal5_export lng getDiskReads(void);
mal5_export lng getDiskWrites(void);
mal5_export lng getUserTime(void);
mal5_export lng getSystemTime(void);
mal5_export void profilerGetCPUStat(lng *user, lng *nice, lng *sys, lng *idle, lng *iowait);
#endif
