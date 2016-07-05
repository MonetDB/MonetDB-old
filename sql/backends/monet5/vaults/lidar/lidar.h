/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

#ifndef _LIDAR_
#define _LIDAR_
#include "mal.h"
#include "mal_client.h"

#ifdef WIN32
#ifndef LIBLIDAR
#define lidar_export extern __declspec(dllimport)
#else
#define lidar_export extern __declspec(dllexport)
#endif
#else
#define lidar_export extern
#endif

lidar_export str LIDARtest(int *res, str *fname);
lidar_export str LIDARdir(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARdirpat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARattach(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARloadTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARexportTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARprelude(void *ret);
lidar_export str LIDARCheckTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARAnalyzeTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str mvc_lidar_bind_wrap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);
lidar_export str LIDARTid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci);


typedef struct lidar_header_info {
	/* Table */
	int recordsCount;
	int pointRecordsCount;
	int dataOffset;
	int headerPadding;
	int byteSize;
	int baseByteSize;
	int creationDOY;
	int creationYear;
	int reserved;
	int dataRecordLength;
	int headerSize;
	int fileSourceId;
	int versionMajor;
	int versionMinor;
	int dataFormatId;
	str WKT;
	str WKT_CompoundOK;
	str proj4;
	/* Columns */
	double scaleX;
	double scaleY;
	double scaleZ;
	double offsetX;
	double offsetY;
	double offsetZ;
	double minX;
	double minY;
	double minZ;
	double maxX;
	double maxY;
	double maxZ;
} lidar_info;

typedef struct either_lidar_header {
	str msg;
	lidar_info *hi;
} lidar_header;

typedef enum {
    LIDAR_TABLE_UNLOADED,
    LIDAR_TABLE_LOADED,
    LIDAR_TABLE_ANALYZE,
    LIDAR_TABLE_DONE
} LIDAR_STATUS;


lidar_header *LIDARopenFile(str fname);
lidar_header *LIDARopenDir(str fname);
#endif
