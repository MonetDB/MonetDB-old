/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/*
 * Author: K. Kyzirakos
 * Author: P. Koutsourakis
 * Author: R. Gonclaves
 *
 * This module contains primitives for accessing data in LiDAR file format.
 */

#include <monetdb_config.h>
#include <glob.h>

/* clash with GDK? */
// #undef ttype

#include <liblas/capi/liblas.h>
#include <liblas/capi/las_version.h>
#include <liblas/capi/las_config.h>

#include <sys/stat.h>
#include <dirent.h>

#include <lidar.h>
#include <mutils.h>
#include <sql_mvc.h>
#include <sql_scenario.h>
#include <sql.h>
#include <clients.h>
#include <mal_exception.h>

static MT_Lock mt_lidar_lock;

#define LIDAR_LOCK MT_lock_set(&mt_lidar_lock)
#define LIDAR_UNLOCK MT_lock_unset(&mt_lidar_lock)

#define LIDAR_INS_COL "INSERT INTO lidar_columns(id, name, type, units, number, table_id) \
	 VALUES(%d,'%s','%s','%s',%d,%d);"
#define FILE_INS "INSERT INTO lidar_files(id, name) VALUES (%d, '%s');"
#define DEL_TABLE "DELETE FROM lidarfiles;"
#define ATTACHDIR "CALL lidarattach('%s');"

#ifndef NDEBUG
static
void print_lidar_header(FILE *file, LASHeaderH header, const char* file_name, int bSkipVLR, int bWKT) {

	char *pszSignature = NULL;
	char *pszProjectId = NULL;
	char *pszSystemId = NULL;
	char *pszSoftwareId = NULL;
	char *pszProj4 = NULL;
	char *pszWKT = NULL;
	char *pszVLRUser = NULL;
	char *pszVLRDescription = NULL;
	unsigned short nVLRLength = 0;
	unsigned short nVLRRecordId = 0;
	LASVLRH pVLR = NULL;
	LASSRSH pSRS = NULL;
	unsigned int nVLR = 0;
	int i = 0;

	char minor, major;

	pszSignature = LASHeader_GetFileSignature(header);
	pszProjectId = LASHeader_GetProjectId(header);
	pszSystemId = LASHeader_GetSystemId(header);
	pszSoftwareId = LASHeader_GetSoftwareId(header);

	pSRS = LASHeader_GetSRS(header);
	pszProj4 = LASSRS_GetProj4(pSRS);
	pszWKT = LASSRS_GetWKT_CompoundOK(pSRS);

	nVLR = LASHeader_GetRecordsCount(header);
	major = LASHeader_GetVersionMajor(header);
	minor = LASHeader_GetVersionMinor(header);

	fprintf(file, "\n---------------------------------------------------------\n");
	fprintf(file, "  Header Summary\n");
	fprintf(file, "---------------------------------------------------------\n");

	fprintf(file, "  File Name: %s\n", file_name);

	if (strcmp(pszSignature,"LASF") !=0) {
		LASError_Print("File signature is not 'LASF'... aborting");
		exit(1);
	}
	fprintf(file, "  Version:                    %d.%d\n", major, minor);

	fprintf(file, "  Source ID:                  %d\n",
			LASHeader_GetFileSourceId(header) ) ;

	fprintf(file, "  Reserved:                   %d\n",
			LASHeader_GetReserved(header) );

	fprintf(file, "  Project ID/GUID:           '%s'\n",
			pszProjectId);

	fprintf(file, "  System Identifier:         '%s'\n",
			pszSystemId);

	fprintf(file, "  Generating Software:       '%s'\n",
			pszSoftwareId);

	fprintf(file, "  File Creation Day/Year:    %d/%d\n",
			LASHeader_GetCreationDOY(header),
			LASHeader_GetCreationYear(header));

	fprintf(file, "  Header Size                %d\n",
			LASHeader_GetHeaderSize(header));

	fprintf(file, "  Offset to Point Data       %d\n",
			LASHeader_GetDataOffset(header));

	fprintf(file, "  Number Var. Length Records %d\n",
			LASHeader_GetRecordsCount(header));

	fprintf(file, "  Point Data Format          %d\n",
			LASHeader_GetDataFormatId(header));

	fprintf(file, "  Point Data Record Length   %d\n",
			LASHeader_GetDataRecordLength(header));

	fprintf(file, "  Number of Point Records    %d\n",
			LASHeader_GetPointRecordsCount(header));

	fprintf(file, "  Number of Points by Return %d %d %d %d %d\n",
			LASHeader_GetPointRecordsByReturnCount(header, 0),
			LASHeader_GetPointRecordsByReturnCount(header, 1),
			LASHeader_GetPointRecordsByReturnCount(header, 2),
			LASHeader_GetPointRecordsByReturnCount(header, 3),
			LASHeader_GetPointRecordsByReturnCount(header, 4));

	fprintf(file, "  Scale Factor X Y Z         %.6g %.6g %.6g\n",
			LASHeader_GetScaleX(header),
			LASHeader_GetScaleY(header),
			LASHeader_GetScaleZ(header));

	fprintf(file, "  Offset X Y Z               %.6f %.6f %.6f\n",
			LASHeader_GetOffsetX(header),
			LASHeader_GetOffsetY(header),
			LASHeader_GetOffsetZ(header));

	fprintf(file, "  Min X Y Z                  %.6f %.6f %.6f\n",
			LASHeader_GetMinX(header),
			LASHeader_GetMinY(header),
			LASHeader_GetMinZ(header));

	fprintf(file, "  Max X Y Z                  %.6f %.6f %.6f\n",
			LASHeader_GetMaxX(header),
			LASHeader_GetMaxY(header),
			LASHeader_GetMaxZ(header));

	fprintf(file, " Spatial Reference           %s\n",
			pszProj4);
	if (bWKT)
	{
		fprintf(file, "%s", pszWKT);
		fprintf(file, "%s", "\n");
	}
	if (nVLR && !bSkipVLR) {

		fprintf(file, "\n---------------------------------------------------------\n");
		fprintf(file, "  VLR Summary\n");
		fprintf(file, "---------------------------------------------------------\n");

		for (i = 0; i < (int)nVLR; i++) {
			pVLR = LASHeader_GetVLR(header, i);

			if (LASError_GetLastErrorNum()) {
				LASError_Print("Unable to fetch VLR");
				exit(1);
			}

			pszVLRUser = LASVLR_GetUserId(pVLR);
			pszVLRDescription = LASVLR_GetDescription(pVLR);
			nVLRLength = LASVLR_GetRecordLength(pVLR);
			nVLRRecordId = LASVLR_GetRecordId(pVLR);


			fprintf(file, "   User: '%s' - Description: '%s'\n", pszVLRUser, pszVLRDescription);
			fprintf(file, "   ID: %d Length: %d\n\n", nVLRRecordId, nVLRLength);

			MT_lock_set(&mt_lidar_lock);
			LASVLR_Destroy(pVLR);
			MT_lock_unset(&mt_lidar_lock);
			pVLR = NULL;

			LASString_Free(pszVLRUser);
			LASString_Free(pszVLRDescription);
		}

	}
	LASString_Free(pszSignature);
	LASString_Free(pszProjectId);
	LASString_Free(pszSystemId);
	LASString_Free(pszSoftwareId);
	LASString_Free(pszProj4);
	LASString_Free(pszWKT);
	MT_lock_set(&mt_lidar_lock);
	LASSRS_Destroy(pSRS);
	MT_lock_unset(&mt_lidar_lock);
}
#endif

static void
LIDARinitCatalog(mvc *m)
{
	sql_schema *sch;
	sql_table *lidar_fl, *lidar_tbl, *lidar_col, *vault_journal;

	sch = mvc_bind_schema(m, "sys");

	lidar_fl = mvc_bind_table(m, sch, "lidar_files");
	if (lidar_fl == NULL) {
		lidar_fl = mvc_create_table(m, sch, "lidar_files", tt_table, 0, SQL_PERSIST, 0, 2);
		mvc_create_column_(m, lidar_fl, "id", "int", 32);
		mvc_create_column_(m, lidar_fl, "name", "varchar", 255);
	}

	lidar_tbl = mvc_bind_table(m, sch, "lidar_tables");
	if (lidar_tbl == NULL) {
		lidar_tbl = mvc_create_table(m, sch, "lidar_tables", tt_table, 0, SQL_PERSIST, 0, 21);
		mvc_create_column_(m, lidar_tbl, "id", "int", 32);
		mvc_create_column_(m, lidar_tbl, "file_id", "int", 32);
		mvc_create_column_(m, lidar_tbl, "name", "varchar", 255);
		mvc_create_column_(m, lidar_tbl, "FileSourceId", "int", 32);
		mvc_create_column_(m, lidar_tbl, "VersionMajor", "int", 16);
		mvc_create_column_(m, lidar_tbl, "VersionMinor", "int", 16);
		mvc_create_column_(m, lidar_tbl, "DataFormatId", "int", 16);
		mvc_create_column_(m, lidar_tbl, "CreationDOY", "int", 32);
		mvc_create_column_(m, lidar_tbl, "CreationYear", "int", 32);
		mvc_create_column_(m, lidar_tbl, "RecordsCount", "int", 32);
		mvc_create_column_(m, lidar_tbl, "PointRecordsCount", "int", 32);
		mvc_create_column_(m, lidar_tbl, "DataOffset", "int", 32);
		mvc_create_column_(m, lidar_tbl, "HeaderPadding", "int", 32);
		mvc_create_column_(m, lidar_tbl, "Reserved", "int", 32);
		mvc_create_column_(m, lidar_tbl, "DataRecordLength", "int", 32);
		mvc_create_column_(m, lidar_tbl, "HeaderSize", "int", 32);
		mvc_create_column_(m, lidar_tbl, "ByteSize", "int", 32);
		mvc_create_column_(m, lidar_tbl, "BaseByteSize", "int", 32);
		mvc_create_column_(m, lidar_tbl, "WKT", "varchar", 255);
		mvc_create_column_(m, lidar_tbl, "WKT_CompoundOK", "varchar", 255);
		mvc_create_column_(m, lidar_tbl, "Proj4", "varchar", 255);
		mvc_create_column_(m, lidar_tbl, "LoadParams", "int", 32);
	}

	lidar_col = mvc_bind_table(m, sch, "lidar_columns");
	if (lidar_col == NULL) {
		lidar_col = mvc_create_table(m, sch, "lidar_columns", tt_table, 0, SQL_PERSIST, 0, 15);
		mvc_create_column_(m, lidar_col, "id", "int", 32);
		mvc_create_column_(m, lidar_col, "file_id", "int", 32);
		mvc_create_column_(m, lidar_col, "table_id", "int", 32);
		mvc_create_column_(m, lidar_col, "ScaleX", "double", 64);
		mvc_create_column_(m, lidar_col, "ScaleY", "double", 64);
		mvc_create_column_(m, lidar_col, "ScaleZ", "double", 64);
		mvc_create_column_(m, lidar_col, "OffsetX", "double", 64);
		mvc_create_column_(m, lidar_col, "OffsetY", "double", 64);
		mvc_create_column_(m, lidar_col, "OffsetZ", "double", 64);
		mvc_create_column_(m, lidar_col, "MinX", "double", 64);
		mvc_create_column_(m, lidar_col, "MinY", "double", 64);
		mvc_create_column_(m, lidar_col, "MinZ", "double", 64);
		mvc_create_column_(m, lidar_col, "MaxX", "double", 64);
		mvc_create_column_(m, lidar_col, "MaxY", "double", 64);
		mvc_create_column_(m, lidar_col, "MaxZ", "double", 64);
		mvc_create_column_(m, lidar_col, "PrecisionX", "int", 16);
		mvc_create_column_(m, lidar_col, "PrecisionY", "int", 16);
		mvc_create_column_(m, lidar_col, "PrecisionZ", "int", 16);
	}

	vault_journal = mvc_bind_table(m, sch, "vault_journal");
	if (vault_journal == NULL) {
		vault_journal = mvc_create_table(m, sch, "vault_journal", tt_table, 0, SQL_PERSIST, 0, 4);
		mvc_create_column_(m, vault_journal, "table_id", "int", 32);
		mvc_create_column_(m, vault_journal, "table_name", "varchar", 255);
		mvc_create_column_(m, vault_journal, "vault_reader_id", "int", 32);
		mvc_create_column_(m, vault_journal, "vault_type", "varchar", 255);
		mvc_create_column_(m, vault_journal, "version", "varchar", 10);
	}
}

#if 0
static int
lidar2mtype(int t)
{
	(void) t;
	switch (t) {
	case TBIT:
	case TLOGICAL:
		return TYPE_bit;
	case TBYTE:
	case TSBYTE:
		return TYPE_bte;
	case TSTRING:
		return TYPE_str;
	case TUSHORT:
	case TSHORT:
		return TYPE_sht;
	case TUINT:
	case TINT:
		return TYPE_int;
	case TLONG:
	case TULONG:
	case TLONGLONG:
		return TYPE_lng;
	case TFLOAT:
		return TYPE_flt;
	case TDOUBLE:
		return TYPE_dbl;
		/* missing */
	case TCOMPLEX:
	case TDBLCOMPLEX:
		return -1;
	}
	return -1;
}
#endif

#if 0
static int
lidar2subtype(sql_subtype *tpe, int t, long rep, long wid)
{
	(void)rep;
	switch (t) {
	case TBIT:
	case TLOGICAL:
		sql_find_subtype(tpe, "boolean", 0, 0);
		break;
	case TBYTE:
	case TSBYTE:
		sql_find_subtype(tpe, "char", 1, 0);
		break;
	case TSTRING:
		sql_find_subtype(tpe, "varchar", (unsigned int)wid, 0);
		break;
	case TUSHORT:
	case TSHORT:
		sql_find_subtype(tpe, "smallint", 16, 0);
		break;
	case TUINT:
	case TINT:
		sql_find_subtype(tpe, "int", 32, 0);
		break;
	case TULONG:
	case TLONG:
	case TLONGLONG:
		sql_find_subtype(tpe, "bigint", 64, 0);
		break;
	case TFLOAT:
		sql_find_subtype(tpe, "real", 32, 0);
		break;
	case TDOUBLE:
		sql_find_subtype(tpe, "double", 51, 0);
		break;
		/* missing */
	case TCOMPLEX:
	case TDBLCOMPLEX:
		return -1;
	}
	return 1;
}
#endif

str LIDARexportTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str msg = MAL_SUCCEED;
//	int* res_id = *getArgReference_str(stk, pci, 1);
	str tname = *getArgReference_str(stk, pci, 1);
	str filename = *getArgReference_str(stk, pci, 2);
	str format = toLower(*getArgReference_str(stk, pci, 2));
	mvc *m = NULL;
	sql_trans *tr;
	sql_schema *sch;
	sql_table *tbl;
	BUN nrows = 0;
	BUN i;

	sql_column *cols[3];
	dbl *cols_dbl[3];
	BAT *bats_dbl[3];

	LASHeaderH header;
	LASWriterH writer;
	LASPointH point;

	if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
		return msg;
	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
		return msg;

	tr = m->session->tr;
	sch = mvc_bind_schema(m, "sys");

	/* First step: look if the table exists in the database. If the table is not in the database, the export function cannot continue */

	tbl = mvc_bind_table(m, sch, tname);
	if (tbl == NULL) {
		msg = createException (MAL, "lidar.exporttable", "Table %s is missing.\n", tname);
		return msg;
	}

	/* Bind the necessary columns */
	cols[0] = mvc_bind_column(m, tbl, "x");
	cols[1] = mvc_bind_column(m, tbl, "y");
	cols[2] = mvc_bind_column(m, tbl, "z");
	if (cols[0] == NULL || cols[1] == NULL || cols[2] == NULL) {
		msg = createException(MAL, "lidar.exporttable", "Could not locate a column with name 'x', 'y', or 'z'.");
		return msg;
	}
	//bats_dbl[0] = mvc_bind(m, *sname, *tname, *cname, *access);
	bats_dbl[0] = store_funcs.bind_col(tr, cols[0], 0);
	bats_dbl[1] = store_funcs.bind_col(tr, cols[1], 0);
	bats_dbl[2] = store_funcs.bind_col(tr, cols[2], 0);

	cols_dbl[0] = (dbl*)Tloc(bats_dbl[0], 0);
	cols_dbl[1] = (dbl*)Tloc(bats_dbl[1], 0);
	cols_dbl[2] = (dbl*)Tloc(bats_dbl[2], 0);

	nrows = store_funcs.count_col(tr, cols[0], 1);

	/* Populate the header */
	header = LASHeader_Create();
	LASHeader_SetCompressed(header, (strcmp(format, "laz") == 0));
/*
  LASHeader_SetCreationDOY
  LASHeader_SetCreationYear
  LASHeader_SetDataFormatId
  LASHeader_SetDataOffset
  LASHeader_SetDataRecordLength
  LASHeader_SetFileSourceId
  LASHeader_SetGUID
  LASHeader_SetHeaderPadding
  LASHeader_SetMax
  LASHeader_SetMin
  LASHeader_SetOffset
  LASHeader_SetPointRecordsByReturnCount
*/
	LASHeader_SetPointRecordsCount(header, nrows);
/*
  LASHeader_SetProjectId
  LASHeader_SetReserved
  LASHeader_SetScale
  LASHeader_SetSchema
*/
	LASHeader_SetSoftwareId(header, "MonetDB B.V.");
/*	LASHeader_SetSRS */
	LASHeader_SetSystemId(header, "MonetDB B.V.");
	LASHeader_SetVersionMajor(header, 1);
	LASHeader_SetVersionMinor(header, 2);

	/* Create the writer*/
	MT_lock_set(&mt_lidar_lock);
	writer = LASWriter_Create(filename, header, LAS_MODE_WRITE);

	// TODO: Failure is always an option
	MT_lock_unset(&mt_lidar_lock);


	/* Iterate over the table */
	for (i = 0; i < nrows; i++) {
		point = LASPoint_Create();
/*
  LASPoint_SetClassification
  LASPoint_SetColor
  LASPoint_SetData
  LASPoint_SetFlightLineEdge
  LASPoint_SetHeader
  LASPoint_SetIntensity
  LASPoint_SetNumberOfReturns
  LASPoint_SetPointSourceId
  LASPoint_SetRawX
  LASPoint_SetRawY
  LASPoint_SetRawZ
  LASPoint_SetReturnNumber
  LASPoint_SetScanAngleRank
  LASPoint_SetScanDirection
  LASPoint_SetScanFlags
  LASPoint_SetTime
  LASPoint_SetUserData
*/
		LASPoint_SetX(point, cols_dbl[0][i]);
		LASPoint_SetY(point, cols_dbl[1][i]);
		LASPoint_SetZ(point, cols_dbl[2][i]);

		LASWriter_WritePoint(writer, point);
		LASPoint_Destroy(point);
	}

	MT_lock_set(&mt_lidar_lock);
	LASHeader_Destroy(header);
	LASWriter_Destroy(writer);
	MT_lock_unset(&mt_lidar_lock);

	return MAL_SUCCEED;
}


str LIDARdir(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
#if 0
	str msg = MAL_SUCCEED;
	str dir = *getArgReference_str(stk, pci, 1);
	DIR *dp;
	struct dirent *ep;
	lidarfile *fptr;
	char *s;
	int status = 0;
	(void)mb;

	dp = opendir(dir);
	if (dp != NULL) {
		char stmt[BUFSIZ];
		char fname[BUFSIZ];

		s = stmt;

		while ((ep = readdir(dp)) != NULL && !msg) {
			snprintf(fname, BUFSIZ, "%s%s", dir, ep->d_name);
			status = 0;
			lidar_open_file(&fptr, fname, READONLY, &status);
			if (status == 0) {
				snprintf(stmt, BUFSIZ, ATTACHDIR, fname);
				msg = SQLstatementIntern(cntxt, &s, "lidar.listofdir", TRUE, FALSE, NULL);
				lidar_close_file(fptr, &status);
			}
		}
		(void)closedir(dp);
	} else
		msg = createException(MAL, "listdir", "Couldn't open the directory");

	return msg;
#endif
	return MAL_SUCCEED;
}

str LIDARdirpat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;
	return MAL_SUCCEED;
#if 0
	str msg = MAL_SUCCEED;
	str dir = *getArgReference_str(stk, pci, 1);
	str pat = *getArgReference_str(stk, pci, 2);
	lidarfile *fptr;
	char *s;
	int status = 0;
	glob_t globbuf;
	char fulldirectory[BUFSIZ];
	size_t j = 0;

	(void)mb;
	globbuf.gl_offs = 0;
	snprintf(fulldirectory, BUFSIZ, "%s%s", dir, pat);
	glob(fulldirectory, GLOB_DOOFFS, NULL, &globbuf);

	/*	fprintf(stderr,"#fulldir: %s \nSize: %lu\n",fulldirectory, globbuf.gl_pathc);*/

	if (globbuf.gl_pathc == 0)
		throw(MAL, "listdir", "Couldn't open the directory or there are no files that match the pattern");

	for (j = 0; j < globbuf.gl_pathc; j++) {
		char stmt[BUFSIZ];
		char fname[BUFSIZ];

		s = stmt;
		snprintf(fname, BUFSIZ, "%s", globbuf.gl_pathv[j]);
		status = 0;
		lidar_open_file(&fptr, fname, READONLY, &status);
		if (status == 0) {
			snprintf(stmt, BUFSIZ, ATTACHDIR, fname);
			msg = SQLstatementIntern(cntxt, &s, "lidar.listofdirpat", TRUE, FALSE, NULL);
			lidar_close_file(fptr, &status);
			break;
		}
	}

	return msg;
#endif
	return MAL_SUCCEED;
}


str
LIDARtest(int *res, str *fname)
{
	str msg = MAL_SUCCEED;
	LASReaderH reader;
	LASHeaderH header;

	LASError_Reset();
	MT_lock_set(&mt_lidar_lock);
	reader=LASReader_Create(*fname);
	MT_lock_unset(&mt_lidar_lock);

	if (LASError_GetErrorCount() != 0) {
		msg = createException(MAL, "lidar.test", "Error accessing LIDAR file %s (%s)",
							  *fname, LASError_GetLastErrorMsg());
	} else 	{
		header=LASReader_GetHeader(reader);
		*res=LASHeader_GetPointRecordsCount(header);
		MT_lock_set(&mt_lidar_lock);
		if (header != NULL) LASHeader_Destroy(header);
		if (reader != NULL) LASReader_Destroy(reader);
		MT_lock_unset(&mt_lidar_lock);
		if (LASError_GetErrorCount() != 0) {
			msg = createException(MAL, "lidar.test", "Error accessing LIDAR file %s (%s)",
								  *fname, LASError_GetLastErrorMsg());
		}
	}

	return msg;
}

typedef struct either_lidar_header {
	str msg;
	lidar_info *hi;
} lidar_header;


static str *
LIDARopenPath(str fname, int *len) {
	str *ret = NULL;
	struct stat buf;
	DIR *dir;
	struct dirent *dir_entry;
	str path;
	int idx = 0;

	stat(fname, &buf);
	if (S_ISDIR(buf.st_mode)) {
		*len = 0;
		dir = opendir(fname);
		while((dir_entry = readdir(dir)) != NULL) {
			if (dir_entry->d_type == DT_REG)
				(*len)++;
		}
		closedir(dir);
		ret = (str *)malloc((*len)*sizeof(str));
		path = (str)malloc(strlen(fname) + 256);
		dir = opendir(fname);
		while((dir_entry = readdir(dir)) != NULL) {
			if (dir_entry->d_type == DT_REG) {
				strncpy(path, fname, strlen(fname));
				path[strlen(fname)] = '\0';
				strncat(path, dir_entry->d_name, strlen(dir_entry->d_name));
				ret[idx++] = strdup(path);
#ifndef NDEBUG
				printf("file: %s %d\n", path, *len);
#endif
			}
		}
		closedir(dir);
		free(path);
	}
	else {
		*len = 1;
		ret = (str *)malloc(sizeof(str));
		ret[0] = strdup(fname);
	}

	return ret;
}

static lidar_header *
LIDARopenFile(str fname)
{
	lidar_header *res = (lidar_header *)malloc(sizeof(lidar_header));
	LASReaderH reader = NULL;
	LASHeaderH header = NULL;
	res->msg = NULL;
	res->hi = NULL;
	/* check if file exists */
	if (access(fname, F_OK) == -1) {
		res->msg = createException(MAL, "lidar.openfile", "File %s not found.", fname);
		return res;
	}

	/* open the LAS/LAZ file */
	MT_lock_set(&mt_lidar_lock);
	LASError_Reset();
	reader = LASReader_Create(fname);
	MT_lock_unset(&mt_lidar_lock);
	if (LASError_GetErrorCount() != 0) {
		res->msg = createException(MAL, "lidar.openfile", "Error accessing LIDAR file %s (%s)",
							  fname, LASError_GetLastErrorMsg());
		return res;
	}

	/* get the header */
	header = LASReader_GetHeader(reader);
	if (!header) {
		res->msg = createException(MAL, "lidar.openfile", "Error accessing LIDAR file %s (%s)",
							  fname, LASError_GetLastErrorMsg());
		return res;
	}
#ifndef NDEBUG
	print_lidar_header(stderr, header, fname, 0, 0);
#endif

	res->hi = (lidar_info *)malloc(sizeof(lidar_info));
	/* read values from the header */
	res->hi->fileSourceId = LASHeader_GetFileSourceId(header);
	res->hi->versionMajor = LASHeader_GetVersionMajor(header);
	res->hi->versionMinor = LASHeader_GetVersionMinor(header);
	res->hi->dataFormatId = LASHeader_GetDataFormatId(header);
	res->hi->creationDOY = LASHeader_GetCreationDOY(header);
	res->hi->creationYear = LASHeader_GetCreationYear(header);
	res->hi->recordsCount = LASHeader_GetRecordsCount(header);
	res->hi->pointRecordsCount = LASHeader_GetPointRecordsCount(header);
	res->hi->dataOffset = LASHeader_GetDataOffset(header);
	res->hi->headerPadding = LASHeader_GetHeaderPadding(header);
	res->hi->reserved = LASHeader_GetReserved(header);
	res->hi->dataRecordLength = LASHeader_GetDataRecordLength(header);
	res->hi->headerSize = LASHeader_GetHeaderSize(header);
	res->hi->byteSize     = LASSchema_GetByteSize(LASHeader_GetSchema(header));
	res->hi->baseByteSize = LASSchema_GetBaseByteSize(LASHeader_GetSchema(header));
	res->hi->WKT = LASSRS_GetWKT(LASHeader_GetSRS(header));
	res->hi->WKT_CompoundOK = LASSRS_GetWKT_CompoundOK(LASHeader_GetSRS(header));
	res->hi->proj4 = LASSRS_GetProj4(LASHeader_GetSRS(header));


	/* read data from the header */
	res->hi->scaleX = LASHeader_GetScaleX(header);
	res->hi->scaleY = LASHeader_GetScaleY(header);
	res->hi->scaleZ = LASHeader_GetScaleZ(header);
	res->hi->offsetX = LASHeader_GetOffsetX(header);
	res->hi->offsetY = LASHeader_GetOffsetY(header);
	res->hi->offsetZ = LASHeader_GetOffsetZ(header);
	res->hi->minX = LASHeader_GetMinX(header);
	res->hi->minY = LASHeader_GetMinY(header);
	res->hi->minZ = LASHeader_GetMinZ(header);
	res->hi->maxX = LASHeader_GetMaxX(header);
	res->hi->maxY = LASHeader_GetMaxY(header);
	res->hi->maxZ = LASHeader_GetMaxZ(header);

	MT_lock_set(&mt_lidar_lock);
	if (header != NULL) LASHeader_Destroy(header);
	if (reader != NULL) LASReader_Destroy(reader);
	MT_lock_unset(&mt_lidar_lock);

	return res;
}

typedef enum {
	PARAMS_ALL_OFF               =       0,
	PARAM_X_COORD                =     0x1, /* x or X */
	PARAM_Y_COORD                =     0x2, /* y or Y */
	PARAM_Z_COORD                =     0x4, /* z or Z */
	PARAM_N_RETURNS              =     0x8, /* n */
	PARAM_N_THIS_RETURN          =    0x10, /* r */
	PARAM_CLASSIFICATION_NUMBER  =    0x20, /* c */
	PARAM_USER_DATA              =    0x40, /* u */
	PARAM_POINT_SOURCE_ID        =    0x80, /* p */
	PARAM_EDGE_OF_FLIGHT_LINE    =   0x100, /* e */
	PARAM_DIRECTION_OF_SCAN_FLAG =   0x200, /* d */
	PARAM_RED_CHANNEL            =   0x400, /* R */
	PARAM_GREEN_CHANNEL          =   0x800, /* G */
	PARAM_BLUE_CHANNEL           =  0x1000, /* B */
	PARAM_VERTEX_INDEX           =  0x2000, /* M */
	PARAM_GPS_TIME               =  0x4000, /* t */
	PARAM_SCAN_ANGLE             =  0x8000, /* a */
	PARAM_INTENSITY              = 0x10000, /* i */
	PARAMS_END_SENTINEL
} ParameterValues;

typedef struct input_parameters {
	int cnum;
	int parameters;
} InputParameters;

/* Parse the parameter string that specifies what columns will be
 * read.
 */
static void
parse_parameters(str params, InputParameters *parsed) {
	/* x, y, and z are always loaded */
    char *p = NULL;
	parsed->cnum = 3;
	parsed->parameters = PARAM_X_COORD | PARAM_Y_COORD | PARAM_Z_COORD;

	if (params == NULL) {
		return;
	}

	for (p = params; *p != '\0'; p++) {
		switch (*p) {
		case 'x':
		case 'X':
		case 'y':
		case 'Y':
		case 'z':
		case 'Z':
			/* Just ignore these */
			break;
		case 't':
			if ((parsed->parameters & PARAM_GPS_TIME)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_GPS_TIME;
			parsed->cnum++;
			break;
		case 'a':
			if ((parsed->parameters & PARAM_SCAN_ANGLE)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_SCAN_ANGLE;
			parsed->cnum++;
			break;
		case 'i':
			if ((parsed->parameters & PARAM_INTENSITY)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_INTENSITY;
			parsed->cnum++;
			break;
		case 'n':
			if ((parsed->parameters & PARAM_N_RETURNS)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_N_RETURNS;
			parsed->cnum++;
			break;
		case 'r':
			if ((parsed->parameters & PARAM_N_THIS_RETURN)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_N_THIS_RETURN;
			parsed->cnum++;
			break;
		case 'c':
			if ((parsed->parameters & PARAM_CLASSIFICATION_NUMBER)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_CLASSIFICATION_NUMBER;
			parsed->cnum++;
			break;
		case 'u':
			if ((parsed->parameters & PARAM_USER_DATA)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_USER_DATA;
			parsed->cnum++;
			break;
		case 'p':
			if ((parsed->parameters & PARAM_POINT_SOURCE_ID)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_POINT_SOURCE_ID;
			parsed->cnum++;
			break;
		case 'e':
			if ((parsed->parameters & PARAM_EDGE_OF_FLIGHT_LINE)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_EDGE_OF_FLIGHT_LINE;
			parsed->cnum++;
			break;
		case 'd':
			if ((parsed->parameters & PARAM_DIRECTION_OF_SCAN_FLAG)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_DIRECTION_OF_SCAN_FLAG;
			parsed->cnum++;
			break;
		case 'R':
			if ((parsed->parameters & PARAM_RED_CHANNEL)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_RED_CHANNEL;
			parsed->cnum++;
			break;
		case 'G':
			if ((parsed->parameters & PARAM_GREEN_CHANNEL)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_GREEN_CHANNEL;
			parsed->cnum++;
			break;
		case 'B':
			if ((parsed->parameters & PARAM_BLUE_CHANNEL)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_BLUE_CHANNEL;
			parsed->cnum++;
			break;
		case 'M':
			if ((parsed->parameters & PARAM_VERTEX_INDEX)) {
				fprintf(stderr, "WARNING: Parameter %c already set. Ignoring.\n", *p);
				continue;
			}
			parsed->parameters |= PARAM_VERTEX_INDEX;
			parsed->cnum++;
			break;
		default:
			fprintf(stderr, "WARNING: Unknown parameter char: %c. Ignoring.\n", *p);
		}
	}
}

#define LIDAR_READER_VERSION "1.8.0"
#define VAULT_LIDAR_READER 1
str
LIDARattach(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	mvc *m = NULL;
	sql_trans *tr;
	sql_schema *sch;
	sql_table *lidar_fl, *lidar_tbl, *lidar_col, *tbl = NULL;
	sql_column *col;
	sql_subtype t;
	str msg = MAL_SUCCEED;
	str fname = *getArgReference_str(stk, pci, 1);
	str tname = NULL;
	str params = NULL;
	InputParameters input_params;
	oid fid, tid, cid, rid = oid_nil;
	char *tname_low = NULL, *s, bname[BUFSIZ];
	char *p;
	lidar_header *header = NULL;
	int scaleX, scaleY, scaleZ;
	int precisionX, precisionY, precisionZ;
	char *istmt=NULL, *cstmt=NULL;
	char maxval[BUFSIZ], minval[BUFSIZ], col_type[BUFSIZ];
	lng nils = 0;
	lng uniq = 0;
	lng sz = 0;
	int prm = 0, files_len = 0, idx;
	str *files = NULL;
	str filename;
	double minimumX, maximumX, minimumY, maximumY, minimumZ, maximumZ;

	switch(pci->argc) {
	case 2:
		tname = fname;
		break;
	case 3:
		tname = *getArgReference_str(stk, pci, 2);
		break;
	case 4:
		tname = *getArgReference_str(stk, pci, 2);
		params = *getArgReference_str(stk, pci, 3);
#ifndef NDEBUG
		fprintf(stderr, "Params: %s\n", params);
#endif
		break;
	default:
		msg = createException(MAL, "lidar.attach", "Wrong number of arguments");
		return msg;
	}

#ifndef NDEBUG
	if (tname != NULL) {
		fprintf(stderr, "Initial table name: %s\n", tname);
	}
#endif

	if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
		return msg;
	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
		return msg;

	/* if needed, instantiate the schema and gather all appropriate tables */
	tr = m->session->tr;
	sch = mvc_bind_schema(m, "sys");

	lidar_fl = mvc_bind_table(m, sch, "lidar_files");
	if (lidar_fl == NULL) {
		LIDARinitCatalog(m);
		lidar_fl = mvc_bind_table(m, sch, "lidar_files");
	}

	lidar_tbl = mvc_bind_table(m, sch, "lidar_tables");
	lidar_col = mvc_bind_table(m, sch, "lidar_columns");

	files = LIDARopenPath(fname, &files_len);
	if (files == NULL || files_len == 0) {
		msg = createException(MAL, "lidar.attach", "Error accessing path %s (%s)",
							  fname, LASError_GetLastErrorMsg());
		return msg;
	}

#ifndef NDEBUG
	printf("tname: %s\n", tname);
#endif
	/* extract the file name from the absolute path */
	if ((s = strrchr(tname, DIR_SEP)) == NULL)
		s = tname;
	else
		s++;
	strcpy(bname, s);
	if (s) *s = 0;

	tname_low = toLower(bname);

	/* Find the last dot in the filename and replace it with '\0' if
	 * it exists. This removes the extension part of the file, unless
	 * the extension part itself contains a dot character.
	 */
	p = strrchr(tname_low, '.');
	if (p != NULL) {
		*p = '\0';
	}

	/* Sanitize table name by substituting dot characters ('.') for underscores
	 * ('_').
	 */
	while ((p = strchr(tname_low, '.')) != NULL) {
		*p = '_';
	}

#ifndef NDEBUG
	printf("tablename: %s\n", tname_low);
#endif

	/* check table name for existence in the lidar catalog */
	col = mvc_bind_column(m, lidar_tbl, "name");
	rid = table_funcs.column_find_row(m->session->tr, col, tname_low, NULL);
	/* or as regular SQL table */
	tbl = mvc_bind_table(m, sch, tname_low);
	if (rid != oid_nil || tbl) {
		msg = createException(SQL, "lidar.attach", "Table %s already exists\n", tname_low);
		return msg;
	}

	for (idx = 0; idx < files_len; idx++) {
		filename = files[idx];
		if (header != NULL) {
			free(header);
		}
		header = LIDARopenFile(filename);
		/* See if anything went wrong */
		if (header->msg != NULL) {
			msg = header->msg;
			free(header);
			return msg;
		}

		/* check if the file is already attached */
		col = mvc_bind_column(m, lidar_fl, "name");
		rid = table_funcs.column_find_row(m->session->tr, col, filename, NULL);
		if (rid != oid_nil) {
			msg = createException(SQL, "lidar.attach", "File %s already attached\n", filename);
			return msg;
		}

		/* add row in the lidar_files catalog table */
		col = mvc_bind_column(m, lidar_fl, "id");
		fid = store_funcs.count_col(tr, col, 1) + 1;
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_fl, "id"), &fid, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_fl, "name"), filename, TYPE_str);
		/* table.id++ */
		col = mvc_bind_column(m, lidar_tbl, "id");
		tid = store_funcs.count_col(tr, col, 1) + 1;

		scaleX = (int)ceil(-log(header->hi->scaleX)/log(10));
		scaleY = (int)ceil(-log(header->hi->scaleY)/log(10));
		scaleZ = (int)ceil(-log(header->hi->scaleZ)/log(10));

		precisionX = scaleX + (int)ceil(log(header->hi->maxX)/log(10));
		precisionY = scaleY + (int)ceil(log(header->hi->maxY)/log(10));
		precisionZ = scaleZ + (int)ceil(log(header->hi->maxZ)/log(10));

#ifndef NDEBUG
		fprintf(stderr, "Scale: %f %f %f\n",
				header->hi->scaleX,
				header->hi->scaleY,
				header->hi->scaleZ);
		fprintf(stderr, "Decimal type scale: %d %d %d\n",
				(int)ceil(-log(header->hi->scaleX)/log(10)),
				(int)ceil(-log(header->hi->scaleY)/log(10)),
				(int)ceil(-log(header->hi->scaleZ)/log(10)));

		fprintf(stderr, "Decimal type precision: %d %d %d\n",
				(int)ceil(log(header->hi->maxX)/log(10)),
				(int)ceil(log(header->hi->maxY)/log(10)),
				(int)ceil(log(header->hi->maxZ)/log(10)));

		fprintf(stderr, "decimal digits: %d %d %d\n", scaleX, scaleY, scaleZ);
		fprintf(stderr, "total digits: %d %d %d\n", precisionX, precisionY, precisionZ);
#endif

		/* Parse the input parameters */
		parse_parameters(params, &input_params);

		/* store data */
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "id"), &tid, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "file_id"), &fid, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "name"), tname_low, TYPE_str);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "FileSourceId"), &header->hi->fileSourceId, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "VersionMajor"), &header->hi->versionMajor, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "VersionMinor"), &header->hi->versionMinor, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "DataFormatId"), &header->hi->dataFormatId, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "CreationDOY"), &header->hi->creationDOY, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "CreationYear"), &header->hi->creationYear, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "RecordsCount"), &header->hi->recordsCount, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "PointRecordsCount"), &header->hi->pointRecordsCount, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "DataOffset"), &header->hi->dataOffset, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "HeaderPadding"), &header->hi->headerPadding, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "Reserved"), &header->hi->reserved, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "DataRecordLength"), &header->hi->dataRecordLength, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "HeaderSize"), &header->hi->headerSize, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "ByteSize"), &header->hi->byteSize, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "BaseByteSize"), &header->hi->baseByteSize, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "WKT"), header->hi->WKT, TYPE_str);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "WKT_CompoundOK"), header->hi->WKT_CompoundOK, TYPE_str);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "Proj4"), header->hi->proj4, TYPE_str);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_tbl, "LoadParams"), &input_params.parameters, TYPE_int);


		/* store */
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "id"), &cid, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "file_id"), &fid, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "table_id"), &tid, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "ScaleX"), &header->hi->scaleX, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "ScaleY"), &header->hi->scaleY, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "ScaleZ"), &header->hi->scaleZ, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "OffsetX"), &header->hi->offsetX, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "OffsetY"), &header->hi->offsetY, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "OffsetZ"), &header->hi->offsetZ, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "MinX"), &header->hi->minX, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "MinY"), &header->hi->minY, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "MinZ"), &header->hi->minZ, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "MaxX"), &header->hi->maxX, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "MaxY"), &header->hi->maxY, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "MaxZ"), &header->hi->maxZ, TYPE_dbl);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "PrecisionX"), &precisionX, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "PrecisionY"), &precisionY, TYPE_int);
		store_funcs.append_col(m->session->tr,
							   mvc_bind_column(m, lidar_col, "PrecisionZ"), &precisionZ, TYPE_int);
		if (idx == 0) {
			minimumX = header->hi->minX;
			minimumY = header->hi->minY;
			minimumZ = header->hi->minZ;
			maximumX = header->hi->maxX;
			maximumY = header->hi->maxY;
			maximumZ = header->hi->maxZ;
		}
		else {
			if (header->hi->minX < minimumX) {
				minimumX = header->hi->minX;
			}
			if (header->hi->minY < minimumY) {
				minimumY = header->hi->minY;
			}
			if (header->hi->minZ < minimumZ) {
				minimumZ = header->hi->minZ;
			}

			if (header->hi->maxX > maximumX) {
				maximumX = header->hi->maxX;
			}
			if (header->hi->maxY > maximumY) {
				maximumY = header->hi->maxY;
			}
			if (header->hi->maxZ > maximumZ) {
				maximumZ = header->hi->maxZ;
			}
		}
		/* add a lidar_column tuple */
		col = mvc_bind_column(m, lidar_col, "id");
		cid = store_funcs.count_col(tr, col, 1) + 1;
	}

	for (idx = 0; idx < files_len; idx++) {
		free(files[idx]);
	}
	free(files);

	/* create an SQL table to hold the LIDAR table */
	tbl = mvc_create_table(m, sch, tname_low, tt_table, 0, SQL_PERSIST, 0, input_params.cnum);

	for (prm = 1; prm < PARAMS_END_SENTINEL; prm <<= 1) {
		if (input_params.parameters & prm) {
			switch(prm) {
			case PARAM_X_COORD:
				sql_find_subtype(&t, "decimal", precisionX, scaleX);
				mvc_create_column(m, tbl, "x", &t);
				break;
			case PARAM_Y_COORD:
				sql_find_subtype(&t, "decimal", precisionY, scaleY);
				mvc_create_column(m, tbl, "y", &t);
				break;
			case PARAM_Z_COORD:
				sql_find_subtype(&t, "decimal", precisionZ, scaleZ);
				mvc_create_column(m, tbl, "z", &t);
				break;
			case PARAM_GPS_TIME:
				sql_find_subtype(&t, "double", 0, 0);
				mvc_create_column(m, tbl, "gpstime", &t);
				break;
			case PARAM_SCAN_ANGLE:
				sql_find_subtype(&t, "tinyint", 0, 0);
				mvc_create_column(m, tbl, "scan_angle", &t);
				break;
			case PARAM_INTENSITY:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "intensity", &t);
				break;
			case PARAM_N_RETURNS:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "n_returns", &t);
				break;
			case PARAM_N_THIS_RETURN:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "n_this_return", &t);
				break;
			case PARAM_CLASSIFICATION_NUMBER:
				sql_find_subtype(&t, "tinyint", 0, 0);
				mvc_create_column(m, tbl, "classification_number", &t);
				break;
			case PARAM_USER_DATA:
				sql_find_subtype(&t, "tinyint", 0, 0);
				mvc_create_column(m, tbl, "user_data", &t);
				break;
			case PARAM_POINT_SOURCE_ID:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "point_source_id", &t);
				break;
			case PARAM_EDGE_OF_FLIGHT_LINE:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "edge_of_flight_line", &t);
				break;
			case PARAM_DIRECTION_OF_SCAN_FLAG:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "direction_of_scan_flag", &t);
				break;
			case PARAM_RED_CHANNEL:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "red_channel", &t);
				break;
			case PARAM_GREEN_CHANNEL:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "green_channel", &t);
				break;
			case PARAM_BLUE_CHANNEL:
				sql_find_subtype(&t, "smallint", 0, 0);
				mvc_create_column(m, tbl, "blue_channel", &t);
				break;
			case PARAM_VERTEX_INDEX:
				sql_find_subtype(&t, "int", 0, 0);
				mvc_create_column(m, tbl, "vertex_index", &t);
				break;
			}
		}
	}

	istmt = (char *) GDKzalloc(8192);
	snprintf(istmt, 8192, "INSERT INTO sys.vault_journal VALUES(%d, '%s', %d, 'lidar', '%s');", tbl->base.id, tname_low, VAULT_LIDAR_READER, LIDAR_READER_VERSION);
	msg = SQLstatementIntern(cntxt, &istmt, "LIDARattach", TRUE, FALSE, NULL);
	GDKfree(istmt);
	if (msg) {
		return msg;
	}

	cstmt = GDKzalloc(8192);
	col = mvc_bind_column(m, tbl, "x");
	if (col) {
		snprintf(minval, BUFSIZ, "%lf", minimumX);
		snprintf(maxval, BUFSIZ, "%lf", maximumX);
		snprintf(col_type, BUFSIZ, "%s(%u,%u)", col->type.type->sqlname, col->type.digits, col->type.scale);
		snprintf(cstmt, 8192, "insert into sys.statistics values(%d,'%s',%d,now()," LLFMT "," LLFMT "," LLFMT "," LLFMT ",'%s','%s',%s,%s);", col->base.id, col_type, precisionX, sz, sz, uniq, nils, minval, maxval, "false", "false");
		msg = SQLstatementIntern(cntxt, &cstmt, "LIDARattach", TRUE, FALSE, NULL);
		if (msg) {
			GDKfree(cstmt);
			return msg;
		}
	}

	col = mvc_bind_column(m, tbl, "y");
	if (col) {
		snprintf(minval, BUFSIZ, "%lf", minimumY);
		snprintf(maxval, BUFSIZ, "%lf", maximumY);
		snprintf(col_type, BUFSIZ, "%s(%u,%u)", col->type.type->sqlname, col->type.digits, col->type.scale);
		snprintf(cstmt, 8192, "insert into sys.statistics values(%d,'%s',%d,now()," LLFMT "," LLFMT "," LLFMT "," LLFMT ",'%s','%s',%s,%s);", col->base.id, col_type, precisionY, sz, sz, uniq, nils, minval, maxval, "false", "false");
		msg = SQLstatementIntern(cntxt, &cstmt, "LIDARattach", TRUE, FALSE, NULL);
		if (msg) {
			GDKfree(cstmt);
			return msg;
		}
	}
	col = mvc_bind_column(m, tbl, "z");
	if (col) {
		snprintf(minval, BUFSIZ, "%lf", minimumZ);
		snprintf(maxval, BUFSIZ, "%lf", maximumZ);
		snprintf(col_type, BUFSIZ, "%s(%u,%u)", col->type.type->sqlname, col->type.digits, col->type.scale);
		snprintf(cstmt, 8192, "insert into sys.statistics values(%d,'%s',%d,now()," LLFMT "," LLFMT "," LLFMT "," LLFMT ",'%s','%s',%s,%s);", col->base.id, col_type, precisionZ, sz, sz, uniq, nils, minval, maxval, "false", "false");
		msg = SQLstatementIntern(cntxt, &cstmt, "LIDARattach", TRUE, FALSE, NULL);
		if (msg) {
			GDKfree(cstmt);
			return msg;
		}
	}
	GDKfree(cstmt);

	free(header->hi);
	free(header);

	return MAL_SUCCEED;
}

typedef union retval {
	bte val_bte;
	sht val_sht;
	int val_int;
	lng val_lng;
#ifdef HAVE_HGE
	hge val_hge;
#endif
	dbl val_dbl;
} RetVal;

static RetVal readValue(LASPointH p, ParameterValues param) {
	RetVal ret;
	LASColorH color;

	switch(param) {
	case PARAM_X_COORD:
		ret.val_dbl = LASPoint_GetX(p);
		break;
	case PARAM_Y_COORD:
		ret.val_dbl = LASPoint_GetY(p);
		break;
	case PARAM_Z_COORD:
		ret.val_dbl = LASPoint_GetZ(p);
		break;
	case PARAM_GPS_TIME:
		ret.val_dbl = LASPoint_GetTime(p);
		break;
	case PARAM_SCAN_ANGLE:
		ret.val_bte = LASPoint_GetScanAngleRank(p);
		break;
	case PARAM_INTENSITY:
		ret.val_sht = LASPoint_GetIntensity(p);
		break;
	case PARAM_N_RETURNS:
		ret.val_sht = LASPoint_GetNumberOfReturns(p);
		break;
	case PARAM_N_THIS_RETURN:
		ret.val_sht = LASPoint_GetReturnNumber(p);
		break;
	case PARAM_CLASSIFICATION_NUMBER:
		ret.val_bte = LASPoint_GetClassification(p);
		break;
	case PARAM_USER_DATA:
		ret.val_bte = LASPoint_GetUserData(p);
		break;
	case PARAM_POINT_SOURCE_ID:
		ret.val_sht = LASPoint_GetPointSourceId(p);
		break;
	case PARAM_EDGE_OF_FLIGHT_LINE:
		ret.val_sht = LASPoint_GetFlightLineEdge(p);
		break;
	case PARAM_DIRECTION_OF_SCAN_FLAG:
		ret.val_sht = LASPoint_GetScanDirection(p);
		break;
	case PARAM_RED_CHANNEL:
		color = LASPoint_GetColor(p);
		ret.val_sht = LASColor_GetRed(color);
		break;
	case PARAM_GREEN_CHANNEL:
		color = LASPoint_GetColor(p);
		ret.val_sht = LASColor_GetRed(color);
		break;
	case PARAM_BLUE_CHANNEL:
		color = LASPoint_GetColor(p);
		ret.val_sht = LASColor_GetRed(color);
		break;
	default:
		fprintf(stderr, "readValue: Unimplemented\n");
	}

	return ret;
}

#define READ_ARRAY(BAT_TYPE)										\
static BAT *														\
read_array_##BAT_TYPE(str fname,									\
					  ParameterValues val, long rows,				\
					  double scale, int *error_code)				\
{																	\
	BAT *b;														\
	BAT_TYPE *d = NULL;											\
	LASPointH p = NULL;											\
	LASReaderH reader = NULL;										\
	int i;															\
	RetVal value;													\
																	\
	b = COLnew(0, TYPE_##BAT_TYPE, rows, PERSISTENT);				\
																	\
	if (b == NULL) {												\
		*error_code = 1;											\
		goto cleanup##BAT_TYPE;									\
	}																\
																	\
	MT_lock_set(&mt_lidar_lock);									\
	LASError_Reset();												\
	reader = LASReader_Create(fname);								\
	MT_lock_unset(&mt_lidar_lock);									\
	if (LASError_GetErrorCount() != 0) {							\
		*error_code = 2;											\
		return NULL;												\
	}																\
																	\
	d = (BAT_TYPE *) Tloc(b, 0);									\
																	\
	p = LASReader_GetNextPoint(reader);							\
	i = 0;															\
	while(p) {														\
		switch(val) {												\
		case PARAM_X_COORD:										\
		case PARAM_Y_COORD:										\
		case PARAM_Z_COORD:										\
			value = readValue(p, val);								\
			d[i] = value.val_dbl/scale;							\
			break;													\
		case PARAM_GPS_TIME:										\
			value = readValue(p, val);								\
			d[i] = value.val_dbl;									\
			break;													\
		case PARAM_SCAN_ANGLE:										\
		case PARAM_CLASSIFICATION_NUMBER:							\
		case PARAM_USER_DATA:										\
			value = readValue(p, val);								\
			d[i] = value.val_bte;									\
			break;													\
		case PARAM_INTENSITY:										\
		case PARAM_N_RETURNS:										\
		case PARAM_N_THIS_RETURN:									\
		case PARAM_POINT_SOURCE_ID:								\
		case PARAM_EDGE_OF_FLIGHT_LINE:							\
		case PARAM_DIRECTION_OF_SCAN_FLAG:							\
		case PARAM_RED_CHANNEL:									\
		case PARAM_GREEN_CHANNEL:									\
		case PARAM_BLUE_CHANNEL:									\
			value = readValue(p, val);								\
			d[i] = value.val_sht;									\
			break;													\
		case PARAM_VERTEX_INDEX:									\
			d[i] = i;												\
			break;													\
		default:													\
			fprintf(stderr,										\
					"read_array_##BAT_TYPE: Unimplemented\n");		\
		}															\
		p = LASReader_GetNextPoint(reader);						\
		i++;														\
	}																\
cleanup##BAT_TYPE:													\
	MT_lock_set(&mt_lidar_lock);									\
	if (p != NULL) LASPoint_Destroy(p);							\
	if (reader != NULL) LASReader_Destroy(reader);					\
	MT_lock_unset(&mt_lidar_lock);									\
																	\
	return b;														\
}


/* Define functions for reading into different typed columns */
READ_ARRAY(bte)

READ_ARRAY(sht)

READ_ARRAY(int)

READ_ARRAY(lng)

#ifdef HAVE_HGE
READ_ARRAY(hge)
#endif

READ_ARRAY(dbl)

static str
LIDARloadTable_(mvc *m, sql_schema *sch, sql_table *lidar_tbl, str tname, sql_table *tbl, oid rid)
{
	sql_table *lidar_fl, *lidar_cl;
	sql_column *col; /*, *colx, *coly, *colz;*/
	str fname;
	str msg = MAL_SUCCEED;
	oid frid = oid_nil, tid = oid_nil;
	int fid, prm = 0;
#ifndef NDEBUG
	int time0;
#endif
	int *tpcode = NULL;
	long *rep = NULL, *wid = NULL, rows = 0;
	/* BAT *x = NULL, *y = NULL, *z = NULL; */
	BAT *bat = NULL;
	sql_column *column;
	int precisionx = 0, precisiony = 0, precisionz = 0;
	int input_params;
	double scalex, scaley, scalez;
	int error_code;
	
	(void) tname;
	/* col = mvc_bind_column(m, lidar_tbl, "name"); */
	/* rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL); */
	/* if (rid == oid_nil) { */
	/* 	msg = createException(MAL, "lidar.loadtable", "Table %s is unknown to the LIDAR catalog. Attach first the containing file\n", tname); */
	/* 	return msg; */
	/* } */

	/* Open LIDAR file */
	col = mvc_bind_column(m, lidar_tbl, "file_id");
	fid = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

	lidar_fl = mvc_bind_table(m, sch, "lidar_files");
	col = mvc_bind_column(m, lidar_fl, "id");
	frid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
	col = mvc_bind_column(m, lidar_fl, "name");
	fname = (char *)table_funcs.column_find_value(m->session->tr, col, frid);

	lidar_cl = mvc_bind_table(m, sch, "lidar_columns");
	if (lidar_cl == NULL) {
		msg = createException(MAL, "lidar.loadtable", "Could not find table lidar_columns.\n");
		return msg;
	}

	/* data load */
	col = mvc_bind_column(m, lidar_tbl, "PointRecordsCount");
	rows = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);
#ifndef NDEBUG
	fprintf(stderr,"#Loading %ld rows in table %s\n", rows, tname);
	time0 = GDKms();
#endif
	/* colx = mvc_bind_column(m, tbl, "x"); */
	/* coly = mvc_bind_column(m, tbl, "y"); */
	/* colz = mvc_bind_column(m, tbl, "z"); */
	col = mvc_bind_column(m, lidar_tbl, "LoadParams");
	input_params = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

	for (prm = 1; prm < PARAMS_END_SENTINEL; prm <<= 1) {
		if (input_params & prm) {
			switch(prm) {
			case PARAM_X_COORD:
				col = mvc_bind_column(m, lidar_cl, "file_id");
				tid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
				col = mvc_bind_column(m, lidar_cl, "ScaleX");
				scalex = *(double*)table_funcs.column_find_value(m->session->tr, col, tid);
				col = mvc_bind_column(m, lidar_cl, "PrecisionX");
				precisionx = *(sht*)table_funcs.column_find_value(m->session->tr, col, tid);
				if (precisionx <= 2)
					bat = read_array_bte(fname, prm, rows, scalex, &error_code);
				else if (precisionx <= 4)
					bat = read_array_sht(fname, prm, rows, scalex, &error_code);
				else if (precisionx <= 8)
					bat = read_array_int(fname, prm, rows, scalex, &error_code);
				else if (precisionx <= 16)
					bat = read_array_int(fname, prm, rows, scalex, &error_code);
				else if (precisionx <= 32)
					bat = read_array_lng(fname, prm, rows, scalex, &error_code);
#ifdef HAVE_HGE
				else if (precisionx <= 64)
					bat = read_array_hge(fname, prm, rows, scalex, &error_code);
#endif
				else {
					bat = NULL;
					error_code = 3;
				}
				column = mvc_bind_column(m, tbl, "x");
				break;
			case PARAM_Y_COORD:
				col = mvc_bind_column(m, lidar_cl, "file_id");
				tid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
				col = mvc_bind_column(m, lidar_cl, "ScaleY");
				scaley = *(double*)table_funcs.column_find_value(m->session->tr, col, tid);
				col = mvc_bind_column(m, lidar_cl, "PrecisionY");
				precisiony = *(sht*)table_funcs.column_find_value(m->session->tr, col, tid);
				if (precisiony <= 2)
					bat = read_array_bte(fname, prm, rows, scaley, &error_code);
				else if (precisiony <= 4)
					bat = read_array_sht(fname, prm, rows, scaley, &error_code);
				else if (precisiony <= 8)
					bat = read_array_int(fname, prm, rows, scaley, &error_code);
				else if (precisiony <= 16)
					bat = read_array_int(fname, prm, rows, scaley, &error_code);
				else if (precisiony <= 32)
					bat = read_array_lng(fname, prm, rows, scaley, &error_code);
#ifdef HAVE_HGE
				else if (precisiony <= 64)
					bat = read_array_hge(fname, prm, rows, scaley, &error_code);
#endif
				else {
					bat = NULL;
					error_code = 4;
				}
				column = mvc_bind_column(m, tbl, "y");
				break;
			case PARAM_Z_COORD:
				col = mvc_bind_column(m, lidar_cl, "file_id");
				tid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
				col = mvc_bind_column(m, lidar_cl, "ScaleZ");
				scalez = *(double*)table_funcs.column_find_value(m->session->tr, col, tid);
				col = mvc_bind_column(m, lidar_cl, "PrecisionZ");
				precisionz = *(sht*)table_funcs.column_find_value(m->session->tr, col, tid);
				if (precisionz <= 2)
					bat = read_array_bte(fname, prm, rows, scalez, &error_code);
				else if (precisionz <= 4)
					bat = read_array_sht(fname, prm, rows, scalez, &error_code);
				else if (precisionz <= 8)
					bat = read_array_int(fname, prm, rows, scalez, &error_code);
				else if (precisionz <= 16)
					bat = read_array_int(fname, prm, rows, scalez, &error_code);
				else if (precisionz <= 32)
					bat = read_array_lng(fname, prm, rows, scalez, &error_code);
#ifdef HAVE_HGE
				else if (precisionz <= 64)
					bat = read_array_hge(fname, prm, rows, scalez, &error_code);
#endif
				else {
					bat = NULL;
					error_code = 5;
				}
				column = mvc_bind_column(m, tbl, "z");
				break;
			case PARAM_GPS_TIME:
				bat = read_array_dbl(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "gpstime");
				break;
			case PARAM_SCAN_ANGLE:
				bat = read_array_bte(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "scan_angle");
				break;
			case PARAM_INTENSITY:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "intensity");
				break;
			case PARAM_N_RETURNS:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "n_returns");
				break;
			case PARAM_N_THIS_RETURN:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "n_this_return");
				break;
			case PARAM_CLASSIFICATION_NUMBER:
				bat = read_array_bte(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "classification_number");
				break;
			case PARAM_USER_DATA:
				bat = read_array_bte(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "user_data");
				break;
			case PARAM_POINT_SOURCE_ID:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "point_source_id");
				break;
			case PARAM_EDGE_OF_FLIGHT_LINE:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "edge_of_flight_line");
				break;
			case PARAM_DIRECTION_OF_SCAN_FLAG:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "direction_of_scan_flag");
				break;
			case PARAM_RED_CHANNEL:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "red_channel");
				break;
			case PARAM_GREEN_CHANNEL:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "green_channel");
				break;
			case PARAM_BLUE_CHANNEL:
				bat = read_array_sht(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "blue_channel");
				break;
			case PARAM_VERTEX_INDEX:
				bat = read_array_int(fname, prm, rows, 0, &error_code);
				column = mvc_bind_column(m, tbl, "vertex_index");
				break;
			default:
				fprintf(stderr, "LIDARloadTable_: unimplemented 0x%x\n", prm);
				abort();
			}
			if ( bat == NULL ) {
				GDKfree(tpcode);
				GDKfree(rep);
				GDKfree(wid);
				switch (error_code) {
				case 1:
					msg = createException(MAL, "lidar.lidarload", "Malloc failed");
					break;
				case 2:
					msg = createException(MAL, "lidar.lidarload",
										  "Error accessing LIDAR file %s (%s)",
										  fname, LASError_GetLastErrorMsg());
					break;
				case 3:
					msg = createException(MAL, "lidar.lidarload",
										  "Unknown precision for X column (%d)",
										  precisionx);
					break;
				case 4:
					msg = createException(MAL, "lidar.lidarload",
										  "Unknown precision for Y column (%d)",
										  precisiony);
					break;
				case 5:
					msg = createException(MAL, "lidar.lidarload",
										  "Unknown precision for Z column (%d)",
										  precisionz);
					break;
				}
				return msg;
			}
			BATsetcount(bat, rows);
			bat->tsorted = 0;
			bat->trevsorted = 0;
			BATmode(bat, PERSISTENT);
			store_funcs.append_col(m->session->tr, column, bat, TYPE_bat);
			BBPdecref(bat->batCacheid, TRUE);
			BBPunfix(bat->batCacheid);
		}
	}


#ifndef NDEBUG
	fprintf(stderr,"#File loaded in %d ms\t", GDKms() - time0);
#endif
	GDKfree(tpcode);
	GDKfree(rep);
	GDKfree(wid);

	return msg;
}

str LIDARloadTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    mvc *m = NULL;
    sql_table *lidar_tbl, *tbl;
	sql_schema *sch;
	str tname = toLower(*getArgReference_str(stk, pci, 1));
    sql_column *col;
	str msg = MAL_SUCCEED;
	size_t sz;
	char *filenames_query = GDKmalloc(BUFSIZ);
	res_table *fres = NULL;

	if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
		return msg;
	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
		return msg;

	sch = mvc_bind_schema(m, "sys");

	lidar_tbl = mvc_bind_table(m, sch, "lidar_tables");
	if (lidar_tbl == NULL) {
		msg = createException(MAL, "lidar.loadtable", "LIDAR catalog is missing.\n");
		return msg;
	}

    tbl = mvc_bind_table(m, sch, tname);
	if (tbl == NULL) {
		msg = createException(MAL, "lidar.loadtable", "Could not find table %s.\n", tname);
		return msg;
	}

	col = mvc_bind_column(m, tbl, "x");
	sz = store_funcs.count_col(m->session->tr, col, 1);
	if (sz != 0) {
		msg = createException(MAL, "lidar.loadtable", "Table %s is not empty.\n", tname);
		return msg;
	}

	snprintf(filenames_query, BUFSIZ, "SELECT id FROM lidar_tables WHERE name='%s';", tname);
	msg = SQLstatementIntern(cntxt, &filenames_query, "rows", 1, 0, &fres);
	if (msg != MAL_SUCCEED) {
		return msg;
	}
	GDKfree(filenames_query);

	if (fres) {
		BATiter fs_rid = bat_iterator(BATdescriptor(fres->cols[0].b));
		oid id = 0, cnt = BATcount(fs_rid.b);
#ifndef NDEBUG
		printf("count: %ld\n", cnt);
#endif
		for (id = 0; id < cnt; id++) {
			int rid = *(int*)BUNtail(fs_rid, id);
#ifndef NDEBUG
			printf("id, rid: %ld %d\n", id, rid);
#endif
			msg = LIDARloadTable_(m, sch, lidar_tbl, tname, tbl, rid - 1);
			if (msg != MAL_SUCCEED) {
				return msg;
			}
		}
	}

	return msg;
}

str
LIDARprelude(void *ret) {
	(void) ret;
	MT_lock_init(&mt_lidar_lock, "lidar.lock");

	return MAL_SUCCEED;
}

str
LIDARCheckTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    str msg = MAL_SUCCEED;
  	mvc *m = NULL;
  	size_t sz;
  	sql_schema *sch = NULL;
  	sql_table *lidar_tbl, *tbl = NULL;
  	sql_column *col;
  	oid rid = oid_nil;
  	str tname = *getArgReference_str(stk, pci, 3);
  	int *res = getArgReference_int(stk, pci, 1);

  	if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    	return msg;
  	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    	return msg;

  	sch = mvc_bind_schema(m, "sys");
  	lidar_tbl = mvc_bind_table(m, sch, "lidar_tables");
  	if (lidar_tbl == NULL) {
    	msg = createException(MAL, "lidar.check", "LIDAR catalog is missing.\n");
    	return msg;
  	}

  	/*Check if is a table which belongs to lidar_tables*/
  	col = mvc_bind_column(m, lidar_tbl, "name");
  	rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
  	if (rid == oid_nil) {
    	return MAL_SUCCEED;
  	}

  	tbl = mvc_bind_table(m, sch, tname);
  	if (tbl == NULL) {
    	msg = createException(MAL, "lidar.check", "Could not find table %s.\n", tname);
    	return msg;
  	}

  	col = mvc_bind_column(m, tbl, "x");
  	sz = store_funcs.count_col(m->session->tr, col, 1);

  	if (sz == 0) {
    	/*Lets load the table*/
		char *filenames_query = GDKmalloc(BUFSIZ);
		res_table *fres = NULL;

		snprintf(filenames_query, BUFSIZ, "SELECT id FROM lidar_tables WHERE name='%s';", tname);
		msg = SQLstatementIntern(cntxt, &filenames_query, "rows", 1, 0, &fres);
		if (msg != MAL_SUCCEED) {
			return msg;
		}
		GDKfree(filenames_query);

		if (fres) {
			BATiter fs_rid = bat_iterator(BATdescriptor(fres->cols[0].b));
			oid id = 0, cnt = BATcount(fs_rid.b);
#ifndef NDEBUG
			printf("count: %ld\n", cnt);
#endif
			for (id = 0; id < cnt; id++) {
				int rid = *(int*)BUNtail(fs_rid, id);
#ifndef NDEBUG
				printf("id, rid: %ld %d\n", id, rid);
#endif
				msg = LIDARloadTable_(m, sch, lidar_tbl, tname, tbl, rid - 1);
				if (msg != MAL_SUCCEED) {
					return msg;
				}
			}
		}
    	*res = VAULT_TABLE_LOADED;
  	} else {
    	if (tbl->access == TABLE_WRITABLE)
      		*res = VAULT_TABLE_ANALYZE;
    	else
      		*res = VAULT_TABLE_DONE;

#ifndef NDEBUG
    	fprintf(stderr, "The table %s is already loaded and its status is %d!!!\n", tname, *res);
#endif
  	}
  	return msg;
}

str
LIDARAnalyzeTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    mvc *m = NULL;
    str msg = MAL_SUCCEED;
    char readonly[BUFSIZ], analyze[BUFSIZ];
    char *sr, *sa;
    sql_schema *sch = NULL;
    sql_table *lidar_tbl;
    sql_column *col;
    oid rid = oid_nil;
    int status, *res;
    str tname = NULL;
    
    if (pci->argc == 4) {
    	status = *getArgReference_int(stk, pci, 2);
    	tname = *getArgReference_str(stk, pci, 3);
    	res = getArgReference_int(stk, pci, 0);
    } else {
        msg = createException(MAL, "lidar.analyze", "incorrect number of arguments.\n");
        return msg;
    }

    if (status != VAULT_TABLE_ANALYZE) {
        *res = status;
        return msg;
    }

    if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
        return msg;
    if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
        return msg;

    sch = mvc_bind_schema(m, "sys");
    lidar_tbl = mvc_bind_table(m, sch, "lidar_tables");
    if (lidar_tbl == NULL) {
        msg = createException(MAL, "lidar.analyze", "LIDAR catalog is missing.\n");
        return msg;
    }

    /*Check if it is a table which belongs to lidar_tables*/
    col = mvc_bind_column(m, lidar_tbl, "name");
    rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
    if (rid == oid_nil) {
        msg = createException(MAL, "lidar.analyze", "Table %s is unknown to the LIDAR catalog. Attach first the containing file\n", tname);
        return msg;
    }

    /*Set table read only*/
    sr = readonly;
    snprintf(readonly, BUFSIZ, "alter table %s set read only;", tname);

#ifndef NDEBUG
    fprintf(stderr, "The readonly stmt is: %s!!!\n", readonly);
#endif
    LIDAR_LOCK;
    msg = SQLstatementIntern(cntxt, &sr, "lidar.analyze", TRUE, FALSE, NULL);
    LIDAR_UNLOCK;
    if (msg)
        return msg;

    /*Analyze table*/
    sa = analyze;
    snprintf(analyze, BUFSIZ, "analyze sys.%s (id, posX, posY, posZ) minmax;", tname);

#ifndef NDEBUG
    fprintf(stderr, "The analyze stmt is: %s!!!\n", analyze);
#endif
    LIDAR_LOCK;
    msg = SQLstatementIntern(cntxt, &sa, "lidar.analyze", TRUE, FALSE, NULL);
    LIDAR_UNLOCK;
    if (msg)
        return msg;

    *res = VAULT_TABLE_DONE;

    return MAL_SUCCEED;
}
