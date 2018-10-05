/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

/*
 * @f vault
 * @a Martin Kersten
 * @v 0.1
 * @+ Data Vaults
 * The Data Vault module provides the basic infrastructure to manage
 * a repository with datafiles whose integrity and use is shared between
 * MonetDB and the repository owner.
 *
 * Once a vault is created, the system administrator or crontab job adds files to the vault catalog.
 * The source attribute provides the universal resource identifier (URI)
 * in a format understood by the CURL library. In most cases, it represents a file
 * to be accessed using FTP.
 *
 * A target denotes its name in the staging area, i.e. a local cache where copies are to be stored.
 * The local cache can be hierarchical structured to spread the load over multiple volumns
 * and to retain the structure of the source repository.
 * Files are dropped from the local cache using a SQL vacuum() operation based on a LRU time stamp.
 * The retention period depends on the amount of disk resources available.
 * The vaultVacuum() operation should be automatically triggered when disk space becomes a scarce resource.
 *
 * An vaultImport() operation copies a file from the remote repository into the staging area.
 *
 * The vaultBasename() operation extract the tail of the argument. It can be used to derive
 * target filename locations.
 *
 * If source and target files reside on the same file system then a symbolic link is sufficient
 * and vaultVacuum() need not be triggered.
 *
 * The file mapping catalog is kept lean. The attribute 'created' marks the actual time
 * the file was obtained from the remote source. The lru attribute is set each time we access its content.
 * Files that are bound to internal database structures may want to set it into the future.
 * @verbatim
 * CREATE SEQUENCE sys.vaultid AS int;
 *
 * CREATE TABLE sys.vault (
 * vid 			int PRIMARY KEY,-- Internal key
 * kind			string,			-- vault kind (CSV, FITS,..)
 * source			string,			-- remote file name for cURL to access
 * target			string,			-- file name of source file in vault
 * created			timestamp,		-- timestamp upon entering the cache
 * lru				timestamp		-- least recently used
 * );
 *
 * create function vaultLocation()
 * returns string
 * external name vault.getdirectory;
 *
 * create function vaultSetLocation(dir string)
 * returns string
 * external name vault.setdirectory;
 *
 * create function vaultBasename(fnme string, split string)
 * returns string
 * external name vault.basename;
 *
 * create function vaultImport(source string, target string)
 * returns timestamp
 * external name vault.import;
 *
 * create function vaultRemove(target string)
 * returns timestamp
 * external name vault.remove;
 *
 * create procedure vaultVacuum( t timestamp)
 * begin
 * update vault
 *   set created= remove(target),
 *   lru = null
 *   where  created < t;
 * end;
 * @end verbatim
 *
 * The module is developed solely for a Linux environment.
 * The vault root is a subdirectory of the dbpath/vault/ and contains
 * a subdirectory for each vault kind. In turn, each vault kind comes
 * with a refinement of the catalog identified above using the vid to relate the two.
 *
 * For large staging pools it may be advisable to pre-create the repository
 * structure, e.g. mounting multiple volumns for its partitions.
 *
 * The session structure would be something like:
 * @begin verbatim
 * insert into vault(vid,kind,source) values(0,'dummy','ftp://ftp.rep.edu/repos/station-1'),
 * 	(1,'dummy','ftp://ftp.rep.edu/repos/station-2');
 * update vault
 *   set target = basename(source,'repos');
 * update vault
 *   set created= import(source,target)
 *   where created is null;
 * select * from vault limit 2;
 * call vacuum(now());
 * @end
 */
/*
 * Module initializaton
 */
#include "monetdb_config.h"
#include "vault.h"
#include "mal_client.h"
#include "mal_interpreter.h"
#include <sql_mvc.h>
#include <sql_scenario.h>
#include <sql.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

/*New Readers*/
#include "lidar.h"
#include "gadget.h"

char vaultpath[BUFSIZ];
/*
 * The curl sample code has been copied from http://curl.haxx.se/libcurl/c/import.html
 */
#ifdef HAVE_CURL
struct FtpFile {
  const char *filename;
  FILE *stream;
};

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
	struct FtpFile *out=(struct FtpFile *)stream;

	if (!out)
		return -1;
	if (!out->stream) {
		/* open file for writing */
		out->stream=fopen(out->filename, "wb");
		if (!out->stream)
			return -1; /* failure, can't open file to write */
	}
	return fwrite(buffer, size, nmemb, out->stream);
}
#endif

str
VLTimport(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	timestamp *ret = getArgReference_TYPE(stk,pci,0,timestamp);
	str *source = getArgReference_str(stk,pci,1);
	str *target = getArgReference_str(stk,pci,2);
	str msg = MAL_SUCCEED;

#ifdef HAVE_CURL
	CURL *curl;
	CURLcode res;
	char path[BUFSIZ];
	struct FtpFile ftpfile={
		path, /* name to store the file as if succesful */
		NULL
	};

	/*curl_global_init(CURL_GLOBAL_DEFAULT);*/

	snprintf(path,BUFSIZ,"%s%c%s", vaultpath, DIR_SEP, *target);
	/*mnstr_printf(GDKout,"#vault.import: %s\n",path);*/
	if (strcmp(path, *source) == 0) 
		return MTIMEcurrent_timestamp(ret);
	/* create the subdir */
	GDKcreatedir(path);
	curl = curl_easy_init();
	if(curl) {
		/* get a copy */
		curl_easy_setopt(curl, CURLOPT_URL, *source);
		/*
		 * Actually, before copying the file it is better to check its
		 * properties, such as last modified date to see if it needs a refresh.
		 * Have to find the CURL method to enact this. It may be protocol
		 * dependent.
		 */
		if (access(path, R_OK) == 0){
			/* TODO */
		}

		/* Define our callback to get called when there's data to be written */
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
		/* Set a pointer to our struct to pass to the callback */
		/* coverity[bad_sizeof] */
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ftpfile);

		/* Switch on full protocol/debug output */
		IODEBUG curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

		res = curl_easy_perform(curl);

		/* always cleanup */
		curl_easy_cleanup(curl);

		if(CURLE_OK != res)
			msg = createException(MAL,"vault.import", SQLSTATE(42000) "curl [%u] %s '%s' -> '%s'\n", res, curl_easy_strerror(res), *source,path);
	}

	if(ftpfile.stream)
		fclose(ftpfile.stream); /* close the local file */

	curl_global_cleanup();
#else
	(void) source;
	(void) target;
	msg = createException(MAL,"vault.import", SQLSTATE(42000) "No curl library");
#endif
	if (msg)
		return msg;
	(void) mb;
	(void) cntxt;
	return MTIMEcurrent_timestamp(ret);
}


str
VLTprelude(void *ret)
{
	(void) ret;
#ifdef HAVE_CURL
	if (vaultpath[0] == 0){
		curl_global_init(CURL_GLOBAL_DEFAULT);
	}
#endif
	if ( vaultpath[0] == 0){
		snprintf(vaultpath, FILENAME_MAX, "%s%cvault", GDKgetenv("gdk_dbpath"), DIR_SEP);
		if (mkdir(vaultpath, 0755) < 0 && errno != EEXIST)
			return createException(MAL,"vault.getLocation", SQLSTATE(42000) "can not access vault directory");
	}
	return MAL_SUCCEED;
}

str
VLTbasename(str *ret, str *fname, str *split)
{
	str r;
	r= strstr(*fname, *split);

	if ( r ){
		*ret = GDKstrdup( r);
		return MAL_SUCCEED;
	}
	throw(MAL,"vault.basename",SQLSTATE(42000) "Split of file failed:%s",*fname);
}

str VLTremove(timestamp *ret, str *t)
{
	(void) remove(*t);
	*ret = *timestamp_nil;
	return MAL_SUCCEED;
}

str
VLTepilogue(void *ret)
{
	(void)ret;
	return MAL_SUCCEED;
}

str
VLTsetLocation(str *ret, str *src){
	strncpy(vaultpath,*src,FILENAME_MAX);
	*ret= GDKstrdup(vaultpath);
	return MAL_SUCCEED;
}

str
VLTgetLocation(str *ret){
	*ret= GDKstrdup(vaultpath);
	return MAL_SUCCEED;
}

str
VLTcheckTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str msg = MAL_SUCCEED;
  	mvc *m = NULL;
  	sql_schema *sch = NULL;
  	sql_table *vaults_tbl, *tbl = NULL;
  	sql_column *col;
  	oid rid = oid_nil;
  	int reader_id = 0, *res = NULL;
  	str tname = NULL;
  
  	if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    	return msg;
  	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    	return msg;

  	if (pci->argc == 4) {	
	  	tname = *getArgReference_str(stk, pci, 3);
  		printf("Table name%s\n", tname);
  		res = getArgReference_int(stk, pci, 0);
  	} else if (pci->argc == 3) {
	  	tname = *getArgReference_str(stk, pci, 2);
  		printf("Table name%s\n", tname);
  		res = getArgReference_int(stk, pci, 0);
  	} else {
    	msg = createException(MAL, "vaults.check", "Incorrect number of arguments %d.\n", pci->argc);
    	return msg;
  	}

  	if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    	return msg;
  	if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    	return msg;

  	sch = mvc_bind_schema(m, "sys");
  	vaults_tbl = mvc_bind_table(m, sch, "vault_journal");
  	if (vaults_tbl == NULL) {
    	msg = createException(MAL, "vaults.check", "VAULTS catalog is missing.\n");
    	return msg;
  	}

  	/*Check if it is a table which belongs to vaults_tables*/
  	col = mvc_bind_column(m, vaults_tbl, "table_name");
  	rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
  	if (rid == oid_nil) {
    	return MAL_SUCCEED;
  	}

  	tbl = mvc_bind_table(m, sch, tname);
  	if (tbl == NULL) {
    	msg = createException(MAL, "vaults.check", "Could not find table %s.\n", tname);
    	return msg;
  	}

    /*Get reader id*/
  	col = mvc_bind_column(m, vaults_tbl, "vault_reader_id");
  	reader_id = *(int*) table_funcs.column_find_value(m->session->tr, col,  (oid) rid);

    /*
     * Check which reader.
     * The info should be preserved in catalog to be more generic
     */
    switch (reader_id) {
    	case VAULT_LIDAR_READER:
			msg = LIDARCheckTable(cntxt, mb, stk, pci);
    		break;
    	case VAULT_GADGET_READER:
			msg = gadgetCheckTable(cntxt, mb, stk, pci);
    		break;
    	default:
    		msg = createException(MAL, "vaults.check", "Reader does not exist %d.\n", reader_id);
    }
    *res = reader_id;

  	return msg;
}

str
VLTanalyzeTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    mvc *m = NULL;
    str msg = MAL_SUCCEED;
    sql_schema *sch = NULL;
    sql_table *vaults_tbl;
    sql_column *col;
    oid rid = oid_nil;
    int reader_id = *getArgReference_int(stk, pci, 1);
    int status = *getArgReference_int(stk, pci, 2);
    str tname = *getArgReference_str(stk, pci, 3);
    int *res = getArgReference_int(stk, pci, 0);

    if (status != VAULT_TABLE_ANALYZE) {
        *res = status;
        return msg;
    }

    if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
        return msg;
    if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
        return msg;

    sch = mvc_bind_schema(m, "sys");
    vaults_tbl = mvc_bind_table(m, sch, "vault_journal");
    if (vaults_tbl == NULL) {
        msg = createException(MAL, "vaults.analyze", "VAULTS catalog is missing.\n");
        return msg;
    }

    /*Check if it is a table which belongs to vaults_tables*/
    col = mvc_bind_column(m, vaults_tbl, "table_name");
    rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
    if (rid == oid_nil) {
        msg = createException(MAL, "vaults.analyze", "Table %s is unknown to the VAULTS catalog. Attach first the containing file\n", tname);
        return msg;
    }

    /*Get reader id*/
  	col = mvc_bind_column(m, vaults_tbl, "vault_reader_id");
  	reader_id = *(int*) table_funcs.column_find_value(m->session->tr, col,  (oid) rid);

    /*
     * Check which reader.
     * The info should be preserved in catalog to be more generic
     */
    switch (reader_id) {
    	case VAULT_LIDAR_READER:
			msg = LIDARAnalyzeTable(cntxt, mb, stk, pci);
    		break;
    	case VAULT_GADGET_READER:
			msg = gadgetAnalyzeTable(cntxt, mb, stk, pci);
    		break;
    	default:
    		msg = createException(MAL, "vaults.analyze", "Reader does not exist %d.\n", reader_id);
    }

    /*TODO: such value should be set by the reader, not by vaults*/
    *res = VAULT_TABLE_DONE;

    return MAL_SUCCEED;
}

/* str mvc_bind_wrap(int *bid, str *sname, str *tname, str *cname, int *access); */
str
mvc_VLT_bind_wrap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  int upd = (pci->argc == 8 || pci->argc == 10);
  BAT *b = NULL, *bn;
  bat *bid = getArgReference_bat(stk, pci, 0);
  int coltype = getBatType(getArgType(mb, pci, 0));
  mvc *m = NULL;
  str msg;
  int status = *getArgReference_int(stk, pci, 1 + upd);
  str *sname = getArgReference_str(stk, pci, 3 + upd);
  str *tname = getArgReference_str(stk, pci, 4 + upd);
  str *cname = getArgReference_str(stk, pci, 5 + upd);
  int *access = getArgReference_int(stk, pci, 6 + upd);

  if (!*access && status == VAULT_TABLE_LOADED)
    *access = RD_INS;

  printf (" Level 0\n" );
  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != NULL)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != NULL)
    return msg;
  b = mvc_bind(m, *sname, *tname, *cname, *access);
  if (b && b->ttype != coltype)
    throw(SQL,"sql.bind","tail type mismatch");
  if (b) {
    printf ("argc %d upd %d", pci->argc, upd);
    if (pci->argc == (9 + upd) && getArgType(mb, pci, 7 + upd) == TYPE_int) {
      BUN cnt = BATcount(b), psz;
      /* partitioned access */
      int part_nr = *getArgReference_int(stk, pci, 7 + upd);
      int nr_parts = *getArgReference_int(stk, pci, 8 + upd);
      printf (" Level 1\n" );

      if (*access == 0) {
        psz = cnt ? (cnt / nr_parts) : 0;
        bn = BATslice(b, part_nr * psz, (part_nr + 1 == nr_parts) ? cnt : ((part_nr + 1) * psz));
        BAThseqbase(bn, part_nr * psz);
      } else {
        /* BAT b holds the UPD_ID bat */
        oid l, h;
        BAT *c = mvc_bind(m, *sname, *tname, *cname, 0);
        if (c == NULL)
          throw(SQL,"sql.bind","Cannot access the update column");

        cnt = BATcount(c);
        psz = cnt ? (cnt / nr_parts) : 0;
        l = part_nr * psz;
        h = (part_nr + 1 == nr_parts) ? cnt : ((part_nr + 1) * psz);
        h--;
        bn = BATselect(b, NULL, &l, &h, 1, 1, 0);
        BBPunfix(c->batCacheid);
      }
      BBPunfix(b->batCacheid);
      b = bn;
    } else if (upd) {
      BAT *uv = mvc_bind(m, *sname, *tname, *cname, RD_UPD_VAL);
      bat *uvl = getArgReference_bat(stk, pci, 1);
      printf (" Level 2\n" );

      if (uv == NULL)
        throw(SQL,"sql.bind","Cannot access the update column");
      BBPkeepref(*bid = b->batCacheid);
      BBPkeepref(*uvl = uv->batCacheid);
      return MAL_SUCCEED;
    }
    if (upd) {
      bat *uvl = getArgReference_bat(stk, pci, 1);
      printf (" Level 3\n" );

      if (BATcount(b)) {
        BAT *uv = mvc_bind(m, *sname, *tname, *cname, RD_UPD_VAL);
        BAT *ui = mvc_bind(m, *sname, *tname, *cname, RD_UPD_ID);
        BAT *id;
        BAT *vl;
        if (ui == NULL)
          throw(SQL,"sql.bind","Cannot access the insert column");
        if (uv == NULL)
          throw(SQL,"sql.bind","Cannot access the update column");
        id = BATproject(b, ui);
        vl = BATproject(b, uv);
        assert(BATcount(id) == BATcount(vl));
        bat_destroy(ui);
        bat_destroy(uv);
        BBPkeepref(*bid = id->batCacheid);
        BBPkeepref(*uvl = vl->batCacheid);
      } else {
        sql_schema *s = mvc_bind_schema(m, *sname);
        sql_table *t = mvc_bind_table(m, s, *tname);
        sql_column *c = mvc_bind_column(m, t, *cname);

        *bid = e_bat(TYPE_oid);
        *uvl = e_bat(c->type.type->localtype);
      }
      BBPunfix(b->batCacheid);
    } else {
      printf (" Level 4\n" );
      BBPkeepref(*bid = b->batCacheid);
    }
    return MAL_SUCCEED;
  }
  if (*sname && strcmp(*sname, str_nil) != 0)
    throw(SQL, "sql.bind", "unable to find %s.%s(%s)", *sname, *tname, *cname);
  throw(SQL, "sql.bind", "unable to find %s(%s)", *tname, *cname);
}

/* str SQLtid(bat *result, mvc *m, str *sname, str *tname) */
str
VLTTid(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  bat *res = getArgReference_bat(stk, pci, 0);
  mvc *m = NULL;
  str msg;
  sql_trans *tr;
  str sname = *getArgReference_str(stk, pci, 3);
  str tname = *getArgReference_str(stk, pci, 4);

  sql_schema *s;
  sql_table *t;
  sql_column *c;
  BAT *tids;
  size_t nr, inr = 0;
  oid sb = 0;

  *res = bat_nil;
  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != NULL)
    return msg;
  tr = m->session->tr;
  if ((msg = checkSQLContext(cntxt)) != NULL)
    return msg;
  s = mvc_bind_schema(m, sname);
  if (s == NULL)
    throw(SQL, "sql.tid", "3F000!Schema missing");
  t = mvc_bind_table(m, s, tname);
  if (t == NULL)
    throw(SQL, "sql.tid", "42S02!Table missing");
  c = t->columns.set->h->data;

  nr = store_funcs.count_col(tr, c, 1);

  if (isTable(t) && t->access == TABLE_WRITABLE && (t->base.flag != TR_NEW /* alter */ ) &&
      t->persistence == SQL_PERSIST && !t->commit_action)
    inr = store_funcs.count_col(tr, c, 0);
  nr -= inr;
  if (pci->argc == 6) {	/* partitioned version */
    size_t cnt = nr;
    int part_nr = *getArgReference_int(stk, pci, 5);
    int nr_parts = *getArgReference_int(stk, pci, 6);

    nr /= nr_parts;
    sb = (oid) (part_nr * nr);
    if (nr_parts == (part_nr + 1)) {	/* last part gets the inserts */
      nr = cnt - (part_nr * nr);	/* keep rest */
      nr += inr;
    }
  } else {
    nr += inr;
  }

  /* create void,void bat with length and oid's set */
  tids = COLnew(sb, TYPE_void, 0, TRANSIENT);
  if (tids == NULL)
    throw(SQL, "sql.tid", MAL_MALLOC_FAIL);
  BATsetcount(tids, (BUN) nr);
  BATtseqbase(tids, sb);

  if (store_funcs.count_del(tr, t)) {
    BAT *d = store_funcs.bind_del(tr, t, RD_INS);
    BAT *diff;
    if( d == NULL)
      throw(SQL,"sql.tid","Can not bind delete column");

    diff = BATdiff(tids, d, NULL, NULL, 0, BUN_NONE);
    BBPunfix(d->batCacheid);
    BBPunfix(tids->batCacheid);
    BAThseqbase(diff, sb);
    tids = diff;
  }

  if (!(tids->batDirtydesc&2)) BATsetaccess(tids, BAT_READ);
  BBPkeepref(*res = tids->batCacheid);
  return MAL_SUCCEED;
}

