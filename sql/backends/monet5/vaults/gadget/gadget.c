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

#include "monetdb_config.h"
#include <glob.h>

/* clash with GDK? */
#undef htype
#undef ttype

#include "gadget.h"
#include "mutils.h"
#include "sql_mvc.h"
#include "sql_scenario.h"
#include "sql.h"
#include "clients.h"
#include "mal_exception.h"

static MT_Lock mt_gadget_lock;

#define GADGET_LOCK MT_lock_set(&mt_gadget_lock) 
#define GADGET_UNLOCK MT_lock_unset(&mt_gadget_lock) 

#define ATTACHDIR "call gadgetAttach('%s');"

static
int replaceCharacter(char s[], const char oldC, const char newC) {
  int j = 0;
  while (s[j] != '\0'){
    if (s[j] == oldC) {
      s[j] = newC;
    }
    j++;
  }
  return 0;
}
#ifndef NDEBUG

static
int printHeader(FILE *fp, Header h, char *fname)
{
  int res = 0;

  fprintf(fp,"fileName: %s\nnpartTotal: %d\nnpart[6]{ %d, %d, %d, %d, %d. %d}\nmass[6]{ %lf, %lf, %lf, %lf, %lf. %lf}\ntime: %lf\nredshift: %lf\nflag_sfr: %d\nflag_feedback: %d\nnpart_total[6]{ %d, %d, %d, %d, %d. %d}\nflag_cooling: %d\nnum_files: %d\nBoxSize: %lf\nOmega0: %lf\nOmegaLambda: %lf\nHubbleParameter: %lf\nflag_stellarage: %d\nflag_metals: %d\nhashtabsize: %d\nheaderOk: %d\nfirst_cell: %d\nlast_cell: %d\n",fname, h.npartTotal,h.npart[0], h.npart[1], h.npart[2], h.npart[3], h.npart[4], h.npart[5],h.mass[0], h.mass[1], h.mass[2], h.mass[3], h.mass[4], h.mass[5],h.time, h.redshift,h.flag_sfr, h.flag_feedback,h.npart_total[0], h.npart_total[1], h.npart_total[2], h.npart_total[3], h.npart_total[4], h.npart_total[5],h.flag_cooling, h.num_files,h.BoxSize, h.Omega0, h.OmegaLambda, h.HubbleParameter,h.flag_stellarage, h.flag_metals, h.hashtabsize,h.headerOk, h.first_cell, h.last_cell);

  return res;
}
#endif

static void
gadgetInitCatalog(mvc *m)
{
  sql_schema *sch;
  sql_table *gadget_fl, *gadget_tbl, *gadget_col;

  MT_lock_init(&mt_gadget_lock, "gadget.lock");

  sch = mvc_bind_schema(m, "sys");

  gadget_fl = mvc_bind_table(m, sch, "gadget_files");
  if (gadget_fl == NULL) {
    gadget_fl = mvc_create_table(m, sch, "gadget_files", tt_table, 0, SQL_PERSIST, 0, 2);
    mvc_create_column_(m, gadget_fl, "id", "int", 32);
    mvc_create_column_(m, gadget_fl, "name", "varchar", 80);
  }

  gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
  if (gadget_tbl == NULL) {
    gadget_tbl = mvc_create_table(m, sch, "gadget_tables", tt_table, 0, SQL_PERSIST, 0, 37);
    mvc_create_column_(m, gadget_tbl, "id", "int", 32);
    mvc_create_column_(m, gadget_tbl, "file_id", "int", 32);
    mvc_create_column_(m, gadget_tbl, "name", "varchar", 80);
    mvc_create_column_(m, gadget_tbl, "npart1", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart2", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart3", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart4", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart5", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart6", "int", 32);
    mvc_create_column_(m, gadget_tbl, "mass1", "double", 64);
    mvc_create_column_(m, gadget_tbl, "mass2", "double", 64);
    mvc_create_column_(m, gadget_tbl, "mass3", "double", 64);
    mvc_create_column_(m, gadget_tbl, "mass4", "double", 64);
    mvc_create_column_(m, gadget_tbl, "mass5", "double", 64);
    mvc_create_column_(m, gadget_tbl, "mass6", "double", 64);
    mvc_create_column_(m, gadget_tbl, "time", "double", 64);
    mvc_create_column_(m, gadget_tbl, "redshift", "double", 64);
    mvc_create_column_(m, gadget_tbl, "flag_sfr", "int", 32);
    mvc_create_column_(m, gadget_tbl, "flag_feedback", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart_total1", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart_total2", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart_total3", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart_total4", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart_total5", "int", 32);
    mvc_create_column_(m, gadget_tbl, "npart_total6", "int", 32);
    mvc_create_column_(m, gadget_tbl, "flag_cooling", "int", 32);
    mvc_create_column_(m, gadget_tbl, "num_files", "int", 32);
    mvc_create_column_(m, gadget_tbl, "BoxSize", "double", 64);
    mvc_create_column_(m, gadget_tbl, "Omega0", "double", 64);
    mvc_create_column_(m, gadget_tbl, "OmegaLambda", "double", 64);
    mvc_create_column_(m, gadget_tbl, "HubbleParameter", "double", 64);
    mvc_create_column_(m, gadget_tbl, "flag_stellarage", "int", 32);
    mvc_create_column_(m, gadget_tbl, "flag_metals", "int", 32);
    mvc_create_column_(m, gadget_tbl, "hashtabsize", "int", 32);
    mvc_create_column_(m, gadget_tbl, "headerOk", "int", 32);
    mvc_create_column_(m, gadget_tbl, "first_cell", "int", 32);
    mvc_create_column_(m, gadget_tbl, "last_cell", "int", 32);
  }

  gadget_col = mvc_bind_table(m, sch, "gadget_columns");
  if (gadget_col == NULL) {
    gadget_col = mvc_create_table(m, sch, "gadget_columns", tt_table, 0, SQL_PERSIST, 0, 15);
    mvc_create_column_(m, gadget_col, "id", "int", 32);
    mvc_create_column_(m, gadget_col, "file_id", "int", 32);
    mvc_create_column_(m, gadget_col, "table_id", "int", 32);
    mvc_create_column_(m, gadget_col, "ScaleX", "double", 64);
    mvc_create_column_(m, gadget_col, "ScaleY", "double", 64);
    mvc_create_column_(m, gadget_col, "ScaleZ", "double", 64);
    mvc_create_column_(m, gadget_col, "OffsetX", "double", 64);
    mvc_create_column_(m, gadget_col, "OffsetY", "double", 64);
    mvc_create_column_(m, gadget_col, "OffsetZ", "double", 64);
    mvc_create_column_(m, gadget_col, "MinX", "double", 64);
    mvc_create_column_(m, gadget_col, "MinY", "double", 64);
    mvc_create_column_(m, gadget_col, "MinZ", "double", 64);
    mvc_create_column_(m, gadget_col, "MaxX", "double", 64);
    mvc_create_column_(m, gadget_col, "MaxY", "double", 64);
    mvc_create_column_(m, gadget_col, "MaxZ", "double", 64);
  }
}

str gadgetListDir(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
#if 0
  (void) cntxt;
  (void) mb;
  (void) stk;
  (void) pci;
  return MAL_SUCCEED;
#endif
  str msg = MAL_SUCCEED;
  str dir = *getArgReference_str(stk, pci, 1);
  DIR *dp;
  struct dirent *ep;
  FILE *stream = NULL;
  char *s;
  (void)mb;

  dp = opendir(dir);
  if (dp != NULL) {
    char stmt[BUFSIZ];
    char fname[BUFSIZ];

    s = stmt;

    while ((ep = readdir(dp)) != NULL && !msg) {
      if (ep->d_name[0] != '.') {
        snprintf(fname, BUFSIZ, "%s%s", dir, ep->d_name);

        if ( !(stream = fopen(fname, "rb")) ) {
          msg = createException(MAL, "gadgetDir", "Missing GADGET file %s: %s", fname, strerror(errno));
          return msg;
        }
        fclose(stream);
        snprintf(stmt, BUFSIZ, ATTACHDIR, fname);
        msg = SQLstatementIntern(cntxt, &s, "gadget.listofdir", TRUE, FALSE, NULL);
      }
    }
    (void)closedir(dp);
  } else
    msg = createException(MAL, "gadgetDir", "Couldn't open the directory");

  return msg;
#if 0
  return MAL_SUCCEED;
#endif
}

static void
getPHkeyMinMax(PHBins bins, int numBins, int i, int *min_phkey, int *max_phkey)
{
  int j, min = bins.id[0], max = bins.id[0];
  for (j = 1; j < numBins; j++) {
    if (bins.id[j] < min)
      min = bins.id[j];
    if (bins.id[j] > max)
      max = bins.id[j];
  }

  if (!i || (min < *min_phkey))
    *min_phkey = min;

  if (!i || (max > *max_phkey))
    *max_phkey = max;

}

str gadgetListDirAll(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
#if 0
  (void) cntxt;
  (void) mb;
  (void) stk;
  (void) pci;
  return MAL_SUCCEED;
#endif
  str msg = MAL_SUCCEED;
  str dir = *getArgReference_str(stk, pci, 1);
  mvc *m = NULL;
  DIR *dp;
  struct dirent *ep;
  FILE *stream = NULL;
  int i = 0, j = 0, min_snapnum = 0, max_snapnum = 0, min_phkey = 0, max_phkey = 0;
  sql_trans *tr;
  sql_schema *sch;
  sql_table *gadget_fl, *gadget_tbl, *tbl = NULL;
  sql_column *col;
  oid rid = oid_nil, fid = oid_nil, tid = oid_nil;
  char *tname_low = NULL;
  int cnum;
  Header h;
  GadgetParticles particles;
  (void)mb;

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    return msg;

  dp = opendir(dir);
  if (dp != NULL) {
    char fname[BUFSIZ];
    char *tname = basename(dir);
    char *dquery = NULL, *query = NULL;
    char maxval[BUFSIZ], minval[BUFSIZ];
    sql_column *c = NULL;
    lng nils = 0;
    lng uniq = 0;
    lng sz = 0;
    int width = 4;
        
#ifndef NDEBUG
    fprintf(stderr, "The dir path is %s and dirname is %s!!!\n", dir, tname);
#endif

    while ((ep = readdir(dp)) != NULL && !msg) {
      if (ep->d_name[0] != '.') {
        char *snapnum, *ptr = NULL;
        PHBins bins;
        snprintf(fname, BUFSIZ, "%s/%s", dir, ep->d_name);

        /*Open the file*/
        if ( !(stream = fopen(fname, "rb")) ) {
          msg = createException(MAL, "ReadGadgetFile", "Missing GADGET file %s: %s", fname, strerror(errno));
          return msg;
        }
        /*Get snapnum from the file name*/
        ptr = strrchr(fname, '_');
        ptr = ptr+1;
        snapnum = GDKstrdup(ptr);
        snapnum[3] = '\0';
        if ( !i || min_snapnum >= atoi(snapnum))
          min_snapnum  = atoi(snapnum);
        if (!i || max_snapnum <= atoi(snapnum))
          max_snapnum  = atoi(snapnum);
        GDKfree(snapnum);

        /*Get the header*/
        headerInit(&h, stream);

#ifndef NDEBUG
        printHeader(stderr, h, fname);
#endif
        /*Get the PHBins min and max*/
        /*Check if there are PHBins, if yes collect them*/
        if (h.hashtabsize > 0)
          {
            long index = 0, retval = 0, nbins = h.last_cell - h.first_cell + 1;
            char* binbytes = NULL;
            PHBin *prev = NULL;

            if (PHBins_init(&bins)) {
              fprintf(stderr, "GadgetPHBins: PHBinList_init failed!!!\n");
              goto out;
            }

            binbytes = (char*) malloc(sizeof(char) * nbins * 4);
            if (!binbytes) {
              PHBins_destroy(&bins);
              fprintf(stderr, "GadgetPHBins: Malloc of binbytes failed!!!\n");
              goto out;
            }

            index = 308 + h.npartTotal * 32; 
            fseek(stream, index, SEEK_SET);

            retval = fread(binbytes, nbins * 4, 1, stream);
#ifndef NDEBUG
            assert(retval);
#else
            (void) retval;
#endif

            for (j = 0; j < nbins; j++)
              {
                PHBin *bin = (PHBin *) malloc (sizeof(PHBin));
                bin->id = h.first_cell + j;
                bin->start = *(int*) (&binbytes[j * 4]);
                if (prev != NULL)
                  {
                    prev->count = bin->start - prev->start;
                    PHBins_add(&bins, prev->id, prev->start, prev->count);
                  }
                if (prev)
                  free(prev);
                prev = bin;
              }
            prev->count = h.npartTotal - prev->start;
            PHBins_add(&bins, prev->id, prev->start, prev->count);
            if (prev)
              free(prev);
            if (binbytes)
              free(binbytes);
          }
        getPHkeyMinMax(bins, h.last_cell - h.first_cell + 1, i, &min_phkey, &max_phkey);

        PHBins_destroy(&bins);
        /*Count the number of particles which will be loaded in the future*/
        sz += h.npartTotal;

        /*TODO: Make sure you really need it otherwise drop it*/
        if (gadgetParticles_init(&particles)) {
          fclose(stream);
          msg = createException(SQL, "gadget.attach", "gadgetParticles_init failed!!!\n");
          return msg;
        }

        /* if needed, instantiate the schema and gather all appropriate tables */
        tr = m->session->tr;
        sch = mvc_bind_schema(m, "sys");

        gadget_fl = mvc_bind_table(m, sch, "gadget_files");
        if (gadget_fl == NULL)
          gadgetInitCatalog(m);

        gadget_fl = mvc_bind_table(m, sch, "gadget_files");
        gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
        //gadget_col = mvc_bind_table(m, sch, "gadget_columns");

        /* check if the file is already attached */
        col = mvc_bind_column(m, gadget_fl, "name");
        rid = table_funcs.column_find_row(m->session->tr, col, fname, NULL);
        if (rid != oid_nil) {
          fclose(stream);
          msg = createException(SQL, "gadget.attach", "File %s already attached\n", fname);
          return msg;
        }
        /* add row in the gadget_files catalog table */
        col = mvc_bind_column(m, gadget_fl, "id");
        fid = store_funcs.count_col(tr, col, 1) + 1;
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_fl, "id"), &fid, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_fl, "name"), fname, TYPE_str);

        if (i == 0) {
          col = mvc_bind_column(m, gadget_tbl, "id");
          tid = store_funcs.count_col(tr, col, 1) + 1;

          tname_low = toLower(tname);

          /* check table name for existence in the gadget catalog */
          col = mvc_bind_column(m, gadget_tbl, "name");
          rid = table_funcs.column_find_row(m->session->tr, col, tname_low, NULL);
          /* or as regular SQL table */
          tbl = mvc_bind_table(m, sch, tname_low);
          if (rid != oid_nil || tbl) {
            msg = createException(SQL, "gadget.attach", "Table %s already exists\n", tname_low);
            return msg;
          }
        }

        /*Store header values*/
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "id"), &tid, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "file_id"), &fid, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "name"), tname_low, TYPE_str);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart1"), &h.npart[0], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart2"), &h.npart[1], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart3"), &h.npart[2], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart4"), &h.npart[3], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart5"), &h.npart[4], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart6"), &h.npart[5], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "mass1"), &h.mass[0], TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "mass2"), &h.mass[1], TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "mass3"), &h.mass[2], TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "mass4"), &h.mass[3], TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "mass5"), &h.mass[4], TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "mass6"), &h.mass[5], TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "time"), &h.time, TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "redshift"), &h.redshift, TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "flag_sfr"), &h.flag_sfr, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "flag_feedback"), &h.flag_feedback, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart_total1"), &h.npart_total[0], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart_total2"), &h.npart_total[1], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart_total3"), &h.npart_total[2], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart_total4"), &h.npart_total[3], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart_total5"), &h.npart_total[4], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "npart_total6"), &h.npart_total[5], TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "flag_cooling"), &h.flag_cooling, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "num_files"), &h.num_files, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "BoxSize"), &h.BoxSize, TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "Omega0"), &h.Omega0, TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "OmegaLambda"), &h.OmegaLambda, TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "HubbleParameter"), &h.HubbleParameter, TYPE_dbl);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "flag_stellarage"), &h.flag_stellarage, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "flag_metals"), &h.flag_metals, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "hashtabsize"), &h.hashtabsize, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "headerOk"), &h.headerOk, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "first_cell"), &h.first_cell, TYPE_int);
        store_funcs.append_col(m->session->tr,
                               mvc_bind_column(m, gadget_tbl, "last_cell"), &h.last_cell, TYPE_int);

        /*Collect stats or metadata for the particles*/

        /*Store that information in gadget_col*/

        /*Let's create a table per GadgetFile*/
        if (i == 0) {
          cnum = 9;//x, y, z. TODO: Add all available columnt 
          tbl = mvc_create_table(m, sch, tname_low, tt_table, 0, SQL_PERSIST, 0, cnum); 
          mvc_create_column_(m, tbl, "id", "bigint", 64); 
          mvc_create_column_(m, tbl, "posx", "real", 32); 
          mvc_create_column_(m, tbl, "posy", "real", 32); 
          mvc_create_column_(m, tbl, "posz", "real", 32); 
          mvc_create_column_(m, tbl, "velx", "real", 32); 
          mvc_create_column_(m, tbl, "vely", "real", 32); 
          mvc_create_column_(m, tbl, "velz", "real", 32); 
          mvc_create_column_(m, tbl, "snapnum", "int", 32); 
          mvc_create_column_(m, tbl, "phkey", "int", 32); 
        }
        if (stream)
          fclose (stream);
        i++;
      }
    }
    (void)closedir(dp);

    /*Prepare statement to update stats table with info about snapnum*/
    dquery = (char *) GDKzalloc(8192);
    query = (char *) GDKzalloc(8192);

    if (!(dquery && query)) {
      GDKfree(dquery);
      GDKfree(query);
      msg = createException(MAL, "gadgetListDirAll", "malloc failed!!!");
      goto out;
    }

    c = mvc_bind_column(m, tbl, "snapnum");
    if (!c) {
      GDKfree(dquery);
      GDKfree(query);
      msg = createException(MAL, "gadgetListDirAll", "Not able to get column snapnum from table %s!!!", tname_low);
      goto out;
    }

    snprintf(minval, BUFSIZ, "%d", min_snapnum);
    snprintf(maxval, BUFSIZ, "%d", max_snapnum);

    /*Remove all stats related with snapnum column from this table*/
    snprintf(dquery, 8192, "delete from sys.statistics where \"column_id\" = %d;", c->base.id);

    /*build the update query*/
    snprintf(query, 8192, "insert into sys.statistics values(%d,'%s',%d,now()," LLFMT "," LLFMT "," LLFMT "," LLFMT ",'%s','%s',%s);", c->base.id, c->type.type->sqlname, width, sz, sz, uniq, nils, minval, maxval, (min_snapnum == max_snapnum) ? "true" : "false");

    msg = SQLstatementIntern(cntxt, &dquery, "gadgetListDirAll", TRUE, FALSE, NULL);
    if (msg) {
      GDKfree(dquery);
      GDKfree(query);
      goto out;
    }
    msg = SQLstatementIntern(cntxt, &query, "gadgetListDirAll", TRUE, FALSE, NULL);
    if (msg) {
      GDKfree(dquery);
      GDKfree(query);
      goto out;
    }
    GDKfree(dquery);
    GDKfree(query);

    /*Prepare statement to update stats table with info about phkey*/
    dquery = (char *) GDKzalloc(8192);
    query = (char *) GDKzalloc(8192);

    if (!(dquery && query)) {
      GDKfree(dquery);
      GDKfree(query);
      msg = createException(MAL, "gadgetListDirAll", "malloc failed!!!");
      goto out;
    }

    c = mvc_bind_column(m, tbl, "phkey");
    if (!c) {
      GDKfree(dquery);
      GDKfree(query);
      msg = createException(MAL, "gadgetListDirAll", "Not able to get column snapnum from table %s!!!", tname_low);
      goto out;
    }

    snprintf(minval, BUFSIZ, "%d", min_phkey);
    snprintf(maxval, BUFSIZ, "%d", max_phkey);

    /*Remove all stats related with snapnum column from this table*/
    snprintf(dquery, 8192, "delete from sys.statistics where \"column_id\" = %d;", c->base.id);

    /*build the update query*/
    snprintf(query, 8192, "insert into sys.statistics values(%d,'%s',%d,now()," LLFMT "," LLFMT "," LLFMT "," LLFMT ",'%s','%s',%s);", c->base.id, c->type.type->sqlname, width, sz, sz, uniq, nils, minval, maxval, (min_phkey == max_phkey) ? "true" : "false");

    msg = SQLstatementIntern(cntxt, &dquery, "gadgetListDirAll", TRUE, FALSE, NULL);
    if (msg) {
      GDKfree(dquery);
      GDKfree(query);
      goto out;
    }
    msg = SQLstatementIntern(cntxt, &query, "gadgetListDirAll", TRUE, FALSE, NULL);
    if (msg) {
      GDKfree(dquery);
      GDKfree(query);
      goto out;
    }
    GDKfree(dquery);
    GDKfree(query);
  } else
    msg = createException(MAL, "gadgetListDirAll", "Couldn't open the directory");

 out:
  return msg;
#if 0
  return MAL_SUCCEED;
#endif
}

str gadgetListDirPat(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
#if 0
  (void) cntxt;
  (void) mb;
  (void) stk;
  (void) pci;
  return MAL_SUCCEED;
#endif
  str msg = MAL_SUCCEED;
  str dir = *getArgReference_str(stk, pci, 1);
  str pat = *getArgReference_str(stk, pci, 2);
  char *s;
  glob_t globbuf;
  FILE *stream = NULL;
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

    if ( !(stream = fopen(fname, "rb")) ) {
      msg = createException(MAL, "gadgetDir", "Missing GADGET file %s: %s", fname, strerror(errno));
      return msg;
    }
    fclose(stream);
    snprintf(stmt, BUFSIZ, ATTACHDIR, fname);
    msg = SQLstatementIntern(cntxt, &s, "gadget.listofdirpat", TRUE, FALSE, NULL);
    break;
  }

  return msg;
#if 0
  return MAL_SUCCEED;
#endif
}

str
gadgetTest(int *res, str *fname)
{
  FILE *fp = NULL;       /* pointer to the GADGET file, defined in gadgetio.h */
  str msg = MAL_SUCCEED;

  *res = 0;
  if ( !(fp = fopen(*fname, "rb")) )
    msg = createException(MAL, "gadget.test", "Missing GADGET file %s: %s", *fname, strerror(errno));
  else {
    Header h;
    headerInit(&h, fp);
    *res = h.npartTotal;
    fclose(fp);
  }

  return msg;
}

str gadgetAttach(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  mvc *m = NULL;
  sql_trans *tr;
  sql_schema *sch;
  sql_table *gadget_fl, *gadget_tbl, *tbl = NULL;
  sql_column *col;
  str msg = MAL_SUCCEED;
  str fname = *getArgReference_str(stk, pci, 1);
  oid rid = oid_nil, fid = oid_nil, tid = oid_nil;
  char *tname_low = NULL, *s, bname[BUFSIZ];
  int cnum;
  Header h;
  FILE *stream = NULL;
  GadgetParticles particles;

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    return msg;

  /*Open the file*/
  if ( !(stream = fopen(fname, "rb")) ) {
    msg = createException(MAL, "ReadGadgetFile", "Missing GADGET file %s: %s", fname, strerror(errno));
    return msg;
  }

  /*Get the header*/
  headerInit(&h, stream);

#ifndef NDEBUG
  printHeader(stderr, h, fname);
#endif


  if (gadgetParticles_init(&particles)) {
    fclose(stream);
    msg = createException(SQL, "gadget.attach", "gadgetParticles_init failed!!!\n");
    return msg;
  }

  /* if needed, instantiate the schema and gather all appropriate tables */
  tr = m->session->tr;
  sch = mvc_bind_schema(m, "sys");

  gadget_fl = mvc_bind_table(m, sch, "gadget_files");
  if (gadget_fl == NULL)
    gadgetInitCatalog(m);

  gadget_fl = mvc_bind_table(m, sch, "gadget_files");
  gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
  //gadget_col = mvc_bind_table(m, sch, "gadget_columns");

  /* check if the file is already attached */
  col = mvc_bind_column(m, gadget_fl, "name");
  rid = table_funcs.column_find_row(m->session->tr, col, fname, NULL);
  if (rid != oid_nil) {
    fclose(stream);
    msg = createException(SQL, "gadget.attach", "File %s already attached\n", fname);
    return msg;
  }

  /* add row in the gadget_files catalog table */
  col = mvc_bind_column(m, gadget_fl, "id");
  fid = store_funcs.count_col(tr, col, 1) + 1;
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_fl, "id"), &fid, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_fl, "name"), fname, TYPE_str);

  col = mvc_bind_column(m, gadget_tbl, "id");
  tid = store_funcs.count_col(tr, col, 1) + 1;

  /* extract the file name from the absolute path */
  if ((s = strrchr(fname, DIR_SEP)) == NULL)
    s = fname;
  else
    s++;

  replaceCharacter(s, '.', '_');
  strcpy(bname, s);
  if (s) *s = 0;

  tname_low = toLower(bname);

  /* check table name for existence in the gadget catalog */
  col = mvc_bind_column(m, gadget_tbl, "name");
  rid = table_funcs.column_find_row(m->session->tr, col, tname_low, NULL);
  /* or as regular SQL table */
  tbl = mvc_bind_table(m, sch, tname_low);
  if (rid != oid_nil || tbl) {
    msg = createException(SQL, "gadget.attach", "Table %s already exists\n", tname_low);
    return msg;
  }

  /*Store header values*/
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "id"), &tid, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "file_id"), &fid, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "name"), tname_low, TYPE_str);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart1"), &h.npart[0], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart2"), &h.npart[1], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart3"), &h.npart[2], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart4"), &h.npart[3], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart5"), &h.npart[4], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart6"), &h.npart[5], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "mass1"), &h.mass[0], TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "mass2"), &h.mass[1], TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "mass3"), &h.mass[2], TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "mass4"), &h.mass[3], TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "mass5"), &h.mass[4], TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "mass6"), &h.mass[5], TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "time"), &h.time, TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "redshift"), &h.redshift, TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "flag_sfr"), &h.flag_sfr, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "flag_feedback"), &h.flag_feedback, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart_total1"), &h.npart_total[0], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart_total2"), &h.npart_total[1], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart_total3"), &h.npart_total[2], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart_total4"), &h.npart_total[3], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart_total5"), &h.npart_total[4], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "npart_total6"), &h.npart_total[5], TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "flag_cooling"), &h.flag_cooling, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "num_files"), &h.num_files, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "BoxSize"), &h.BoxSize, TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "Omega0"), &h.Omega0, TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "OmegaLambda"), &h.OmegaLambda, TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "HubbleParameter"), &h.HubbleParameter, TYPE_dbl);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "flag_stellarage"), &h.flag_stellarage, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "flag_metals"), &h.flag_metals, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "hashtabsize"), &h.hashtabsize, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "headerOk"), &h.headerOk, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "first_cell"), &h.first_cell, TYPE_int);
  store_funcs.append_col(m->session->tr,
                         mvc_bind_column(m, gadget_tbl, "last_cell"), &h.last_cell, TYPE_int);

  /*Collect stats or metadata for the particles*/

  /*Store that information in gadget_col*/

  /*Let's create a table per GadgetFile*/
  cnum = 9;//x, y, z. TODO: Add all available columnt 
  tbl = mvc_create_table(m, sch, tname_low, tt_table, 0, SQL_PERSIST, 0, cnum); 
  mvc_create_column_(m, tbl, "id", "bigint", 64); 
  mvc_create_column_(m, tbl, "posx", "real", 32); 
  mvc_create_column_(m, tbl, "posy", "real", 32); 
  mvc_create_column_(m, tbl, "posz", "real", 32); 
  mvc_create_column_(m, tbl, "velx", "real", 32); 
  mvc_create_column_(m, tbl, "vely", "real", 32); 
  mvc_create_column_(m, tbl, "velz", "real", 32); 
  mvc_create_column_(m, tbl, "snapnum", "int", 32); 
  mvc_create_column_(m, tbl, "phkey", "int", 32); 

  if (stream)
    fclose (stream);

  return MAL_SUCCEED;
}

str gadgetAttachAll(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  int i = 0;
  mvc *m = NULL;
  sql_trans *tr;
  sql_schema *sch;
  sql_table *gadget_fl, *gadget_tbl, *tbl = NULL;
  sql_column *col;
  str msg = MAL_SUCCEED;
  str fname = NULL, tname = *getArgReference_str(stk, pci, 1);
  oid rid = oid_nil, fid = oid_nil, tid = oid_nil;
  char *tname_low = NULL;
  int cnum;
  Header h;
  FILE *stream = NULL;
  GadgetParticles particles;

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    return msg;

  for (i = 2; i < pci->argc; i++) {
    fname = *getArgReference_str(stk, pci, i);
    /*Open the file*/
    if ( !(stream = fopen(fname, "rb")) ) {
      msg = createException(MAL, "ReadGadgetFile", "Missing GADGET file %s: %s", fname, strerror(errno));
      return msg;
    }

    /*Get the header*/
    headerInit(&h, stream);

#ifndef NDEBUG
    printHeader(stderr, h, fname);
#endif

    if (gadgetParticles_init(&particles)) {
      fclose(stream);
      msg = createException(SQL, "gadget.attach", "gadgetParticles_init failed!!!\n");
      return msg;
    }

    /* if needed, instantiate the schema and gather all appropriate tables */
    tr = m->session->tr;
    sch = mvc_bind_schema(m, "sys");

    gadget_fl = mvc_bind_table(m, sch, "gadget_files");
    if (gadget_fl == NULL)
      gadgetInitCatalog(m);

    gadget_fl = mvc_bind_table(m, sch, "gadget_files");
    gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
    //gadget_col = mvc_bind_table(m, sch, "gadget_columns");

    /* check if the file is already attached */
    col = mvc_bind_column(m, gadget_fl, "name");
    rid = table_funcs.column_find_row(m->session->tr, col, fname, NULL);
    if (rid != oid_nil) {
      fclose(stream);
      msg = createException(SQL, "gadget.attach", "File %s already attached\n", fname);
      return msg;
    }
    /* add row in the gadget_files catalog table */
    col = mvc_bind_column(m, gadget_fl, "id");
    fid = store_funcs.count_col(tr, col, 1) + 1;
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_fl, "id"), &fid, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_fl, "name"), fname, TYPE_str);

    if (i == 2) {
      col = mvc_bind_column(m, gadget_tbl, "id");
      tid = store_funcs.count_col(tr, col, 1) + 1;

      tname_low = toLower(tname);

      /* check table name for existence in the gadget catalog */
      col = mvc_bind_column(m, gadget_tbl, "name");
      rid = table_funcs.column_find_row(m->session->tr, col, tname_low, NULL);
      /* or as regular SQL table */
      tbl = mvc_bind_table(m, sch, tname_low);
      if (rid != oid_nil || tbl) {
        msg = createException(SQL, "gadget.attach", "Table %s already exists\n", tname_low);
        return msg;
      }
    }

    /*Store header values*/
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "id"), &tid, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "file_id"), &fid, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "name"), tname_low, TYPE_str);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart1"), &h.npart[0], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart2"), &h.npart[1], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart3"), &h.npart[2], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart4"), &h.npart[3], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart5"), &h.npart[4], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart6"), &h.npart[5], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "mass1"), &h.mass[0], TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "mass2"), &h.mass[1], TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "mass3"), &h.mass[2], TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "mass4"), &h.mass[3], TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "mass5"), &h.mass[4], TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "mass6"), &h.mass[5], TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "time"), &h.time, TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "redshift"), &h.redshift, TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "flag_sfr"), &h.flag_sfr, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "flag_feedback"), &h.flag_feedback, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart_total1"), &h.npart_total[0], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart_total2"), &h.npart_total[1], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart_total3"), &h.npart_total[2], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart_total4"), &h.npart_total[3], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart_total5"), &h.npart_total[4], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "npart_total6"), &h.npart_total[5], TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "flag_cooling"), &h.flag_cooling, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "num_files"), &h.num_files, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "BoxSize"), &h.BoxSize, TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "Omega0"), &h.Omega0, TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "OmegaLambda"), &h.OmegaLambda, TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "HubbleParameter"), &h.HubbleParameter, TYPE_dbl);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "flag_stellarage"), &h.flag_stellarage, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "flag_metals"), &h.flag_metals, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "hashtabsize"), &h.hashtabsize, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "headerOk"), &h.headerOk, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "first_cell"), &h.first_cell, TYPE_int);
    store_funcs.append_col(m->session->tr,
                           mvc_bind_column(m, gadget_tbl, "last_cell"), &h.last_cell, TYPE_int);

    /*Collect stats or metadata for the particles*/

    /*Store that information in gadget_col*/

    /*Let's create a table per GadgetFile*/
    if (i == 2) {
      cnum = 9;//x, y, z. TODO: Add all available columnt 
      tbl = mvc_create_table(m, sch, tname_low, tt_table, 0, SQL_PERSIST, 0, cnum); 
      mvc_create_column_(m, tbl, "id", "bigint", 64); 
      mvc_create_column_(m, tbl, "posx", "real", 32); 
      mvc_create_column_(m, tbl, "posy", "real", 32); 
      mvc_create_column_(m, tbl, "posz", "real", 32); 
      mvc_create_column_(m, tbl, "velx", "real", 32); 
      mvc_create_column_(m, tbl, "vely", "real", 32); 
      mvc_create_column_(m, tbl, "velz", "real", 32); 
      mvc_create_column_(m, tbl, "snapnum", "int", 32); 
      mvc_create_column_(m, tbl, "phkey", "int", 32); 
    }
    if (stream)
      fclose (stream);

  }
  return MAL_SUCCEED;
}

str gadgetLoadTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  mvc *m = NULL;
  sql_schema *sch;
  sql_table *gadget_fl, *gadget_tbl, *tbl = NULL;
  sql_column *col;
  str tname = *getArgReference_str(stk, pci, 1);
  str fname;
  str msg = MAL_SUCCEED;
  oid rid = oid_nil, frid = oid_nil;
  int fid, i, retval; 
  size_t sz;
  FILE *stream = NULL;
  Header h;
  long index;
  char *posB = NULL, *velB=NULL, *idB=NULL;
  sql_column *colid, *colposX, *colposY, *colposZ, *colvelX, *colvelY, *colvelZ; 
  BAT *id, *posX, *posY, *posZ, *velX, *velY, *velZ;
  lng *pid;
  flt *pposX, *pposY, *pposZ, *pvelX, *pvelY, *pvelZ;

#ifndef NDEBUG
  int time0;
#endif
  (void) retval;

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    return msg;
  sch = mvc_bind_schema(m, "sys");

  gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
  if (gadget_tbl == NULL) {
    msg = createException(MAL, "gadget.loadtable", "GADGET catalog is missing.\n");
    return msg;
  }

  tbl = mvc_bind_table(m, sch, tname);
  if (tbl == NULL) {
    msg = createException(MAL, "gadget.loadtable", "Could not find table %s.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, tbl, "id");
  sz = store_funcs.count_col(m->session->tr, col, 1);
  if (sz != 0) {
    msg = createException(MAL, "gadget.loadtable", "Table %s is not empty.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, gadget_tbl, "name");
  rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
  if (rid == oid_nil) {
    msg = createException(MAL, "gadget.loadtable", "Table %s is unknown to the GADGET catalog. Attach first the containing file\n", tname);
    return msg;
  }

  /* Open GADGET file */
  col = mvc_bind_column(m, gadget_tbl, "file_id");
  fid = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

  gadget_fl = mvc_bind_table(m, sch, "gadget_files");
  col = mvc_bind_column(m, gadget_fl, "id");
  frid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
  col = mvc_bind_column(m, gadget_fl, "name");
  fname = (char *)table_funcs.column_find_value(m->session->tr, col, frid);

  /*Open Gadget File*/
  stream = fopen(fname, "rb");

  if (!stream) {
    msg = createException(MAL, "ReadGadgetFile", "Missing GADGET file %s: %s", fname, strerror(errno));
    return msg;
  }

  /*Get the header*/
  headerInit(&h, stream);

  /*data load*/
#ifndef NDEBUG
  fprintf(stderr,"#Loading %d rows in table %s\n", h.npartTotal, tname);
  time0 = GDKms();
#endif

  colid = mvc_bind_column(m, tbl, "id"); 
  colposX = mvc_bind_column(m, tbl, "posx"); 
  colposY = mvc_bind_column(m, tbl, "posy"); 
  colposZ = mvc_bind_column(m, tbl, "posz"); 
  colvelX = mvc_bind_column(m, tbl, "velx"); 
  colvelY = mvc_bind_column(m, tbl, "vely"); 
  colvelZ = mvc_bind_column(m, tbl, "velz"); 
  id = COLnew(0, TYPE_lng, h.npartTotal, PERSISTENT); 
  posX = COLnew(0, TYPE_flt, h.npartTotal, PERSISTENT); 
  posY = COLnew(0, TYPE_flt, h.npartTotal, PERSISTENT); 
  posZ = COLnew(0, TYPE_flt, h.npartTotal, PERSISTENT); 
  velX = COLnew(0, TYPE_flt, h.npartTotal, PERSISTENT); 
  velY = COLnew(0, TYPE_flt, h.npartTotal, PERSISTENT); 
  velZ = COLnew(0, TYPE_flt, h.npartTotal, PERSISTENT); 

  if ( id == NULL || posX == NULL || posY == NULL || posZ == NULL || velX == NULL || velY == NULL || velZ == NULL) { 
    fclose(stream);
    msg = createException(MAL, "gadget.gadgetload", "Malloc failed"); 
    return msg; 
  } 

  pid = (lng *) Tloc(id, 0);
  pposX = (flt *) Tloc(posX, 0);
  pposY = (flt *) Tloc(posY, 0);
  pposZ = (flt *) Tloc(posZ, 0);
  pvelX = (flt *) Tloc(velX, 0);
  pvelY = (flt *) Tloc(velY, 0);
  pvelZ = (flt *) Tloc(velZ, 0);

  posB = (char*)GDKmalloc(sizeof(char) * 12 * h.npartTotal);
  velB = (char*)GDKmalloc(sizeof(char) * 12 * h.npartTotal);
  idB = (char*)GDKmalloc(sizeof(char) * 8 * h.npartTotal);

  index = MRSNAP_POS_OFFSET;

  //NOTE: Convert C# blob.Seek to fseek
  //stream.Seek(index, SeekOrigin.Begin);
  fseek(stream, index, SEEK_SET);

  //NOTE: Convert C# blob.Read to fread
  //int retval = stream.Read(pos, 0, 12 * header.npartTotal);
  retval = fread(posB, 12 * h.npartTotal, 1, stream);
  assert(retval);

  index = MRSNAP_VEL_OFFSET + h.npartTotal * 12; // skip over positions

  //NOTE: Convert C# blob.Seek to fseek
  //stream.Seek(index, SeekOrigin.Begin);
  fseek(stream, index, SEEK_SET);

  //NOTE: Convert C# blob.Read to fread
  //retval = stream.Read(vel, 0, 12 * header.npartTotal);
  retval = fread(velB, 12 * h.npartTotal, 1, stream);
  assert(retval);

  index = MRSNAP_ID_OFFSET + h.npartTotal * 24; // skip over positions and velocities
  //NOTE: Convert C# blob.Seek to fseek
  //stream.Seek(index, SeekOrigin.Begin);
  fseek(stream, index, SEEK_SET);

  //NOTE: Convert C# blob.Read to fread
  //retval = stream.Read(id, 0, 8 * header.npartTotal);
  retval = fread(idB, 8 * h.npartTotal, 1, stream);
  assert(retval);

  for (i = 0; i < h.npartTotal; i++)
    {
      int j = i * 12;
      pid[i] = *(int64_t*)(&idB[i*8]);
      //pposX[i] = convertToFloat(&posB[j]);
      //pposY[i] = convertToFloat(&posB[j+4]);
      //pposZ[i] = convertToFloat(&posB[j+8]);
      pposX[i] = *(float*)(&posB[j]);
      pposY[i] = *(float*)(&posB[j+4]);
      pposZ[i] = *(float*)(&posB[j+8]);
      //pvelX[i] = convertToFloat(&velB[j]);
      //pvelY[i] = convertToFloat(&velB[j+4]);
      //pvelZ[i] = convertToFloat(&velB[j+8]);
      pvelX[i] = *(float*)(&velB[j]);
      pvelY[i] = *(float*)(&velB[j+4]);
      pvelZ[i] = *(float*)(&velB[j+8]);
    }

  BATsetcount(id, h.npartTotal);
  BATsetcount(posX, h.npartTotal);
  BATsetcount(posY, h.npartTotal);
  BATsetcount(posZ, h.npartTotal);
  BATsetcount(velX, h.npartTotal);
  BATsetcount(velY, h.npartTotal);
  BATsetcount(velZ, h.npartTotal);

  id->tsorted = 0;
  id->trevsorted = 0;
  posX->tsorted = 0;
  posX->trevsorted = 0;
  posY->tsorted = 0;
  posY->trevsorted = 0;
  posZ->tsorted = 0;
  posZ->trevsorted = 0;
  velX->tsorted = 0;
  velX->trevsorted = 0;
  velY->tsorted = 0;
  velY->trevsorted = 0;
  velZ->tsorted = 0;
  velZ->trevsorted = 0;

#ifndef NDEBUG
  fprintf(stderr,"#File loaded in %d ms\t", GDKms() - time0);
#endif

  BATmode(id, PERSISTENT);
  BATmode(posX, PERSISTENT);
  BATmode(posY, PERSISTENT);
  BATmode(posZ, PERSISTENT);
  BATmode(velX, PERSISTENT);
  BATmode(velY, PERSISTENT);
  BATmode(velZ, PERSISTENT);

  store_funcs.append_col(m->session->tr, colid, id, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposX, posX, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposY, posY, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposZ, posZ, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelX, velX, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelY, velY, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelZ, velZ, TYPE_bat);

#ifndef NDEBUG
  fprintf(stderr,"#Total time %d ms\n", GDKms() - time0);
#endif

  BBPdecref(id->batCacheid, TRUE);
  BBPdecref(posX->batCacheid, TRUE);
  BBPdecref(posY->batCacheid, TRUE);
  BBPdecref(posZ->batCacheid, TRUE);
  BBPdecref(velX->batCacheid, TRUE);
  BBPdecref(velY->batCacheid, TRUE);
  BBPdecref(velZ->batCacheid, TRUE);

  BBPunfix(id->batCacheid);
  BBPunfix(posX->batCacheid);
  BBPunfix(posY->batCacheid);
  BBPunfix(posZ->batCacheid);
  BBPunfix(velX->batCacheid);
  BBPunfix(velY->batCacheid);
  BBPunfix(velZ->batCacheid);

  //Free structures
  if (posB)
    GDKfree(posB);
  if (velB)
    GDKfree(velB);
  if (idB)
    GDKfree(idB);

  if (stream)
    fclose(stream);

  return msg;
}

str gadgetLoadTableAll(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  mvc *m = NULL;
  sql_schema *sch;
  sql_table *gadget_fl, *gadget_tbl, *tbl = NULL;
  sql_column *col;
  str tname = *getArgReference_str(stk, pci, 1);
  str *fnames = NULL;
  str msg = MAL_SUCCEED;
  oid rid = oid_nil, frid = oid_nil;
  rids *rids = NULL;
  int fid, i = 0, j = 0, numFiles = 0, npartTotal = 0, retval = 0, table_id = 0, jump = 0; 
  size_t sz;
  FILE **streams = NULL;
  Header *hs = NULL;
  long index;
  char *posB = NULL, *velB=NULL, *idB=NULL;
  sql_column *colid, *colposX, *colposY, *colposZ, *colvelX, *colvelY, *colvelZ; 
  BAT *id, *posX, *posY, *posZ, *velX, *velY, *velZ;
  lng *pid;
  flt *pposX, *pposY, *pposZ, *pvelX, *pvelY, *pvelZ;

#ifndef NDEBUG
  int time0;
#endif
  (void) retval;

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    return msg;
  sch = mvc_bind_schema(m, "sys");

  gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
  if (gadget_tbl == NULL) {
    msg = createException(MAL, "gadget.loadtable", "GADGET catalog is missing.\n");
    return msg;
  }

  tbl = mvc_bind_table(m, sch, tname);
  if (tbl == NULL) {
    msg = createException(MAL, "gadget.loadtable", "Could not find table %s.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, tbl, "id");
  sz = store_funcs.count_col(m->session->tr, col, 1);
  if (sz != 0) {
    msg = createException(MAL, "gadget.loadtable", "Table %s is not empty.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, gadget_tbl, "name");

  /*Get all the rids referring to this table name*/
  rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);

  col = mvc_bind_column(m, gadget_tbl, "id");
  table_id = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

  rids = table_funcs.rids_select(m->session->tr, col, &table_id, &table_id, NULL);

  numFiles = BATcount((BAT *) rids->data);
  fnames = (str*) GDKmalloc(sizeof(str) * numFiles);
  streams = (FILE**) GDKmalloc(sizeof(FILE*) * numFiles);
  hs = (Header*) GDKmalloc(sizeof(Header) * numFiles);
  while ( (rid = table_funcs.rids_next(rids)) != oid_nil) {

    /* Open GADGET file */
    col = mvc_bind_column(m, gadget_tbl, "file_id");
    fid = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

    gadget_fl = mvc_bind_table(m, sch, "gadget_files");
    col = mvc_bind_column(m, gadget_fl, "id");
    frid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
    col = mvc_bind_column(m, gadget_fl, "name");
    fnames[i] = (char *)table_funcs.column_find_value(m->session->tr, col, frid);

    /*Open Gadget File*/
    streams[i] = fopen(fnames[i], "rb");

    if (!streams[i]) {
      msg = createException(MAL, "ReadGadgetFile", "Missing GADGET file %s: %s", fnames[i], strerror(errno));
      goto out;
    }

    /*Get the header*/
    headerInit(&hs[i], streams[i]);
    npartTotal += hs[i].npartTotal;
    i++;
  }

  /*data load*/
#ifndef NDEBUG
  fprintf(stderr,"#Loading %d rows in table %s\n", npartTotal, tname);
  time0 = GDKms();
#endif

  colid = mvc_bind_column(m, tbl, "id"); 
  colposX = mvc_bind_column(m, tbl, "posx"); 
  colposY = mvc_bind_column(m, tbl, "posy"); 
  colposZ = mvc_bind_column(m, tbl, "posz"); 
  colvelX = mvc_bind_column(m, tbl, "velx"); 
  colvelY = mvc_bind_column(m, tbl, "vely"); 
  colvelZ = mvc_bind_column(m, tbl, "velz"); 

  id = COLnew(0, TYPE_lng, npartTotal, PERSISTENT); 
  posX = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  posY = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  posZ = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  velX = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  velY = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  velZ = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 

  if ( id == NULL || posX == NULL || posY == NULL || posZ == NULL || velX == NULL || velY == NULL || velZ == NULL) { 
    msg = createException(MAL, "gadget.gadgetload", "Malloc failed"); 
    goto out;
  } 

  pid = (lng *) Tloc(id, 0);
  pposX = (flt *) Tloc(posX, 0);
  pposY = (flt *) Tloc(posY, 0);
  pposZ = (flt *) Tloc(posZ, 0);
  pvelX = (flt *) Tloc(velX, 0);
  pvelY = (flt *) Tloc(velY, 0);
  pvelZ = (flt *) Tloc(velZ, 0);

  for (i = 0; i < numFiles; i++) {
    posB = (char*)GDKmalloc(sizeof(char) * 12 * hs[i].npartTotal);
    velB = (char*)GDKmalloc(sizeof(char) * 12 * hs[i].npartTotal);
    idB = (char*)GDKmalloc(sizeof(char) * 8 * hs[i].npartTotal);

    index = MRSNAP_POS_OFFSET;

    //NOTE: Convert C# blob.Seek to fseek
    //stream.Seek(index, SeekOrigin.Begin);
    fseek(streams[i], index, SEEK_SET);

    //NOTE: Convert C# blob.Read to fread
    //int retval = stream.Read(pos, 0, 12 * header.npartTotal);
    retval = fread(posB, 12 * hs[i].npartTotal, 1, streams[i]);
    assert(retval);

    index = MRSNAP_VEL_OFFSET + hs[i].npartTotal * 12; // skip over positions

    //NOTE: Convert C# blob.Seek to fseek
    //stream.Seek(index, SeekOrigin.Begin);
    fseek(streams[i], index, SEEK_SET);

    //NOTE: Convert C# blob.Read to fread
    //retval = stream.Read(vel, 0, 12 * header.npartTotal);
    retval = fread(velB, 12 * hs[i].npartTotal, 1, streams[i]);
    assert(retval);

    index = MRSNAP_ID_OFFSET + hs[i].npartTotal * 24; // skip over positions and velocities
    //NOTE: Convert C# blob.Seek to fseek
    //stream.Seek(index, SeekOrigin.Begin);
    fseek(streams[i], index, SEEK_SET);

    //NOTE: Convert C# blob.Read to fread
    //retval = stream.Read(id, 0, 8 * header.npartTotal);
    retval = fread(idB, 8 * hs[i].npartTotal, 1, streams[i]);
    assert(retval);

    for (j = 0; j < hs[i].npartTotal; j++)
      {
        int k = j * 12;
        pid[jump + j] = *(int64_t*)(&idB[j*8]);
        pposX[jump + j] = *(float*)(&posB[k]);
        pposY[jump + j] = *(float*)(&posB[k+4]);
        pposZ[jump + j] = *(float*)(&posB[k+8]);
        pvelX[jump + j] = *(float*)(&velB[k]);
        pvelY[jump + j] = *(float*)(&velB[k+4]);
        pvelZ[jump + j] = *(float*)(&velB[k+8]);
      }
    jump += hs[i].npartTotal;
  }

  BATsetcount(id, npartTotal);
  BATsetcount(posX, npartTotal);
  BATsetcount(posY, npartTotal);
  BATsetcount(posZ, npartTotal);
  BATsetcount(velX, npartTotal);
  BATsetcount(velY, npartTotal);
  BATsetcount(velZ, npartTotal);

  id->tsorted = 0;
  id->trevsorted = 0;
  posX->tsorted = 0;
  posX->trevsorted = 0;
  posY->tsorted = 0;
  posY->trevsorted = 0;
  posZ->tsorted = 0;
  posZ->trevsorted = 0;
  velX->tsorted = 0;
  velX->trevsorted = 0;
  velY->tsorted = 0;
  velY->trevsorted = 0;
  velZ->tsorted = 0;
  velZ->trevsorted = 0;

#ifndef NDEBUG
  fprintf(stderr,"#File loaded in %d ms\t", GDKms() - time0);
#endif

  BATmode(id, PERSISTENT);
  BATmode(posX, PERSISTENT);
  BATmode(posY, PERSISTENT);
  BATmode(posZ, PERSISTENT);
  BATmode(velX, PERSISTENT);
  BATmode(velY, PERSISTENT);
  BATmode(velZ, PERSISTENT);

  store_funcs.append_col(m->session->tr, colid, id, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposX, posX, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposY, posY, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposZ, posZ, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelX, velX, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelY, velY, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelZ, velZ, TYPE_bat);

#ifndef NDEBUG
  fprintf(stderr,"#Total time %d ms\n", GDKms() - time0);
#endif

  BBPdecref(id->batCacheid, TRUE);
  BBPdecref(posX->batCacheid, TRUE);
  BBPdecref(posY->batCacheid, TRUE);
  BBPdecref(posZ->batCacheid, TRUE);
  BBPdecref(velX->batCacheid, TRUE);
  BBPdecref(velY->batCacheid, TRUE);
  BBPdecref(velZ->batCacheid, TRUE);

  BBPunfix(id->batCacheid);
  BBPunfix(posX->batCacheid);
  BBPunfix(posY->batCacheid);
  BBPunfix(posZ->batCacheid);
  BBPunfix(velX->batCacheid);
  BBPunfix(velY->batCacheid);
  BBPunfix(velZ->batCacheid);

  //Free structures
 out:
  if (fnames)
    GDKfree(fnames);
  if (streams) {
    for (i = 0; i < numFiles; i++) {
      if(streams[i])
        fclose(streams[i]);
      streams[i] = NULL;
    }
    GDKfree(streams);
  }
  if (hs)
    GDKfree(hs);
  if (posB)
    GDKfree(posB);
  if (velB)
    GDKfree(velB);
  if (idB)
    GDKfree(idB);

  return msg;
}

static str
gadgetLoadTableAll_(mvc *m, sql_schema *sch, sql_table *gadget_tbl, char *tname)
{
  sql_table *gadget_fl = NULL, *tbl = NULL;
  sql_column *col;
  str *fnames = NULL;
  str msg = MAL_SUCCEED;
  oid rid = oid_nil, frid = oid_nil;
  rids *rids = NULL;
  int fid, *snapnums = NULL, i = 0, j = 0, numFiles = 0, npartTotal = 0, retval = 0, table_id = 0, jump = 0; 
  size_t sz;
  FILE **streams = NULL;
  Header *hs = NULL;
  long index;
  char *posB = NULL, *velB=NULL, *idB=NULL;
  sql_column *colid, *colposX, *colposY, *colposZ, *colvelX, *colvelY, *colvelZ, *colsnapnum, *colphkey; 
  BAT *id, *posX, *posY, *posZ, *velX, *velY, *velZ, *snapnum, *phkey;
  lng *pid;
  flt *pposX, *pposY, *pposZ, *pvelX, *pvelY, *pvelZ;
  int *psnapnum, *pphkey;

#ifndef NDEBUG
  int time0;
#endif
  (void) retval;

  GADGET_LOCK;
  tbl = mvc_bind_table(m, sch, tname);
  if (tbl == NULL) {
    msg = createException(MAL, "gadget.loadtable", "Could not find table %s.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, tbl, "id");
  sz = store_funcs.count_col(m->session->tr, col, 1);
  if (sz != 0) {
    msg = createException(MAL, "gadget.loadtable", "Table %s is not empty.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, gadget_tbl, "name");

  /*Get all the rids referring to this table name*/
  rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);

  col = mvc_bind_column(m, gadget_tbl, "id");
  table_id = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

  rids = table_funcs.rids_select(m->session->tr, col, &table_id, &table_id, NULL);

  numFiles = BATcount((BAT *) rids->data);
  fnames = (str*) GDKmalloc(sizeof(str) * numFiles);
  streams = (FILE**) GDKmalloc(sizeof(FILE*) * numFiles);
  hs = (Header*) GDKmalloc(sizeof(Header) * numFiles);
  snapnums = (int*) GDKzalloc(sizeof(int) * numFiles);
  while ( (rid = table_funcs.rids_next(rids)) != oid_nil) {
    char *snapnum_str, *ptr = NULL;

    /* Open GADGET file */
    col = mvc_bind_column(m, gadget_tbl, "file_id");
    fid = *(int*)table_funcs.column_find_value(m->session->tr, col, rid);

    gadget_fl = mvc_bind_table(m, sch, "gadget_files");
    col = mvc_bind_column(m, gadget_fl, "id");
    frid = table_funcs.column_find_row(m->session->tr, col, (void *)&fid, NULL);
    col = mvc_bind_column(m, gadget_fl, "name");
    fnames[i] = (char *)table_funcs.column_find_value(m->session->tr, col, frid);

    /*Open Gadget File*/
    streams[i] = fopen(fnames[i], "rb");

    if (!streams[i]) {
      msg = createException(MAL, "ReadGadgetFile", "Missing GADGET file %s: %s", fnames[i], strerror(errno));
      goto out;
    }

    /*Get snapnum from the file name*/
    ptr = strrchr(fnames[i], '_');
    ptr = ptr+1;
    snapnum_str = GDKstrdup(ptr);
    snapnum_str[3] = '\0';
    snapnums[i] = atoi(snapnum_str);
    GDKfree(snapnum_str);

    /*Get the header*/
    headerInit(&hs[i], streams[i]);
    npartTotal += hs[i].npartTotal;
    i++;
  }
  GADGET_UNLOCK;

  /*data load*/
#ifndef NDEBUG
  fprintf(stderr,"#Loading %d rows in table %s\n", npartTotal, tname);
  time0 = GDKms();
#endif

  colid = mvc_bind_column(m, tbl, "id"); 
  colposX = mvc_bind_column(m, tbl, "posx"); 
  colposY = mvc_bind_column(m, tbl, "posy"); 
  colposZ = mvc_bind_column(m, tbl, "posz"); 
  colvelX = mvc_bind_column(m, tbl, "velx"); 
  colvelY = mvc_bind_column(m, tbl, "vely"); 
  colvelZ = mvc_bind_column(m, tbl, "velz"); 
  colsnapnum = mvc_bind_column(m, tbl, "snapnum"); 
  colphkey = mvc_bind_column(m, tbl, "phkey"); 

  id = COLnew(0, TYPE_lng, npartTotal, PERSISTENT); 
  posX = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  posY = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  posZ = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  velX = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  velY = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  velZ = COLnew(0, TYPE_flt, npartTotal, PERSISTENT); 
  snapnum = COLnew(0, TYPE_int, npartTotal, PERSISTENT); 
  phkey = COLnew(0, TYPE_int, npartTotal, PERSISTENT); 

  if ( id == NULL || posX == NULL || posY == NULL || posZ == NULL || velX == NULL || velY == NULL || velZ == NULL || snapnum == NULL || phkey == NULL) { 
    msg = createException(MAL, "gadget.gadgetload", "Malloc failed"); 
    goto out;
  } 

  pid = (lng *) Tloc(id, 0);
  pposX = (flt *) Tloc(posX, 0);
  pposY = (flt *) Tloc(posY, 0);
  pposZ = (flt *) Tloc(posZ, 0);
  pvelX = (flt *) Tloc(velX, 0);
  pvelY = (flt *) Tloc(velY, 0);
  pvelZ = (flt *) Tloc(velZ, 0);
  psnapnum = (int *) Tloc(snapnum, 0);
  pphkey = (int *) Tloc(phkey, 0);

  for (i = 0; i < numFiles; i++) {
    int m = 0;
    PHBins bins;
    posB = (char*)GDKmalloc(sizeof(char) * 12 * hs[i].npartTotal);
    velB = (char*)GDKmalloc(sizeof(char) * 12 * hs[i].npartTotal);
    idB = (char*)GDKmalloc(sizeof(char) * 8 * hs[i].npartTotal);

    index = MRSNAP_POS_OFFSET;

    //NOTE: Convert C# blob.Seek to fseek
    //stream.Seek(index, SeekOrigin.Begin);
    fseek(streams[i], index, SEEK_SET);

    //NOTE: Convert C# blob.Read to fread
    //int retval = stream.Read(pos, 0, 12 * header.npartTotal);
    retval = fread(posB, 12 * hs[i].npartTotal, 1, streams[i]);
    assert(retval);

    index = MRSNAP_VEL_OFFSET + hs[i].npartTotal * 12; // skip over positions

    //NOTE: Convert C# blob.Seek to fseek
    //stream.Seek(index, SeekOrigin.Begin);
    fseek(streams[i], index, SEEK_SET);

    //NOTE: Convert C# blob.Read to fread
    //retval = stream.Read(vel, 0, 12 * header.npartTotal);
    retval = fread(velB, 12 * hs[i].npartTotal, 1, streams[i]);
    assert(retval);

    index = MRSNAP_ID_OFFSET + hs[i].npartTotal * 24; // skip over positions and velocities
    //NOTE: Convert C# blob.Seek to fseek
    //stream.Seek(index, SeekOrigin.Begin);
    fseek(streams[i], index, SEEK_SET);

    //NOTE: Convert C# blob.Read to fread
    //retval = stream.Read(id, 0, 8 * header.npartTotal);
    retval = fread(idB, 8 * hs[i].npartTotal, 1, streams[i]);
    assert(retval);

    /*Check if there are PHBins, if yes collect them*/
    if (hs[i].hashtabsize > 0)
      {
        long index = 0, retval = 0, nbins = hs[i].last_cell - hs[i].first_cell + 1;
        char* binbytes = NULL;
        PHBin *prev = NULL;

        if (PHBins_init(&bins)) {
          fprintf(stderr, "GadgetPHBins: PHBinList_init failed!!!\n");
          goto out;
        }

        binbytes = (char*) malloc(sizeof(char) * nbins * 4);
        if (!binbytes) {
          PHBins_destroy(&bins);
          fprintf(stderr, "GadgetPHBins: Malloc of binbytes failed!!!\n");
          goto out;
        }

        index = 308 + hs[i].npartTotal * 32; 
        fseek(streams[i], index, SEEK_SET);

        retval = fread(binbytes, nbins * 4, 1, streams[i]);
#ifndef NDEBUG
        assert(retval);
#else
        (void) retval;
#endif

        for (j = 0; j < nbins; j++)
          {
            PHBin *bin = (PHBin *) malloc (sizeof(PHBin));
            bin->id = hs[i].first_cell + j;
            bin->start = *(int*) (&binbytes[j * 4]);
            if (prev != NULL)
              {
                prev->count = bin->start - prev->start;
                PHBins_add(&bins, prev->id, prev->start, prev->count);
              }
            if (prev)
              free(prev);
            prev = bin;
          }
        prev->count = hs[i].npartTotal - prev->start;
        PHBins_add(&bins, prev->id, prev->start, prev->count);
        if (prev)
          free(prev);
        if (binbytes)
          free(binbytes);
      }


    for (j = 0; j < hs[i].npartTotal; j++)
      {
        int k = j * 12;
        pid[jump + j] = *(int64_t*)(&idB[j*8]);
        pposX[jump + j] = *(float*)(&posB[k]);
        pposY[jump + j] = *(float*)(&posB[k+4]);
        pposZ[jump + j] = *(float*)(&posB[k+8]);
        pvelX[jump + j] = *(float*)(&velB[k]);
        pvelY[jump + j] = *(float*)(&velB[k+4]);
        pvelZ[jump + j] = *(float*)(&velB[k+8]);
        psnapnum[jump + j] = snapnums[i];
        if (j == (bins.start[m] + bins.count[m]))
          m++;
        pphkey[jump + j] = bins.id[m]; 
      }

    PHBins_destroy(&bins);
    jump += hs[i].npartTotal;
  }

  BATsetcount(id, npartTotal);
  BATsetcount(posX, npartTotal);
  BATsetcount(posY, npartTotal);
  BATsetcount(posZ, npartTotal);
  BATsetcount(velX, npartTotal);
  BATsetcount(velY, npartTotal);
  BATsetcount(velZ, npartTotal);
  BATsetcount(snapnum, npartTotal);
  BATsetcount(phkey, npartTotal);

  id->tsorted = 0;
  id->trevsorted = 0;
  posX->tsorted = 0;
  posX->trevsorted = 0;
  posY->tsorted = 0;
  posY->trevsorted = 0;
  posZ->tsorted = 0;
  posZ->trevsorted = 0;
  velX->tsorted = 0;
  velX->trevsorted = 0;
  velY->tsorted = 0;
  velY->trevsorted = 0;
  velZ->tsorted = 0;
  velZ->trevsorted = 0;
  snapnum->tsorted = 0;
  snapnum->trevsorted = 0;
  phkey->tsorted = 0;
  phkey->trevsorted = 0;

#ifndef NDEBUG
  fprintf(stderr,"#File loaded in %d ms\t", GDKms() - time0);
#endif

  BATmode(id, PERSISTENT);
  BATmode(posX, PERSISTENT);
  BATmode(posY, PERSISTENT);
  BATmode(posZ, PERSISTENT);
  BATmode(velX, PERSISTENT);
  BATmode(velY, PERSISTENT);
  BATmode(velZ, PERSISTENT);
  BATmode(snapnum, PERSISTENT);
  BATmode(phkey, PERSISTENT);

  store_funcs.append_col(m->session->tr, colid, id, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposX, posX, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposY, posY, TYPE_bat);
  store_funcs.append_col(m->session->tr, colposZ, posZ, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelX, velX, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelY, velY, TYPE_bat);
  store_funcs.append_col(m->session->tr, colvelZ, velZ, TYPE_bat);
  store_funcs.append_col(m->session->tr, colsnapnum, snapnum, TYPE_bat);
  store_funcs.append_col(m->session->tr, colphkey, phkey, TYPE_bat);

#ifndef NDEBUG
  fprintf(stderr,"#Total time %d ms\n", GDKms() - time0);
#endif

  BBPdecref(id->batCacheid, TRUE);
  BBPdecref(posX->batCacheid, TRUE);
  BBPdecref(posY->batCacheid, TRUE);
  BBPdecref(posZ->batCacheid, TRUE);
  BBPdecref(velX->batCacheid, TRUE);
  BBPdecref(velY->batCacheid, TRUE);
  BBPdecref(velZ->batCacheid, TRUE);
  BBPdecref(snapnum->batCacheid, TRUE);
  BBPdecref(phkey->batCacheid, TRUE);

  BBPunfix(id->batCacheid);
  BBPunfix(posX->batCacheid);
  BBPunfix(posY->batCacheid);
  BBPunfix(posZ->batCacheid);
  BBPunfix(velX->batCacheid);
  BBPunfix(velY->batCacheid);
  BBPunfix(velZ->batCacheid);
  BBPunfix(snapnum->batCacheid);
  BBPunfix(phkey->batCacheid);

  //Free structures
 out:
  if (fnames)
    GDKfree(fnames);
  if (streams) {
    for (i = 0; i < numFiles; i++) {
      if(streams[i])
        fclose(streams[i]);
      streams[i] = NULL;
    }
    GDKfree(streams);
  }
  if (hs)
    GDKfree(hs);
  if (posB)
    GDKfree(posB);
  if (velB)
    GDKfree(velB);
  if (idB)
    GDKfree(idB);
  if (snapnums)
    GDKfree(snapnums);

  return msg;
}

str
gadgetCheckTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  mvc *m = NULL;
  str msg = MAL_SUCCEED;
  size_t sz;
  sql_schema *sch = NULL;
  sql_table *gadget_tbl, *tbl = NULL;
  sql_column *col;
  oid rid = oid_nil;
  str tname = *getArgReference_str(stk, pci, 3);
  int *res = getArgReference_int(stk, pci, 1);

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != MAL_SUCCEED)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != MAL_SUCCEED)
    return msg;

  sch = mvc_bind_schema(m, "sys");
  gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
  if (gadget_tbl == NULL) {
    msg = createException(MAL, "gadget.check", "GADGET catalog is missing.\n");
    return msg;
  }

  /*Check if is a table which belongs to gadget_tables*/
  col = mvc_bind_column(m, gadget_tbl, "name");
  rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
  if (rid == oid_nil) {
    return MAL_SUCCEED;
  }

  tbl = mvc_bind_table(m, sch, tname);
  if (tbl == NULL) {
    msg = createException(MAL, "gadget.check", "Could not find table %s.\n", tname);
    return msg;
  }

  col = mvc_bind_column(m, tbl, "id");
  sz = store_funcs.count_col(m->session->tr, col, 1);

  if (sz == 0) {
    /*Lets load the table*/
    msg = gadgetLoadTableAll_(m, sch, gadget_tbl, tname);
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
gadgetAnalyzeTable(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
  mvc *m = NULL;
  str msg = MAL_SUCCEED;
  char readonly[BUFSIZ], analyze[BUFSIZ];
  char *sr, *sa;
  sql_schema *sch = NULL;
  sql_table *gadget_tbl;
  sql_column *col;
  oid rid = oid_nil;
  int status, *res;
  str tname = NULL;

  if (pci->argc == 4) {
      status = *getArgReference_int(stk, pci, 2);
      tname = *getArgReference_str(stk, pci, 3);
      res = getArgReference_int(stk, pci, 0);
  } else {
      msg = createException(MAL, "gadget.analyze", "incorrect number of arguments.\n");
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
  gadget_tbl = mvc_bind_table(m, sch, "gadget_tables");
  if (gadget_tbl == NULL) {
    msg = createException(MAL, "gadget.analyze", "GADGET catalog is missing.\n");
    return msg;
  }

  /*Check if is a table which belongs to gadget_tables*/
  col = mvc_bind_column(m, gadget_tbl, "name");
  rid = table_funcs.column_find_row(m->session->tr, col, tname, NULL);
  if (rid == oid_nil) {
    msg = createException(MAL, "gadget.analyze", "Table %s is unknown to the GADGET catalog. Attach first the containing file\n", tname);
    return msg;
  }


  /*Set table read only*/
  sr = readonly;
  snprintf(readonly, BUFSIZ, "alter table %s set read only;", tname);

#ifndef NDEBUG
  fprintf(stderr, "The readonly stmt is: %s!!!\n", readonly);
#endif
  GADGET_LOCK;
  msg = SQLstatementIntern(cntxt, &sr, "gadget.analyze", TRUE, FALSE, NULL);
  GADGET_UNLOCK;
  if (msg)
    return msg;

  /*Analyze table*/
  sa = analyze;
  snprintf(analyze, BUFSIZ, "analyze sys.%s (id, posX, posY, posZ) minmax;", tname);

#ifndef NDEBUG
  fprintf(stderr, "The analyze stmt is: %s!!!\n", analyze);
#endif
  GADGET_LOCK;
  msg = SQLstatementIntern(cntxt, &sa, "gadget.analyze", TRUE, FALSE, NULL);
  GADGET_UNLOCK;
  if (msg)
    return msg;
    
  *res = VAULT_TABLE_DONE;

  return MAL_SUCCEED;
}

str
gadgetPHkeyConvert(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
  lng *phkey = getArgReference_lng(stk, pci, 0);
  float x = *getArgReference_flt(stk, pci, 1);
  float y = *getArgReference_flt(stk, pci, 2);
  float z = *getArgReference_flt(stk, pci, 3);
  float boxSize = *getArgReference_flt(stk, pci, 4);
  int bits = *getArgReference_int(stk, pci, 5);
  (void) mb;
  (void) cntxt;
    
  *phkey = peano_hilbert_key(x, y, z, boxSize, bits);

  return MAL_SUCCEED;
}

str
gadgetPHkeyInvert(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
  BAT *x, *y, *z, *cSize;
  float posX, posY, posZ, cellSize;
  mvc *m = NULL;
  str msg;

  /*Get the bats for the table*/
  bat *rx = getArgReference_bat(stk, pci, 0);
  bat *ry = getArgReference_bat(stk, pci, 1);
  bat *rz = getArgReference_bat(stk, pci, 2);
  bat *rcSize = getArgReference_bat(stk, pci, 3);

  /*Get input args*/
  peanokey phkey = *getArgReference_lng(stk, pci, 4);
  float boxSize = *getArgReference_flt(stk, pci, 5);
  int bits = *getArgReference_int(stk, pci, 6);

  if ((msg = getSQLContext(cntxt, mb, &m, NULL)) != NULL)
    return msg;
  if ((msg = checkSQLContext(cntxt)) != NULL)
    return msg;

  x = COLnew(0, TYPE_flt, 0, TRANSIENT);
  y = COLnew(0, TYPE_flt, 0, TRANSIENT);
  z = COLnew(0, TYPE_flt, 0, TRANSIENT);
  cSize = COLnew(0, TYPE_flt, 0, TRANSIENT);

  if (x == NULL || y == NULL || z == NULL || cSize == NULL) {
    if (x)
      BBPunfix(x->batCacheid);
    if (y)
      BBPunfix(y->batCacheid);
    if (z)
      BBPunfix(z->batCacheid);
              
    throw(SQL, "sql.storage", MAL_MALLOC_FAIL);
  }

  peano_hilbert_inverse_key(&posX, &posY, &posZ, phkey, boxSize, bits);
  cellSize = boxSize / (1 << bits);

  BUNappend(x, &posX, FALSE);
  BUNappend(y, &posY, FALSE);
  BUNappend(z, &posZ, FALSE);
  BUNappend(cSize, &cellSize, FALSE);

  BBPkeepref(*rx = x->batCacheid);
  BBPkeepref(*ry = y->batCacheid);
  BBPkeepref(*rz = z->batCacheid);
  BBPkeepref(*rcSize = cSize->batCacheid);

  return MAL_SUCCEED;
}
