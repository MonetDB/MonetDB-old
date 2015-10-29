/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * H. Muehleisen, M. Raasveldt
 * Inverse RAPI
 */

/*
 * Taken from the embedded branch tools/embedded/embedded.h
 * Stripped of the R-specific functions
 */
#include "embedded.h"

#include "monetdb_config.h"
#include "monet_options.h"
#include "mal.h"
#include "mal_client.h"
#include "mal_linker.h"
#include "msabaoth.h"
#include "sql_scenario.h"
#include "gdk_utils.h"

typedef str (*SQLstatementIntern_ptr_tpe)(Client, str*, str, bit, bit, res_table**);
SQLstatementIntern_ptr_tpe SQLstatementIntern_ptr = NULL;
typedef str (*SQLautocommit_ptr_tpe)(Client, mvc*);
SQLautocommit_ptr_tpe SQLautocommit_ptr = NULL;
typedef str (*SQLinitClient_ptr_tpe)(Client);
SQLinitClient_ptr_tpe SQLinitClient_ptr = NULL;
typedef str (*getSQLContext_ptr_tpe)(Client, MalBlkPtr, mvc**, backend**);
getSQLContext_ptr_tpe getSQLContext_ptr = NULL;
typedef void (*res_table_destroy_ptr_tpe)(res_table *t);
res_table_destroy_ptr_tpe res_table_destroy_ptr = NULL;
typedef str (*mvc_append_wrap_ptr_tpe)(Client, MalBlkPtr, MalStkPtr, InstrPtr);
mvc_append_wrap_ptr_tpe mvc_append_wrap_ptr = NULL;
typedef sql_schema* (*mvc_bind_schema_ptr_tpe)(mvc*, const char*);
mvc_bind_schema_ptr_tpe mvc_bind_schema_ptr = NULL;
typedef sql_table* (*mvc_bind_table_ptr_tpe)(mvc*, sql_schema*, const char*);
mvc_bind_table_ptr_tpe mvc_bind_table_ptr = NULL;
typedef int (*sqlcleanup_ptr_tpe)(mvc*, int);
sqlcleanup_ptr_tpe sqlcleanup_ptr = NULL;
typedef void (*mvc_trans_ptr_tpe)(mvc*);
mvc_trans_ptr_tpe mvc_trans_ptr = NULL;

static bit monetdb_embedded_initialized = 0;
static MT_Lock monetdb_embedded_lock;

static void* lookup_function(char* func) {
	void *dl, *fun;
	dl = mdlopen("libmonetdb5", RTLD_NOW | RTLD_GLOBAL);
	if (dl == NULL) {
		return NULL;
	}
	fun = dlsym(dl, func);
	dlclose(dl);
	return fun;
}

char* monetdb_startup(char* installdir, char* dbdir, char silent) {
	opt *set = NULL;
	int setlen = 0;
	char* retval = NULL;
	char* sqres = NULL;
	void* res = NULL;
	char mod_path[1000];
	GDKfataljumpenable = 1;
	if(setjmp(GDKfataljump) != 0) {
		retval = GDKfatalmsg;
		// we will get here if GDKfatal was called.
		if (retval != NULL) {
			retval = GDKstrdup("GDKfatal() with unspecified error?");
		}
		goto cleanup;
	}
	MT_lock_init(&monetdb_embedded_lock, "monetdb_embedded_lock");
	MT_lock_set(&monetdb_embedded_lock, "monetdb.startup");
	if (monetdb_embedded_initialized) goto cleanup;

	setlen = mo_builtin_settings(&set);
	setlen = mo_add_option(&set, setlen, opt_cmdline, "gdk_dbpath", dbdir);
	BBPaddfarm(dbdir, (1 << PERSISTENT) | (1 << TRANSIENT));

	if (GDKinit(set, setlen) == 0) {
		retval = GDKstrdup("GDKinit() failed");
		goto cleanup;
	}
	snprintf(mod_path, 1000, "%s/lib/monetdb5", installdir);
	GDKsetenv("monet_mod_path", mod_path);
	GDKsetenv("mapi_disable", "true");
	GDKsetenv("max_clients", "0");
	// TODO: SELECT * FROM table should not use mitosis in the first place (?).
	GDKsetenv("sql_optimizer", "sequential_pipe");
	printf("Starting 1.1\n");
	if (silent) THRdata[0] = stream_blackhole_create();
	msab_dbpathinit(dbdir);
	printf("Starting 1.1.1\n");

	if (mal_init() != 0) { // mal_init() does not return meaningful codes on failure
		printf("Starting 1.1 fail\n");
		retval = GDKstrdup("mal_init() failed");
		goto cleanup;
	}
	printf("Starting 1.2\n");
	if (silent) mal_clients[0].fdout = THRdata[0];
	// This dynamically looks up functions, because the libraries containing them are loaded at runtime.
	SQLstatementIntern_ptr = (SQLstatementIntern_ptr_tpe) lookup_function("SQLstatementIntern");
	SQLautocommit_ptr      = (SQLautocommit_ptr_tpe)      lookup_function("SQLautocommit");
	SQLinitClient_ptr      = (SQLinitClient_ptr_tpe)      lookup_function("SQLinitClient");
	getSQLContext_ptr      = (getSQLContext_ptr_tpe)      lookup_function("getSQLContext");
	res_table_destroy_ptr  = (res_table_destroy_ptr_tpe)  lookup_function("res_table_destroy");
	mvc_append_wrap_ptr    = (mvc_append_wrap_ptr_tpe)    lookup_function("mvc_append_wrap");
	mvc_bind_schema_ptr    = (mvc_bind_schema_ptr_tpe)    lookup_function("mvc_bind_schema");
	mvc_bind_table_ptr     = (mvc_bind_table_ptr_tpe)     lookup_function("mvc_bind_table");
	sqlcleanup_ptr         = (sqlcleanup_ptr_tpe)         lookup_function("sqlcleanup");
	mvc_trans_ptr          = (mvc_trans_ptr_tpe)          lookup_function("mvc_trans");
	printf("Starting 1.3\n");
	if (SQLstatementIntern_ptr == NULL || SQLautocommit_ptr == NULL ||
			SQLinitClient_ptr == NULL || getSQLContext_ptr == NULL ||
			res_table_destroy_ptr == NULL || mvc_append_wrap_ptr == NULL ||
			mvc_bind_schema_ptr == NULL || mvc_bind_table_ptr == NULL ||
			sqlcleanup_ptr == NULL || mvc_trans_ptr == NULL) {
		retval = GDKstrdup("Dynamic function lookup failed");
		goto cleanup;
	}
	// call this, otherwise c->sqlcontext is empty
	(*SQLinitClient_ptr)(&mal_clients[0]);
	((backend *) mal_clients[0].sqlcontext)->mvc->session->auto_commit = 1;
	monetdb_embedded_initialized = true;
	// we do not want to jump after this point, since we cannot do so between threads
	GDKfataljumpenable = 0;
	printf("Starting 1.4\n");
	// sanity check, run a SQL query
	sqres = monetdb_query("SELECT * FROM tables;", res);
	if (sqres != NULL) {
		monetdb_embedded_initialized = false;
		retval = sqres;
		goto cleanup;
	}
cleanup:
	printf("Starting 1. cleanup\n");
	mo_free_options(set, setlen);
	printf("Starting 1. cleanup2\n");
	MT_lock_unset(&monetdb_embedded_lock, "monetdb.startup");
	return retval;
}

char* monetdb_query(char* query, void** result) {
	str res = MAL_SUCCEED;
	Client c = &mal_clients[0];
	mvc* m = ((backend *) c->sqlcontext)->mvc;
	if (!monetdb_embedded_initialized) {
		return GDKstrdup("Embedded MonetDB is not started");
	}

	while (*query == ' ' || *query == '\t') query++;
	if (strncasecmp(query, "START", 5) == 0) { // START TRANSACTION
		m->session->auto_commit = 0;
	}
	else if (strncasecmp(query, "ROLLBACK", 8) == 0) {
		m->session->status = -1;
		m->session->auto_commit = 1;
	}
	else if (strncasecmp(query, "COMMIT", 6) == 0) {
		m->session->auto_commit = 1;
	}
	else if (strncasecmp(query, "SHIBBOLEET", 10) == 0) {
		res = GDKstrdup("\x46\x6f\x72\x20\x69\x6d\x6d\x65\x64\x69\x61\x74\x65\x20\x74\x65\x63\x68\x6e\x69\x63\x61\x6c\x20\x73\x75\x70\x70\x6f\x72\x74\x20\x63\x61\x6c\x6c\x20\x2b\x33\x31\x20\x32\x30\x20\x35\x39\x32\x20\x34\x30\x33\x39");
	}
	else if (m->session->status < 0 && m->session->auto_commit ==0){
		res = GDKstrdup("Current transaction is aborted (please ROLLBACK)");
	} else {
		res = (*SQLstatementIntern_ptr)(c, &query, "name", 1, 0, (res_table **) result);
	}

	(*SQLautocommit_ptr)(c, m);
	return res;
}

void monetdb_cleanup_result(void* output) {
	(*res_table_destroy_ptr)((res_table*) output);
}
