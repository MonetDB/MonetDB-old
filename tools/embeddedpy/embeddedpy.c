/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * @a M. Raasveldt
 * MonetDB embedded in Python library. Contains functions to initialize MonetDB and to perform SQL queries after it has been initialized.
 */
#include "embeddedpy.h"

// Numpy Library
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __INTEL_COMPILER
// Intel compiler complains about trailing comma's in numpy source code
#pragma warning(disable:271)
#endif
#include <numpy/arrayobject.h>

#include "monetdb_config.h"
#include "monet_options.h"
#include "mal.h"
#include "mal_client.h"
#include "mal_linker.h"
#include "msabaoth.h"
#include "sql_scenario.h"
#include "gdk_utils.h"

#include "pyapi.h"
#include "pytypes.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// copy paste from embedded.h
typedef struct append_data {
	char* colname;
	bat batid;
} append_data;

str monetdb_startup(char* dir, char silent);
str monetdb_query(char* query, void** result);
str monetdb_create_table(char *schema, char *table_name, append_data *ad, int ncols);
str monetdb_append(const char* schema, const char* table, append_data *ad, int ncols);
void monetdb_cleanup_result(void* output);
static str monetdb_get_columns(const char* schema_name, const char *table_name, int *column_count, char ***column_names, int **column_types) ;

static bit monetdb_embedded_initialized = 0;
static MT_Lock monetdb_embedded_lock;
/////////////////////////////////////////////////////////////////////////////////////////////////

PyObject *monetdb_init(PyObject *self, PyObject *args)
{
	(void) self;
	if (!PyString_CheckExact(args)) {
   		PyErr_SetString(PyExc_TypeError, "Expected a directory name as an argument.");
		return NULL;
	}

	{
		char *msg;
		char *directory = &(((PyStringObject*)args)->ob_sval[0]);
		printf("Making directory %s\n", directory);
		if (GDKcreatedir(directory) != GDK_SUCCEED) {
   			PyErr_Format(PyExc_Exception, "Failed to create directory %s.", directory);
   			return NULL;
		}
		msg = monetdb_startup(directory, 1);
		if (msg != MAL_SUCCEED) {
	   		PyErr_Format(PyExc_Exception, "Failed to initialize MonetDB. %s", msg);
			return NULL;
		}
		PyAPIprelude(NULL);
	}
	Py_RETURN_NONE;
}

PyObject *monetdb_sql(PyObject *self, PyObject *args)
{
	(void) self;
	if (!PyString_CheckExact(args)) {
   		PyErr_SetString(PyExc_TypeError, "Expected a SQL query as a single string argument.");
		return NULL;
	}
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}

	{
		PyObject *result;
		res_table* output = NULL;
		char* err;
		// Append ';'' to the SQL query, just in case
		args = PyString_FromFormat("%s;", &(((PyStringObject*)args)->ob_sval[0]));
		// Perform the SQL query
		err = monetdb_query(&(((PyStringObject*)args)->ob_sval[0]), (void**)&output);
		if (err != NULL) { 
	   		PyErr_Format(PyExc_Exception, "SQL Query Failed: %s", (err ? err : "<no error>"));
			return NULL;
		}
		// Construct a dictionary from the output columns (dict[name] = column)
		result = PyDict_New();
		if (output && output->nr_cols > 0) {
			PyInput input;
			PyObject *numpy_array;
			char *msg = NULL;
			int i;
			for (i = 0; i < output->nr_cols; i++) {
				res_col col = output->cols[i];
				BAT* b = BATdescriptor(col.b);

            	input.bat = b;
				input.count = BATcount(b);
            	input.bat_type = ATOMstorage(getColumnType(b->T->type));
            	input.scalar = false;
            	input.sql_subtype = &col.type;

            	numpy_array = PyMaskedArray_FromBAT(&mal_clients[0], &input, 0, input.count, &msg, true);
            	if (!numpy_array) {
					monetdb_cleanup_result(output);
			   		PyErr_Format(PyExc_Exception, "SQL Query Failed: %s", (msg ? msg : "<no error>"));
					return NULL;
            	}
            	PyDict_SetItem(result, PyString_FromString(output->cols[i].name), numpy_array);
			}
			monetdb_cleanup_result(output);
			return result;
		} else {
			Py_RETURN_NONE;
		}
	}
}

PyObject *monetdb_create(PyObject *self, PyObject *args)
{
	char *schema = "sys";
	char *table_name;
	PyObject *dict = NULL, *values = NULL;
	int i;
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "sO|O", &table_name, &dict, &values)) {
		return NULL;
	}
	if (values == NULL) {
		if (!PyDict_CheckExact(dict)) {
	   		PyErr_SetString(PyExc_TypeError, "the second argument has to be a dict");
			return NULL;
		}
   		PyErr_Format(PyExc_Exception, "dict is not implemented yet");
		return NULL;
		Py_RETURN_NONE;
	} else {
		if (!PyList_CheckExact(dict)) {
	   		PyErr_SetString(PyExc_TypeError, "the second argument (column names) must be a list");
			return NULL;
		}
		if (PyList_Size(dict) == 0) {
	   		PyErr_SetString(PyExc_TypeError, "the list of column names must be non-zero");
			return NULL;
		}

		for(i = 0; i < PyList_Size(dict); i++) {
			PyObject *column_name = PyList_GetItem(dict, i);
			if (!PyString_CheckExact(column_name)) {
		   		PyErr_Format(PyExc_TypeError, "the entry %d in the column names is not a string", i);
				return NULL;
			}
		}
		//convert the values to BATs and create the table
		{
			char *msg = NULL;
			PyObject *pResult;
			int columns = PyList_Size(dict);
			append_data *append_bats = NULL;
			PyReturn *pyreturn_values = NULL;

			pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
			if (pResult == NULL) goto cleanup;

			pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
			if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;


			append_bats = GDKzalloc(sizeof(append_data) * columns);
			for(i = 0; i < columns; i++) {
				append_bats[i].batid = int_nil;
			}
			for(i = 0; i < columns; i++) {
				BAT *b = PyObject_ConvertToBAT(&pyreturn_values[i], NULL, PyType_ToBat(pyreturn_values[i].result_type), i, 0, &msg, true);
				if (b == NULL) goto cleanup; 
				append_bats[i].batid = b->batCacheid;
				append_bats[i].colname = PyString_AS_STRING(PyList_GetItem(dict, i));
 			}
 			msg = monetdb_create_table(schema, table_name, append_bats, columns);
cleanup:
			if (pyreturn_values) GDKfree(pyreturn_values);
			if (append_bats) {
				for(i = 0; i < columns; i++) {
					if (append_bats[i].batid != int_nil) 
						BBPunfix(append_bats[i].batid);
				}
				GDKfree(append_bats);
			}
			if (msg != NULL) {
				PyErr_Format(PyExc_Exception, "%s", msg);
				return NULL;
			}
		}
	}
	Py_RETURN_NONE;
}

PyObject *monetdb_insert(PyObject *self, PyObject *args)
{
	char *schema_name = "sys";
	char *table_name;
	PyObject *values = NULL;
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "sO", &table_name, &values)) {
		return NULL;
	}
	{
		char *msg = NULL;
		PyObject *pResult;
		PyReturn *pyreturn_values = NULL;
		append_data *append_bats = NULL;
		int i;
		char **column_names = NULL;
		int *column_types = NULL;
		int columns = 0;

		msg = monetdb_get_columns(schema_name, table_name, &columns, &column_names, &column_types);

		pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
		if (pResult == NULL) goto cleanup;
		pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
		if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;

		append_bats = GDKzalloc(sizeof(append_bats) * columns);
		for(i = 0; i < columns; i++) {
			append_bats[i].batid = int_nil;
			append_bats[i].colname = column_names[i];
		}
		for(i = 0; i < columns; i++) {
			BAT *b = PyObject_ConvertToBAT(&pyreturn_values[i], NULL, column_types[i], i, 0, &msg, true);

			if (b == NULL) goto cleanup; 
			append_bats[i].batid = b->batCacheid;
		}
		msg = monetdb_append(schema_name, table_name, append_bats, columns);
cleanup:
		if (pyreturn_values) GDKfree(pyreturn_values);
		if (column_names) GDKfree(column_names);
		if (column_types) GDKfree(column_types);
		if (append_bats) {
			for(i = 0; i < columns; i++) {
				if (append_bats[i].batid != int_nil) BBPunfix(append_bats[i].batid);
			}
			GDKfree(append_bats);
		}
		if (msg != NULL) {
			PyErr_Format(PyExc_Exception, "%s", msg);
			return NULL;
		}
	}
	Py_RETURN_NONE;
}

void init_embedded_py(void)
{
    //import numpy stuff
    import_array();
}

/////////////////////////////////////////////////////////
// This code is copy pasted from embedded.c, should be changed to a shared library
/////////////////////////////////////////////////////////
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

static void* lookup_function(char* lib, char* func) {
	void *dl, *fun;
	dl = mdlopen(lib, RTLD_NOW | RTLD_GLOBAL);
	if (dl == NULL) {
		return NULL;
	}
	fun = dlsym(dl, func);
	dlclose(dl);
	return fun;
}

char* monetdb_startup(char* dir, char silent) {
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
	setlen = mo_add_option(&set, setlen, opt_cmdline, "gdk_dbpath", dir);
	if (GDKinit(set, setlen) == 0) {
		retval = GDKstrdup("GDKinit() failed");
		goto cleanup;
	}

	snprintf(mod_path, 1000, "%s/../lib/monetdb5", BINDIR);
	GDKsetenv("monet_mod_path", mod_path);
	GDKsetenv("mapi_disable", "true");
	GDKsetenv("max_clients", "0");
	GDKsetenv("sql_optimizer", "sequential_pipe"); // TODO: SELECT * FROM table should not use mitosis in the first place.

	if (silent) THRdata[0] = stream_blackhole_create();
	msab_dbpathinit(GDKgetenv("gdk_dbpath"));
	if (mal_init() != 0) {
		retval = GDKstrdup("mal_init() failed");
		goto cleanup;
	}
	if (silent) mal_clients[0].fdout = THRdata[0];

	// This dynamically looks up functions, because the library containing them is loaded at runtime.
	// argh
	SQLstatementIntern_ptr = (SQLstatementIntern_ptr_tpe) lookup_function("lib_sql",  "SQLstatementIntern");
	SQLautocommit_ptr = (SQLautocommit_ptr_tpe) lookup_function("lib_sql",  "SQLautocommit");
	SQLinitClient_ptr = (SQLinitClient_ptr_tpe) lookup_function("lib_sql",  "SQLinitClient");
	getSQLContext_ptr = (getSQLContext_ptr_tpe) lookup_function("lib_sql",  "getSQLContext");
	res_table_destroy_ptr  = (res_table_destroy_ptr_tpe)  lookup_function("libstore", "res_table_destroy");
	mvc_append_wrap_ptr = (mvc_append_wrap_ptr_tpe)  lookup_function("lib_sql", "mvc_append_wrap");
	mvc_bind_schema_ptr = (mvc_bind_schema_ptr_tpe)  lookup_function("lib_sql", "mvc_bind_schema");
	mvc_bind_table_ptr = (mvc_bind_table_ptr_tpe)  lookup_function("lib_sql", "mvc_bind_table");
	sqlcleanup_ptr = (sqlcleanup_ptr_tpe)  lookup_function("lib_sql", "sqlcleanup");
	mvc_trans_ptr = (mvc_trans_ptr_tpe) lookup_function("lib_sql", "mvc_trans");

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

	// sanity check, run a SQL query
	sqres = monetdb_query("SELECT * FROM tables;", res);
	if (sqres != NULL) {
		monetdb_embedded_initialized = false;
		retval = sqres;
		goto cleanup;
	}
cleanup:
	mo_free_options(set, setlen);
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

char* monetdb_append(const char* schema, const char* table, append_data *data, int col_ct) {
	int i;
	int nvar = 6; // variables we need to make up
	MalBlkRecord mb;
	MalStack*     stk = NULL;
	InstrRecord*  pci = NULL;
	str res = MAL_SUCCEED;
	VarRecord bat_varrec;
	mvc* m = ((backend *) mal_clients[0].sqlcontext)->mvc;

	assert(table != NULL && data != NULL && col_ct > 0);

	// very black MAL magic below
	mb.var = GDKmalloc(nvar * sizeof(VarRecord*));
	stk = GDKmalloc(sizeof(MalStack) + nvar * sizeof(ValRecord));
	pci = GDKmalloc(sizeof(InstrRecord) + nvar * sizeof(int));
	assert(mb.var != NULL && stk != NULL && pci != NULL); // cough, cough
	bat_varrec.type = TYPE_bat;
	for (i = 0; i < nvar; i++) {
		pci->argv[i] = i;
	}
	stk->stk[0].vtype = TYPE_int;
	stk->stk[2].val.sval = (str) schema;
	stk->stk[2].vtype = TYPE_str;
	stk->stk[3].val.sval = (str) table;
	stk->stk[3].vtype = TYPE_str;
	stk->stk[4].vtype = TYPE_str;
	stk->stk[5].vtype = TYPE_bat;
	mb.var[5] = &bat_varrec;
	if (!m->session->active) (*mvc_trans_ptr)(m);
	for (i=0; i < col_ct; i++) {
		append_data ad = data[i];
		stk->stk[4].val.sval = ad.colname;
		stk->stk[5].val.bval = ad.batid;

		res = (*mvc_append_wrap_ptr)(&mal_clients[0], &mb, stk, pci);
		if (res != NULL) {
			break;
		}
	}
	if (res == MAL_SUCCEED) {
		(*sqlcleanup_ptr)(m, 0);
	}
	GDKfree(mb.var);
	GDKfree(stk);
	GDKfree(pci);
	return res;
}

void monetdb_cleanup_result(void* output) {
	(*res_table_destroy_ptr)((res_table*) output);
}

static str monetdb_get_columns(const char* schema_name, const char *table_name, int *column_count, char ***column_names, int **column_types) 
{
	Client c = &mal_clients[0];
	mvc *m;
	sql_schema *s;
	sql_table *t;
	char *msg = MAL_SUCCEED;
	int columns;
	node *n;

	assert(column_count != NULL && column_names != NULL && column_types != NULL);

	if ((msg = (*getSQLContext_ptr)(c, NULL, &m, NULL)) != NULL)
		return msg;

	s = (*mvc_bind_schema_ptr)(m, schema_name);
	if (s == NULL)
		return createException(MAL, "embedded", "Missing schema!");
	t = (*mvc_bind_table_ptr)(m, s, table_name);
	if (t == NULL)
		return createException(MAL, "embedded", "Could not find table %s", table_name);

	columns = t->columns.set->cnt;
	*column_count = columns;
	*column_names = GDKzalloc(sizeof(char*) * columns);
	*column_types = GDKzalloc(sizeof(int) * columns);

	if (*column_names == NULL || *column_types == NULL) {
		return MAL_MALLOC_FAIL;
	}

	for (n = t->columns.set->h; n; n = n->next) {
		sql_column *c = n->data;
		(*column_names)[c->colnr] = c->base.name;
		(*column_types)[c->colnr] = c->type.type->localtype;
	}

	return msg;
}

static char *BatType_ToSQLType(int type)
{
    switch (type)
    {
        case TYPE_bit:
        case TYPE_bte: return "TINYINT";
        case TYPE_sht: return "SMALLINT";
        case TYPE_int: return "INTEGER";
        case TYPE_lng: return "BIGINT";
        case TYPE_flt: return "FLOAT";
        case TYPE_dbl: return "DOUBLE";
        case TYPE_str: return "STRING";
        case TYPE_hge: return "HUGEINT";
        case TYPE_oid: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

str monetdb_create_table(char *schema, char *table_name, append_data *ad, int ncols)
{
	const int max_length = 10000;
	char query[max_length], copy[max_length];
	int i;
	void *res;
	char *msg = MAL_SUCCEED;

	if (!monetdb_embedded_initialized) {
		fprintf(stderr, "Embedded MonetDB is not started.\n");
		return NULL;
	}
	//format the CREATE TABLE query
	snprintf(query, max_length, "create table %s.%s(", schema, table_name);
	for(i = 0; i < ncols; i++) {
		BAT *b = BBP_cache(ad[i].batid);
		snprintf(copy, max_length, "%s %s %s%s", query, ad[i].colname, BatType_ToSQLType(b->T->type), i < ncols - 1 ? "," : ");");
		strcpy(query, copy);
	}

	msg = monetdb_query(query, &res);
	if (msg != MAL_SUCCEED) {
		return msg;
	}

	return monetdb_append(schema, table_name, ad, ncols);
}
