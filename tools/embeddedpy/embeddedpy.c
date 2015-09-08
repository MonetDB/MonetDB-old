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

#include "pyapi.h"
#include "pytypes.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// copy paste from embedded.c
bool monetdb_isinit(void);
int monetdb_startup(char* dir, char silent);
char* monetdb_query(char* query, void** result);
void monetdb_cleanup_result(void* output);
str monetdb_get_columns(char *table_name, int *column_count, char ***column_names, int **column_types);
str monetdb_create_table(char *table_name, size_t columns, BAT **bats, char **column_names);
str monetdb_insert_into_table(char *table_name, size_t columns, BAT **bats, char **column_names);
/////////////////////////////////////////////////////////////////////////////////////////////////

//LOCALSTATEDIR
PyObject *monetdb_init(PyObject *self, PyObject *args)
{
	(void) self;
	if (!PyString_CheckExact(args)) {
   		PyErr_SetString(PyExc_TypeError, "Expected a directory name as an argument.");
		return NULL;
	}

	{
		char *directory = &(((PyStringObject*)args)->ob_sval[0]);
		printf("Making directory %s\n", directory);
		if (GDKcreatedir(directory) != GDK_SUCCEED) {
   			PyErr_Format(PyExc_Exception, "Failed to create directory %s.", directory);
   			return NULL;
		}
		if (monetdb_startup(directory, 1) < 0) {
	   		PyErr_SetString(PyExc_Exception, "Failed to initialize MonetDB with the specified directory.");
			return NULL;
		}
		GDKsetenv("enable_numpystringarray", "true");
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
	if (!monetdb_isinit()) {
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

            	numpy_array = PyMaskedArray_FromBAT(&input, 0, input.count, &msg);
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
	char *table_name;
	PyObject *dict = NULL, *values = NULL;
	int i;
	(void) self;
	if (!monetdb_isinit()) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "sO|O", &table_name, &dict, &values)) {
		return NULL;
	}
	if (values == NULL) {
		if (!PyDict_CheckExact(dict)) {
	   		PyErr_SetString(PyExc_TypeError, "the second argument to be a dict");
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
			PyReturn *pyreturn_values = NULL;
			BAT **bats = NULL;
			char **column_names = NULL;

			pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
			if (pResult == NULL) goto cleanup;
			pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
			if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;

			bats = GDKzalloc(sizeof(BAT*) * columns);
			column_names = GDKzalloc(sizeof(char*) * columns);
			for(i = 0; i < columns; i++) {
				column_names[i] = PyString_AS_STRING(PyList_GetItem(dict, i));
				bats[i] = PyObject_ConvertToBAT(&pyreturn_values[i], PyType_ToBat(pyreturn_values[i].result_type), i, 0, &msg);
				if (bats[i] == NULL) goto cleanup; 
 			}
 			msg = monetdb_create_table(table_name, columns, bats, column_names);
cleanup:
			if (pyreturn_values) GDKfree(pyreturn_values);
			if (bats) {
				for(i = 0; i < columns; i++) {
					if (bats[i] != NULL) BBPunfix(bats[i]->batCacheid);
				}
				GDKfree(bats);
			}
			if (column_names) GDKfree(column_names);
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
	char *table_name;
	PyObject *values = NULL;
	(void) self;
	if (!monetdb_isinit()) {
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
		BAT **bats = NULL;
		int i;
		char **column_names = NULL;
		int *column_types = NULL;
		int columns;

		msg = monetdb_get_columns(table_name, &columns, &column_names, &column_types);

		pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
		if (pResult == NULL) goto cleanup;
		pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
		if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;

		bats = GDKzalloc(sizeof(BAT*) * columns);
		for(i = 0; i < columns; i++) {
			bats[i] = PyObject_ConvertToBAT(&pyreturn_values[i], column_types[i], i, 0, &msg);
			if (bats[i] == NULL) goto cleanup; 
			}
			msg = monetdb_insert_into_table(table_name, columns, bats, column_names);
cleanup:
		if (pyreturn_values) GDKfree(pyreturn_values);
		if (column_names) GDKfree(column_names);
		if (column_types) GDKfree(column_types);
		if (bats) {
			for(i = 0; i < columns; i++) {
				if (bats[i] != NULL) BBPunfix(bats[i]->batCacheid);
			}
			GDKfree(bats);
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
typedef void (*res_table_destroy_ptr_tpe)(res_table *t);
res_table_destroy_ptr_tpe res_table_destroy_ptr = NULL;

static bit monetdb_embedded_initialized = 0;
static MT_Lock monetdb_embedded_lock;

bool monetdb_isinit(void) {
	return monetdb_embedded_initialized;
}

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

int monetdb_startup(char* dir, char silent) {
	opt *set = NULL;
	int setlen = 0;
	int retval = -1;
	void* res = NULL;
	char mod_path[1000];

	MT_lock_init(&monetdb_embedded_lock, "monetdb_embedded_lock");
	MT_lock_set(&monetdb_embedded_lock, "monetdb.startup");
	if (monetdb_embedded_initialized) goto cleanup;

	setlen = mo_builtin_settings(&set);
	setlen = mo_add_option(&set, setlen, opt_cmdline, "gdk_dbpath", dir);
	if (GDKinit(set, setlen) == 0)  goto cleanup;

	snprintf(mod_path, 1000, "%s/../lib/monetdb5", BINDIR);
	GDKsetenv("monet_mod_path", mod_path);
	GDKsetenv("mapi_disable", "true");
	GDKsetenv("max_clients", "0");

	if (silent) THRdata[0] = stream_blackhole_create();
	msab_dbpathinit(GDKgetenv("gdk_dbpath"));
	if (mal_init() != 0) goto cleanup;
	if (silent) mal_clients[0].fdout = THRdata[0];

	// This dynamically looks up functions, because the library containing them is loaded at runtime.
	SQLstatementIntern_ptr = (SQLstatementIntern_ptr_tpe) lookup_function("lib_sql",  "SQLstatementIntern");
	res_table_destroy_ptr  = (res_table_destroy_ptr_tpe)  lookup_function("libstore", "res_table_destroy");
	if (SQLstatementIntern_ptr == NULL || res_table_destroy_ptr == NULL) goto cleanup;

	monetdb_embedded_initialized = true;
	// sanity check, run a SQL query
	if (monetdb_query("SELECT * FROM tables;", res) != NULL) {
		monetdb_embedded_initialized = false;
		goto cleanup;
	}
	if (res != NULL) monetdb_cleanup_result(res);
	retval = 0;

cleanup:
	mo_free_options(set, setlen);
	MT_lock_unset(&monetdb_embedded_lock, "monetdb.startup");
	return retval;
}

char* monetdb_query(char* query, void** result) {
	str res;
	Client c = &mal_clients[0];
	if (!monetdb_embedded_initialized) {
		fprintf(stderr, "Embedded MonetDB is not started.\n");
		return NULL;
	}
	res = (*SQLstatementIntern_ptr)(c, &query, "name", 1, 0, (res_table **) result);
	SQLautocommit(c, ((backend *) c->sqlcontext)->mvc);
	return res;
}

void monetdb_cleanup_result(void* output) {
	(*res_table_destroy_ptr)((res_table*) output);
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

static char *BatType_DefaultValue(int type)
{
    switch (type)
    {
        case TYPE_bit:
        case TYPE_bte: 
        case TYPE_sht: 
        case TYPE_int:
        case TYPE_lng:
        case TYPE_flt: 
        case TYPE_hge:
        case TYPE_dbl: return "0";
        case TYPE_str: return "''";
        default: return "UNKNOWN";
    }
}

str monetdb_insert_into_table(char *table_name, size_t columns, BAT **bats, char **column_names)
{
	size_t i;
	const int max_length = 10000;
	char query[max_length], copy[max_length];
	Client c = &mal_clients[0];
	char *msg = MAL_SUCCEED;
	void *res;

	snprintf(query, max_length, "insert into %s values ", table_name);
	for(i = 0; i < columns; i++) {
		snprintf(copy, max_length, "%s%s%s%s", query, i == 0 ? "(" : "", BatType_DefaultValue(bats[i]->T->type), i < columns - 1 ? "," : ");");
		strcpy(query, copy);
	}

	c->_append_columns = columns;
	c->_append_bats = (void**)bats;
	c->_append_column_names = column_names;
	msg = monetdb_query(query, &res);
	c->_append_columns = 0;
	c->_append_bats = NULL;
	c->_append_column_names = NULL;
	if (msg != MAL_SUCCEED) {
		return msg;
	}
	return msg;
}

str monetdb_create_table(char *table_name, size_t columns, BAT **bats, char **column_names)
{
	const int max_length = 10000;
	char query[max_length], copy[max_length];
	size_t i;
	void *res;
	char *msg = MAL_SUCCEED;

	if (!monetdb_embedded_initialized) {
		fprintf(stderr, "Embedded MonetDB is not started.\n");
		return NULL;
	}
	//format the CREATE TABLE query
	snprintf(query, max_length, "create table %s(", table_name);
	for(i = 0; i < columns; i++) {
		snprintf(copy, max_length, "%s %s %s%s", query, column_names[i], BatType_ToSQLType(bats[i]->T->type), i < columns - 1 ? "," : ");");
		strcpy(query, copy);
	}

	msg = monetdb_query(query, &res);
	if (msg != MAL_SUCCEED) {
		return msg;
	}

	return monetdb_insert_into_table(table_name, columns, bats, column_names);
}


str monetdb_get_columns(char *table_name, int *column_count, char ***column_names, int **column_types)
{
	Client c = &mal_clients[0];
	mvc *m;
	sql_schema *s;
	sql_table *t;
	char *msg = MAL_SUCCEED;

	if ((msg = getSQLContext(c, NULL, &m, NULL)) != NULL)
		return msg;

	s = mvc_bind_schema(m, "sys");
	if (s == NULL)
		msg = createException(MAL, "embedded", "Missing schema!");
	t = mvc_bind_table(m, s, table_name);
	if (t == NULL)
		msg = createException(MAL, "embedded", "Could not find table %s", table_name);

	{
		const int columns = t->columns.set->cnt;
		if (column_count != NULL) {
			*column_count = columns;
		}
		if (column_names != NULL) {
			int i;
			*column_names = GDKzalloc(sizeof(char*) * columns);
			for(i = 0; i < columns; i++) {
				*column_names[i] = ((sql_base*)t->columns.set->h->data)[i].name;
			}
		}
		if (column_types != NULL) {
			int i;
			*column_types = GDKzalloc(sizeof(int) * columns);
			for(i = 0; i < columns; i++) {
				*column_types[i] = ((sql_column*)t->columns.set->h->data)[i].type.type->localtype;
			}
		}
	}

	return msg;
}
