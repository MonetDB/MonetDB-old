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

#include "pyclient.h"
#include "embedded.h"

static str monetdblite_insert(PyObject *client, char *schema, char *table, PyObject *values, char **column_names, int *column_types, sql_subtype **sql_subtypes, int columns);

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
		char installdir[1024];
		printf("Making directory %s\n", directory);
		if (GDKcreatedir(directory) != GDK_SUCCEED) {
   			PyErr_Format(PyExc_Exception, "Failed to create directory %s.", directory);
   			return NULL;
		}
		snprintf(installdir, 1024, "%s/../", BINDIR);
		msg = monetdb_startup(installdir, directory, 1);
		if (msg != MAL_SUCCEED) {
	   		PyErr_Format(PyExc_Exception, "Failed to initialize MonetDB. %s", msg);
			return NULL;
		}
		PyAPIprelude(NULL);
	}
	Py_RETURN_NONE;
}

PyObject *monetdb_sql(PyObject *self, PyObject *args, PyObject *keywds)
{
	Client c = monetdb_default_client;
	char *query;
	PyObject *client = NULL;
	static char *kwlist[] = {"query", "conn", NULL};
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|O", kwlist, &query, &client)) {
		return NULL;
	}
	if (client != NULL) {
		if (!PyClient_CheckExact(client)) {
	   		PyErr_SetString(PyExc_Exception, "conn must be a connection object created by monetdblite.connect().");
			return NULL;
		}
		c = ((PyClientObject*)client)->cntxt;
	}
	{
		PyObject *result;
		res_table* output = NULL;
		PyObject *querystring;
		char* err;
		// Append ';'' to the SQL query, just in case
		querystring = PyString_FromFormat("%s;", query);
		// Perform the SQL query
Py_BEGIN_ALLOW_THREADS
		MT_lock_set(&c->query_lock, "client.query_lock");
		err = monetdb_query(c, &(((PyStringObject*)querystring)->ob_sval[0]), (void**)&output);
		MT_lock_unset(&c->query_lock, "client.query_lock");
Py_END_ALLOW_THREADS
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

            	numpy_array = PyMaskedArray_FromBAT(c, &input, 0, input.count, &msg, true);
            	if (!numpy_array) {
					monetdb_cleanup_result(c, output);
			   		PyErr_Format(PyExc_Exception, "SQL Query Failed: %s", (msg ? msg : "<no error>"));
					return NULL;
            	}
            	PyDict_SetItem(result, PyString_FromString(output->cols[i].name), numpy_array);
			}
			monetdb_cleanup_result(c, output);
			return result;
		} else {
			Py_RETURN_NONE;
		}
	}
}

PyObject *monetdb_create(PyObject *self, PyObject *args, PyObject *keywds)
{
	char *schema_name = "sys";
	char *table_name;
	PyObject *values = NULL, *client = NULL, *colnames = NULL;
	char **column_names = NULL;
	char *msg = NULL;
	PyObject *keys = NULL;
	static char *kwlist[] = {"name", "values", "colnames", "schema", "conn", NULL};
	int columns;
	int i;
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO|OsO", kwlist, &table_name, &values, &colnames, &schema_name, &client)) {
		return NULL;
	}

	if (PyDict_CheckExact(values)) {
		keys = PyDict_Keys(values);
		colnames = keys;
	} else {
		if (colnames == NULL) {
	   		PyErr_SetString(PyExc_TypeError, "no colnames are specified and values is not a dict");
			return NULL;
		} 
		if (!PyList_Check(colnames)) {
	   		PyErr_SetString(PyExc_TypeError, "colnames must be a list");
			return NULL;
		}
		if (PyList_Size(colnames) == 0) {
	   		PyErr_SetString(PyExc_TypeError, "colnames must have at least one element");
			return NULL;
		}
	}
	columns = PyList_Size(colnames);
	column_names = GDKzalloc(columns * sizeof(char*));
	for(i = 0; i < columns; i++) {
		PyObject *key = PyList_GetItem(colnames, i);
		if (!PyString_CheckExact(key)) {
			msg = GDKzalloc(1024);
			snprintf(msg, 1024, "expected a key of type 'str', but key was of type %s", key->ob_type->tp_name);
			goto cleanup;
		}
		column_names[i] = PyString_AS_STRING(key);
	}
	msg = monetdblite_insert(client, schema_name, table_name, values, column_names, NULL, NULL, columns);
cleanup:
	if (column_names) GDKfree(column_names);
	if (keys) Py_DECREF(keys);
	if (msg != NULL) {
		PyErr_Format(PyExc_Exception, "%s", msg);
		return NULL;
	}
	Py_RETURN_NONE;
}

static str PyClientObject_GetClient(PyObject *client, Client *c);
static str PyClientObject_GetClient(PyObject *client, Client *c) {
	*c = monetdb_default_client;
	if (client != NULL) {
		if (!PyClient_CheckExact(client)) {
			return GDKstrdup("conn must be a connection object created by monetdblite.connect().");
		}
		*c =((PyClientObject*)client)->cntxt;
	}
	return MAL_SUCCEED;
}

static str monetdblite_insert(PyObject *client, char *schema_name, char *table_name, PyObject *values, char **column_names, int *column_types, sql_subtype **sql_subtypes, int columns) {
	Client c;
	PyReturn *pyreturn_values = NULL;
	append_data *append_bats = NULL;
	PyObject *dict_vals = NULL;
	PyObject *keys = NULL;
	PyObject *pResult = NULL;
	int *key_map = NULL;
	char *msg = MAL_SUCCEED;
	int i, j;
	msg = PyClientObject_GetClient(client, &c);
	if (msg != NULL) goto cleanup;

	if (PyDict_Check(values)) {
		int key_cnt; 
		keys = PyDict_Keys(values);
		key_cnt = PyList_Size(keys);
		key_map = GDKzalloc(sizeof(int) * key_cnt);
		for(i = 0; i < key_cnt; i++) {
			PyObject *key = PyList_GetItem(keys, i);
			if (!PyString_CheckExact(key)) {
				msg = GDKzalloc(1024);
				snprintf(msg, 1024, "expected a key of type 'str', but key was of type %s", key->ob_type->tp_name);
				goto cleanup;
			}
			key_map[i] = -1;
			for(j = 0; j < columns; j++) {
				if (strcasecmp(PyString_AS_STRING(key), column_names[j]) == 0)
					key_map[i] = j;
			}
		}
		dict_vals = PyList_New(columns);
		for(j = 0; j < columns; j++) {
			int index = -1;
			for(i = 0; i < key_cnt; i++) {
				if (key_map[i] == j) {
					index = i;
					break;
				}
			}
			if (index < 0) {
				msg = GDKzalloc(1024);
				snprintf(msg, 1024, "could not find required key %s", column_names[j]);
				goto cleanup;
			}
			PyList_SetItem(dict_vals, j, PyDict_GetItem(values, PyList_GetItem(keys, index)));
		}
		values = dict_vals;
	}
	
	pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
	if (pResult == NULL) goto cleanup;
	pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
	if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;

	append_bats = GDKzalloc(sizeof(append_data) * columns);
	for(i = 0; i < columns; i++) {
		append_bats[i].batid = int_nil;
		append_bats[i].colname = column_names[i];
	}
	for(i = 0; i < columns; i++) {
		BAT *b = PyObject_ConvertToBAT(&pyreturn_values[i], sql_subtypes ? sql_subtypes[i] : NULL, column_types ? column_types[i] : PyType_ToBat(pyreturn_values[i].result_type), i, 0, &msg, true);

		if (b == NULL) goto cleanup; 
		append_bats[i].batid = b->batCacheid;
	}
Py_BEGIN_ALLOW_THREADS
	MT_lock_set(&c->query_lock, "client.query_lock");
	if (!column_types) 
		msg = monetdb_create_table(c, schema_name, table_name, append_bats, columns);
	else
		msg = monetdb_append(c, schema_name, table_name, append_bats, columns);
	MT_lock_unset(&c->query_lock, "client.query_lock");
Py_END_ALLOW_THREADS
cleanup:
	if (pyreturn_values) GDKfree(pyreturn_values);
	if (dict_vals) Py_DECREF(dict_vals);
	if (key_map) GDKfree(key_map);
	if (keys) Py_DECREF(keys);
	if (append_bats) {
		for(i = 0; i < columns; i++) {
			if (append_bats[i].batid != int_nil) BBPunfix(append_bats[i].batid);
		}
		GDKfree(append_bats);
	}
	return msg;
}

PyObject *monetdb_insert(PyObject *self, PyObject *args, PyObject *keywds)
{
	char *schema_name = "sys";
	char *table_name;
	PyObject *values = NULL, *client = NULL;
	static char *kwlist[] = {"name", "values", "schema", "conn", NULL};
	char *msg = NULL;
	int columns;
	char **column_names = NULL;
	int *column_types = NULL;
	sql_subtype **sql_subtypes = NULL;
	Client c;
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO|sO", kwlist, &table_name, &values, &schema_name, &client)) {
		return NULL;
	}
	msg = PyClientObject_GetClient(client, &c);
	if (msg != NULL) {
		goto cleanup;
	}
	msg = monetdb_get_columns(c, schema_name, table_name, &columns, &column_names, &column_types, (void***)&sql_subtypes);
	if (msg != NULL) {
		goto cleanup;
	}
	msg = monetdblite_insert(client, schema_name, table_name, values, column_names, column_types, sql_subtypes, columns);
cleanup:
	if (column_names) GDKfree(column_names);
	if (column_types) GDKfree(column_types);
	if (sql_subtypes) GDKfree(sql_subtypes);
	if (msg != NULL) {
		PyErr_Format(PyExc_Exception, "%s", msg);
		return NULL;
	}
	Py_RETURN_NONE;
}

PyObject *monetdb_client(PyObject *self)
{
	Client c = monetdb_connect();
	(void) self;
	if (c == NULL) {
		PyErr_Format(PyExc_Exception, "Failed to create client context.");
		return NULL;
	}
	return PyClient_Create(c);
}

void monetdblite_init(void)
{
    //import numpy stuff
    import_array();
    //init monetdb client
    monetdbclient_init();
}
