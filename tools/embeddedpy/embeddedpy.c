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
	Client c = monetdb_default_client;
	static char *kwlist[] = {"name", "values", "colnames", "schema", "conn", NULL};
	int i;
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO|OsO", kwlist, &table_name, &values, &colnames, &schema_name, &client)) {
		return NULL;
	}
	if (client != NULL) {
		if (!PyClient_CheckExact(client)) {
	   		PyErr_SetString(PyExc_Exception, "conn must be a connection object created by monetdblite.connect().");
			return NULL;
		}
		c = ((PyClientObject*)client)->cntxt;
	}
	if (colnames == NULL) {
		if (!PyDict_CheckExact(values)) {
	   		PyErr_SetString(PyExc_TypeError, "no colnames are specified and values is not a dict");
			return NULL;
		}
   		PyErr_Format(PyExc_Exception, "dict is not implemented yet");
		return NULL;
		Py_RETURN_NONE;
	} else {
		if (!PyList_Check(colnames)) {
	   		PyErr_SetString(PyExc_TypeError, "colnames must be a list");
			return NULL;
		}
		if (PyList_Size(colnames) == 0) {
	   		PyErr_SetString(PyExc_TypeError, "colnames must have at least one element");
			return NULL;
		}

		for(i = 0; i < PyList_Size(colnames); i++) {
			PyObject *column_name = PyList_GetItem(colnames, i);
			if (!PyString_CheckExact(column_name)) {
		   		PyErr_Format(PyExc_TypeError, "the entry %d in the column names is not a string", i);
				return NULL;
			}
		}
		//convert the values to BATs and create the table
		{
			char *msg = NULL;
			PyObject *pResult;
			int columns = PyList_Size(colnames);
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
				append_bats[i].colname = PyString_AS_STRING(PyList_GetItem(colnames, i));
 			}
Py_BEGIN_ALLOW_THREADS
			MT_lock_set(&c->query_lock, "client.query_lock");
 			msg = monetdb_create_table(c, schema_name, table_name, append_bats, columns);
			MT_lock_unset(&c->query_lock, "client.query_lock");
Py_END_ALLOW_THREADS
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

PyObject *monetdb_insert(PyObject *self, PyObject *args, PyObject *keywds)
{
	char *schema_name = "sys";
	char *table_name;
	PyObject *values = NULL, *client = NULL;
	Client c = monetdb_default_client;
	static char *kwlist[] = {"name", "values", "schema", "conn", NULL};
	(void) self;
	if (!monetdb_embedded_initialized) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO|sO", kwlist, &table_name, &values, &schema_name, &client)) {
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
		char *msg = NULL;
		PyObject *pResult;
		PyReturn *pyreturn_values = NULL;
		append_data *append_bats = NULL;
		int i;
		char **column_names = NULL;
		int *column_types = NULL;
		sql_subtype **sql_subtypes = NULL;
		int columns = 0;

		msg = monetdb_get_columns(c, schema_name, table_name, &columns, &column_names, &column_types, (void***)&sql_subtypes);

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
			BAT *b = PyObject_ConvertToBAT(&pyreturn_values[i], sql_subtypes[i], column_types[i], i, 0, &msg, true);

			if (b == NULL) goto cleanup; 
			append_bats[i].batid = b->batCacheid;
		}
Py_BEGIN_ALLOW_THREADS
		MT_lock_set(&c->query_lock, "client.query_lock");
		msg = monetdb_append(c, schema_name, table_name, append_bats, columns);
		MT_lock_unset(&c->query_lock, "client.query_lock");
Py_END_ALLOW_THREADS
cleanup:
		if (pyreturn_values) GDKfree(pyreturn_values);
		if (column_names) GDKfree(column_names);
		if (column_types) GDKfree(column_types);
		if (sql_subtypes) GDKfree(sql_subtypes);
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
