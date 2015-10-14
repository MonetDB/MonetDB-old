/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * @a M. Raasveldt
 * The actual Embedded Python module that holds the functions. Just a wrapper, the actual functions are declared in embeddedpy.c
 */
#include "embeddedpy.h"

static char module_docstring[] =
    "This module provides a MonetDB client.";
static char init_docstring[] =
    "monetdblite.init(directory) => Initialize the SQL client with the given database directory.";
static char sql_docstring[] =
    "monetdblite.sql(query) => Execute a SQL query on the database. Returns the result as a dictionary of Numpy Arrays.";
static char create_docstring[] =
    "monetdblite.create(tablename, dictionary), monetdblite.create(tablename, column_names, values) => Create a SQL table from the given Python objects, objects must either be a (column name, value) dictionary or a list of column names and a list of values";
static char insert_docstring[] =
    "monetdblite.insert(tablename, dictionary), monetdblite.insert(tablename, column_names, values) => Insert a set of values into a SQL table";

static PyMethodDef module_methods[] = {
    {"init", monetdb_init, METH_O, init_docstring},
    {"sql", monetdb_sql, METH_O, sql_docstring},
    {"create", monetdb_create, METH_VARARGS, create_docstring},
    {"insert", monetdb_insert, METH_VARARGS, insert_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmonetdblite(void);
PyMODINIT_FUNC initmonetdblite(void)
{
    //initialize module
    PyObject *m = Py_InitModule3("monetdblite", module_methods, module_docstring);
    if (m == NULL)
        return;

    //import numpy stuff
    init_embedded_py();
}
