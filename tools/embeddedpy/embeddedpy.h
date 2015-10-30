/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * @a M. Raasveldt
 * MonetDB embedded in Python library.
 */

#ifndef _EMBEDDEDPY_LIB_
#define _EMBEDDEDPY_LIB_

// Python library
#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

//! Initializes MonetDB with the specified directory, "args" must be a PyStringObject
PyObject *monetdb_init(PyObject *self, PyObject *args);
//! Performs a SQL query, monetdb_init must be called first and "args" must be a PyStringObject
PyObject *monetdb_sql(PyObject *self, PyObject *args, PyObject *keywds);
//! Creates a SQL table, monetdb_init must be called first and "args" must be either (table_name, dictionary) or (table_name, list(column_names), list(values))
PyObject *monetdb_create(PyObject *self, PyObject *args, PyObject *keywds);
//! Inserts values into a SQL table, monetdb_init must be called first and "args" must be either (table_name, dictionary) or (table_name, list(column_names), list(values))
PyObject *monetdb_insert(PyObject *self, PyObject *args, PyObject *keywds);
//! Creates a new MonetDB client context and returns it as a PyClientObject
PyObject *monetdb_client(PyObject *self);

//! Initializes numpy, if this isn't called you will get segfaults when calling numpy functions
void monetdblite_init(void);


#endif /* _EMBEDDEDPY_LIB_ */
