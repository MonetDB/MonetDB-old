/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * 
 */

#ifndef _LOOPBACK_QUERY_
#define _LOOPBACK_QUERY_

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

#include "monetdb_config.h"
#include "mal.h"
#include "mal_exception.h"
#include "mal_interpreter.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "gdk_utils.h"
#include "gdk.h"
#include "sql_catalog.h"

extern PyTypeObject *_connection_type;

typedef struct {
    PyObject_HEAD
    Client cntxt;
} Py_ConnectionObject;

PyAPI_DATA(PyTypeObject) Py_ConnectionType;

#define Py_Connection_Check(op) (Py_TYPE(op) == &Py_ConnectionType)
#define Py_Connection_CheckExact(op) (Py_TYPE(op) == &Py_ConnectionType)

PyObject *Py_Connection_Create(Client cntxt);

void _connection_init(void);

#endif /* _LOOPBACK_QUERY_ */
