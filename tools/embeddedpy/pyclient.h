/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * @a M. Raasveldt
 * Python Object wrapper for a MonetDB Client Context object.
 */

#ifndef _PYCLIENT_LIB_
#define _PYCLIENT_LIB_

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

#include "monetdb_config.h"
#include "monet_options.h"
#include "mal.h"
#include "mal_client.h"

extern PyTypeObject *_connection_type;

typedef struct {
    PyObject_HEAD
    Client cntxt;
} PyClientObject;

PyAPI_DATA(PyTypeObject) PyClientType;

#define PyClient_Check(op) (Py_TYPE(op) == &PyClientType)
#define PyClient_CheckExact(op) (Py_TYPE(op) == &PyClientType)

PyObject *PyClient_Create(Client cntxt);

//! Initialize PyClientObject class, called by monetdblite_init()
void monetdbclient_init(void);

#endif /* _PYCLIENT_LIB_ */
