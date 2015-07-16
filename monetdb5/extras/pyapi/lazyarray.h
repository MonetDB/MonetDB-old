/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains the Python type "LazyArray", which is a type that holds a string or hge BAT that will only convert them to python types when they are required.
 */

#ifndef _LAZYARRAY_
#define _LAZYARRAY_

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

#include "monetdb_config.h"
#include "mal.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "gdk_utils.h"
#include "gdk.h"
#include "sql_catalog.h"

typedef struct {
    PyObject_VAR_HEAD
    BAT *bat;
    PyObject **array;
    PyObject *numpy_array;
} PyLazyArrayObject;

hge PyLazyArray_GetHuge(PyLazyArrayObject *, Py_ssize_t index);
char* PyLazyArray_GetString(PyLazyArrayObject *, Py_ssize_t index);
bool PyLazyArray_IsNil(PyLazyArrayObject*, Py_ssize_t index);

PyObject* PyLazyArray_GetItem(PyLazyArrayObject *, Py_ssize_t index);

PyObject *PyLazyArray_FromBAT(BAT *b);

PyAPI_DATA(PyTypeObject) PyLazyArray_Type;

PyObject *PyLazyArray_AsNumpyArray(PyLazyArrayObject *a, size_t start, size_t end);

#define PyLazyArray_Check(op) (Py_TYPE(op) == &PyLazyArray_Type)
#define PyLazyArray_CheckExact(op) (Py_TYPE(op) == &PyLazyArray_Type)

#define PyLazyArray_GET_SIZE(op)  Py_SIZE(op)
#define PyLazyArray_GET_TYPE(op)  ATOMstorage(getColumnType(LAZYARRAY_BAT(op)->T->type))

void lazyarray_init(void);

#endif /* _LAZYARRAY_ */
