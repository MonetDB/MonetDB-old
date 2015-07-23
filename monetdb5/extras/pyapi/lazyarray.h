/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains the Python type "LazyArray", which is a type that holds a BAT that will only convert them to python types when they are required.
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

extern PyTypeObject *_maskedarray_type;
extern PyTypeObject *_lazyarray_type;
extern PyObject *_no_mask;
extern size_t _lazyarray_size;

typedef struct {
    PyObject_HEAD
    BAT *bat;
} PyBATObject;

PyObject *PyBAT_FromBAT(BAT *b);

PyAPI_DATA(PyTypeObject) PyBAT_Type;

#define PyBAT_Check(op) (Py_TYPE(op) == &PyBAT_Type)
#define PyBAT_CheckExact(op) (Py_TYPE(op) == &PyBAT_Type)

PyObject *PyLazyArray_AsNumpyArray(PyObject *a);
PyObject *PyLazyArray_FromBAT(BAT *b);

#define PyLazyArray_Check(op) (Py_TYPE(op) == _lazyarray_type)
#define PyLazyArray_CheckExact(op) (Py_TYPE(op) == _lazyarray_type)

#define PyLazyArray_GET_SIZE(op)  Py_SIZE(op)
#define PyLazyArray_GET_TYPE(op)  ATOMstorage(getColumnType(LAZYARRAY_BAT(op)->T->type))

void lazyarray_init(void);

#endif /* _LAZYARRAY_ */
