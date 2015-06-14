/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains a number of helper functions for Python and Numpy types
 */

#ifndef _PYTYPE_LIB_
#define _PYTYPE_LIB_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

//! Returns true if a NPY_#type is an integral type, and false otherwise
bool PyType_IsInteger(int);
//! Returns true if a NPY_#type is a float type, and false otherwise
bool PyType_IsFloat(int);
//! Returns true if a NPY_#type is a double type, and false otherwise
bool PyType_IsDouble(int);
//! Formats NPY_#type as a String (so NPY_INT => "INT"), for usage in error reporting and warnings
char *PyType_Format(int);
//! Returns true if a PyObject is a scalar type ('scalars' in this context means numeric or string types)
bool PyType_IsPyScalar(PyObject *object);
//! Returns true if the PyObject is of type numpy.ndarray, and false otherwise
bool PyType_IsNumpyArray(PyObject *object);
//! Returns true if the PyObject is of type numpy.ma.core.MaskedArray, and false otherwise
bool PyType_IsNumpyMaskedArray(PyObject *object);
//! Returns true if the PyObject is of type pandas.core.frame.DataFrame, and false otherwise
bool PyType_IsPandasDataFrame(PyObject *object);

char *BatType_Format(int);

int PyType_ToBat(int);
int BatType_ToPyType(int);

#endif /* _PYTYPE_LIB_ */
