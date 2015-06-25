/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains a function PyByteArray_Override that overrides 
 */

#ifndef _BYTEARRAY_OVERRIDE_
#define _BYTEARRAY_OVERRIDE_

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

//! Create a PyByteArray from a reference to a string. The string is not actually copied, so make sure not to release the string before Python is done with it.
PyObject *PyByteArray_FromString(char *str);
//! Override a number of PyByteArray functions so PyByteArrayObjects created through PyByteArray_FromString() cannot be modified through Python functions.
void PyByteArray_Override(void);

#endif /* _BYTEARRAY_OVERRIDE_ */
