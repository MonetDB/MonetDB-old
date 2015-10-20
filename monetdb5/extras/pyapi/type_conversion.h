/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains a number of helper functions for converting between types, mainly used to convert from an object from a numpy array to the type requested by the BAT.
 */

#ifndef _TYPE_CONVERSION_
#define _TYPE_CONVERSION_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "monetdb_config.h"
#include "mal.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "gdk_utils.h"
#include "gdk.h"

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>



//! Copies the string of size up to max_size from the source to the destination, returns FALSE if "source" is not a legal ASCII string (i.e. a character is >= 128)
bool string_copy(char * source, char* dest, size_t max_size);
//! Converts a base-10 string to a dbl value
bool s_to_dbl(char *ptr, size_t size, dbl *value);
//! Converts a base-10 string to a lng value
bool s_to_lng(char *ptr, size_t size, lng *value);
//! Converts a base-10 utf32-encoded string to a lng value
bool utf32_to_lng(Py_UNICODE *utf32, size_t maxsize, lng *value);
//! Converts a base-10 utf32-encoded string to a dbl value
bool utf32_to_dbl(Py_UNICODE *utf32, size_t maxsize, dbl *value);
//! Converts a base-10 utf32-encoded string to a hge value
bool utf32_to_hge(Py_UNICODE *utf32, size_t maxsize, hge *value);
//! Converts a PyObject to a dbl value
bool py_to_dbl(PyObject *ptr, dbl *value);
//! Converts a PyObject to a lng value
bool py_to_lng(PyObject *ptr, lng *value);

#ifdef HAVE_HGE
//! Converts a hge to a string and writes it into the string "str"
int hge_to_string(char *str, hge );
//! Converts a base-10 string to a hge value
bool s_to_hge(char *ptr, size_t size, hge *value);
//! Converts a PyObject to a hge value
bool py_to_hge(PyObject *ptr, hge *value);
//! Create a PyLongObject from a hge integer
PyObject *PyLong_FromHge(hge h);

void printhuge(hge h);
#endif


//using macros, create a number of str_to_<type>, unicode_to_<type> and pyobject_to_<type> functions (we are Java now)
#define CONVERSION_FUNCTION_HEADER_FACTORY(tpe)          \
    bool str_to_##tpe(void *ptr, size_t size, tpe *value);          \
    bool unicode_to_##tpe(void *ptr, size_t size, tpe *value);                  \
    bool pyobject_to_##tpe(void *ptr, size_t size, tpe *value);                  \

CONVERSION_FUNCTION_HEADER_FACTORY(bit)
CONVERSION_FUNCTION_HEADER_FACTORY(sht)
CONVERSION_FUNCTION_HEADER_FACTORY(int)
CONVERSION_FUNCTION_HEADER_FACTORY(lng)
CONVERSION_FUNCTION_HEADER_FACTORY(flt)
CONVERSION_FUNCTION_HEADER_FACTORY(dbl)
#ifdef HAVE_HGE
CONVERSION_FUNCTION_HEADER_FACTORY(hge)
#endif

#endif /* _TYPE_CONVERSION_ */
