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

bool PyType_IsInteger(int type)
{
    switch (type)
    {
        case NPY_BOOL:
        case NPY_BYTE: 
        case NPY_SHORT: 
        case NPY_INT: 
        case NPY_LONG: 
        case NPY_LONGLONG: 
        case NPY_UBYTE:
        case NPY_USHORT: 
        case NPY_UINT:
        case NPY_ULONG: 
        case NPY_ULONGLONG: return true;
        default: return false;
    }
}

bool PyType_IsFloat(int type)
{
    switch (type)
    {
        case NPY_FLOAT16: 
        case NPY_FLOAT: return true;
        default: return false;
    }
}

bool PyType_IsDouble(int type)
{
    switch (type)
    {
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE: return true;
        default: return false;
    }
}

char *PyType_Format(int type)
{
    switch (type)
    {
        case NPY_BOOL: return "BOOL";
        case NPY_BYTE: return "BYTE";
        case NPY_SHORT: return "SHORT";
        case NPY_INT: return "INT";
        case NPY_LONG: return "LONG";
        case NPY_LONGLONG: return "LONG LONG";
        case NPY_UBYTE: return "UNSIGNED BYTE";
        case NPY_USHORT: return "UNSIGNED SHORT";
        case NPY_UINT: return "UNSIGNED INT";
        case NPY_ULONG: return "UNSIGNED LONG";
        case NPY_ULONGLONG: return "UNSIGNED LONG LONG";
        case NPY_FLOAT16: return "HALF-FLOAT (FLOAT16)";
        case NPY_FLOAT: return "FLOAT";
        case NPY_DOUBLE: return "DOUBLE";
        case NPY_LONGDOUBLE: return "LONG DOUBLE";
        case NPY_COMPLEX64: return "COMPLEX FLOAT";
        case NPY_COMPLEX128: return "COMPLEX DOUBLE";
        case NPY_CLONGDOUBLE: return "COMPLEX LONG DOUBLE";
        case NPY_DATETIME: return "DATETIME";
        case NPY_TIMEDELTA: return "TIMEDELTA";
        case NPY_STRING: return "STRING";
        case NPY_UNICODE: return "UNICODE STRING";
        case NPY_OBJECT: return "PYTHON OBJECT";
        case NPY_VOID: return "VOID";
        default: return "UNKNOWN";
    }
}

char *BatType_Format(int type)
{
    switch (type)
    {
        case TYPE_bit: return "BIT";
        case TYPE_bte: return "BYTE";
        case TYPE_sht: return "SHORT";
        case TYPE_int: return "INT";
        case TYPE_lng: return "LONG";
        case TYPE_flt: return "FLOAT";
        case TYPE_dbl: return "DOUBLE";
        case TYPE_str: return "STRING";
        case TYPE_hge: return "HUGE";
        case TYPE_oid: return "OID";
        default: return "UNKNOWN";
    }
}

int PyType_ToBat(int type)
{
    switch (type)
    {
        case NPY_BOOL: return TYPE_bit;
        case NPY_BYTE: return TYPE_bte;
        case NPY_SHORT: return TYPE_sht;
        case NPY_INT: return TYPE_int;
        case NPY_LONG: 
        case NPY_LONGLONG: return TYPE_lng;
        case NPY_UBYTE: return TYPE_bte;
        case NPY_USHORT: return TYPE_sht;
        case NPY_UINT: return TYPE_int;
        case NPY_ULONG: 
        case NPY_ULONGLONG: return TYPE_lng;
        case NPY_FLOAT16: 
        case NPY_FLOAT: return TYPE_flt;
        case NPY_DOUBLE: 
        case NPY_LONGDOUBLE: return TYPE_dbl;
        case NPY_STRING: return TYPE_str;
        case NPY_UNICODE: return TYPE_str;
        default: return TYPE_void;
    }
}

int BatType_ToPyType(int type)
{
    switch (type)
    {
        case TYPE_bit: return NPY_BOOL;
        case TYPE_bte: return NPY_INT8;
        case TYPE_sht: return NPY_INT16;
        case TYPE_int: return NPY_INT32;
        case TYPE_lng: return NPY_INT64;
        case TYPE_flt: return NPY_FLOAT32;
        case TYPE_dbl: return NPY_FLOAT64;
        case TYPE_str: return NPY_UNICODE;
        case TYPE_hge: return NPY_STRING;
        case TYPE_oid: return NPY_INT32;
        default: return NPY_STRING;
    }
}

//Returns true if the type of [object] is a scalar (i.e. numeric scalar or string, basically "not an array but a single value")
bool PyType_IsPyScalar(PyObject *object)
{
    PyArray_Descr *descr;

    if (object == NULL) return false;
    if (PyList_Check(object)) return false;
    if (PyObject_HasAttrString(object, "mask")) return false;

    descr = PyArray_DescrFromScalar(object);
    if (descr == NULL) return false;
    if (descr->type_num != NPY_OBJECT) return true; //check if the object is a numpy scalar
    if (PyInt_Check(object) || PyFloat_Check(object) || PyLong_Check(object) || PyString_Check(object) || PyBool_Check(object) || PyUnicode_Check(object)) return true;

    return false;
}

bool PyType_IsPandasDataFrame(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<class 'pandas.core.frame.DataFrame'>") == 0;
    Py_DECREF(str);
    return ret;
}

bool PyType_IsNumpyArray(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<type 'numpy.ndarray'>") == 0;
    Py_DECREF(str);
    return ret;
}

bool PyType_IsNumpyMaskedArray(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<class 'numpy.ma.core.MaskedArray'>") == 0;
    Py_DECREF(str);
    return ret;
}

#endif /* _PYTYPE_LIB_ */
