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

#include "pyapi.h"

#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>


// This describes return values, used in multiprocessing to tell the main process the size of the shared memory to allocate
struct _ReturnBatDescr
{
    int npy_type;                        //npy type 
    size_t element_size;                 //element size in bytes
    size_t bat_count;                    //number of elements in bat
    size_t bat_size;                     //bat size in bytes
    size_t bat_start;                    //start position of bat
    bool has_mask;                       //if the return value has a mask or not
#ifdef _PYAPI_TESTING_
    unsigned long long peak_memory_usage;            //peak memory usage of the thread in bytes, used for testing
#endif
};
#define ReturnBatDescr struct _ReturnBatDescr

struct _PyReturn{
    PyObject *numpy_array;              //PyArrayObject* with data (can be NULL, as long as array_data is set)
    PyObject *numpy_mask;               //PyArrayObject* with mask (NULL if there is no mask)
    void *array_data;                   //void* pointer to data
    bool *mask_data;                    //bool* pointer to mask data
    size_t count;                       //amount of return elements
    size_t memory_size;                 //memory size of each element
    int result_type;                    //result type as NPY_<TYPE>
    bool multidimensional;              //whether or not the result is multidimensional
};
#define PyReturn struct _PyReturn

struct _PyInput{
    void *dataptr;                      //pointer to input data
    BAT *bat;                           //pointer to input BAT
    int bat_type;                       //BAT type as TYPE_<type>
    size_t count;                       //amount of elements in BAT
    bool scalar;                        //True if the input is a scalar (in this case, BAT* is NULL)
};
#define PyInput struct _PyInput

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
//! Create a Numpy Array Object from a PyInput structure
PyObject *PyArrayObject_FromBAT(PyInput *input_bat, size_t start, size_t end, char **return_message);

char *BatType_Format(int);

int PyType_ToBat(int);
int BatType_ToPyType(int);

#define bte_TO_PYSCALAR(value) PyInt_FromLong((lng)value)
#define bit_TO_PYSCALAR(value) PyInt_FromLong((lng)value)
#define sht_TO_PYSCALAR(value) PyInt_FromLong((lng)value)
#define int_TO_PYSCALAR(value) PyInt_FromLong((lng)value)
#define lng_TO_PYSCALAR(value) PyLong_FromLong(value)
#define flt_TO_PYSCALAR(value) PyFloat_FromDouble(value)
#define dbl_TO_PYSCALAR(value) PyFloat_FromDouble(value)

// A simple #define that converts a numeric TYPE_<mtpe> value to a Python scalar
#define SCALAR_TO_PYSCALAR(mtpe, value) mtpe##_TO_PYSCALAR(value)

#endif /* _PYTYPE_LIB_ */
