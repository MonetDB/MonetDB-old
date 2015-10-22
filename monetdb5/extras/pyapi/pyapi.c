/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "mal.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "gdk_utils.h"
#include "gdk.h"
#include "sql_catalog.h"
#include "pyapi.h"
#include "connection.h"
#include "mal_dataflow.h"

// Python library
#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

// Numpy Library
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __INTEL_COMPILER
// Intel compiler complains about trailing comma's in numpy source code,
#pragma warning(disable:271)
#endif
#include <numpy/arrayobject.h>

#include "unicode.h"
#include "pytypes.h"
#include "shared_memory.h"
#include "bytearray.h"
#include "type_conversion.h"
#include "formatinput.h"
#include "benchmark.h"
#include "lazyarray.h"

#ifndef WIN32
// These libraries are used for PYTHON_MAP operations on Linux [to start new processes and wait on them]
#include <sys/types.h>
#include <sys/wait.h>
#endif

const char* pyapi_enableflag = "embedded_py";
const char* verbose_enableflag = "enable_pyverbose";
const char* warning_enableflag = "enable_pywarnings";
const char* debug_enableflag = "enable_pydebug";
#ifdef _PYAPI_TESTING_
const char* zerocopyinput_disableflag = "disable_pyzerocopyinput";
const char* zerocopyoutput_disableflag = "disable_pyzerocopyoutput";
const char* numpy_string_array_disableflag = "disable_numpystringarray";
const char* alwaysunicode_enableflag = "enable_alwaysunicode";
const char* lazyarray_enableflag = "enable_lazyarray";
const char* oldnullmask_enableflag = "enable_oldnullmask";
const char* bytearray_enableflag = "enable_bytearray";
const char* benchmark_output_flag = "pyapi_benchmark_output";
const char* disable_malloc_tracking = "disable_malloc_tracking";
static bool option_zerocopyinput = true;
static bool option_zerocopyoutput  = true;
static bool option_numpy_string_array = true;
static bool option_bytearray = false;
static bool option_lazyarray = false;
static bool option_oldnullmask = false;
static bool option_alwaysunicode = false;
static bool option_disablemalloctracking = false;
static char *benchmark_output = NULL;
#endif
#ifdef _PYAPI_VERBOSE_
static bool option_verbose;
#endif
#ifdef _PYAPI_DEBUG_
static bool option_debug;
#endif
#ifdef _PYAPI_WARNINGS_
static bool option_warning;
#endif

static PyObject *marshal_module = NULL;
PyObject *marshal_loads = NULL;

const int utf8string_minlength = 256;

int PyAPIEnabled(void) {
    return (GDKgetenv_istrue(pyapi_enableflag)
            || GDKgetenv_isyes(pyapi_enableflag));
}

static MT_Lock pyapiLock;
static int pyapiInitialized = FALSE;

static bool python_call_active = false;

#ifdef _PYAPI_TESTING_
// This #define converts a BAT 'bat' of BAT type 'TYPE_mtpe' to a Numpy array of type 'nptpe'
// This only works with numeric types (bit, byte, int, long, float, double), strings are handled separately
// if _PYAPI_TESTING_ is enabled, and option_zerocopyinput is set to FALSE, the BAT is copied. Otherwise the internal BAT pointer is passed to the numpy array (zero copy)
#define BAT_TO_NP(bat, mtpe, nptpe)                                                                                                 \
        if (!option_zerocopyinput) {                                                                                                \
            mtpe *array;                                                                                                            \
            vararray = PyArray_Zeros(1, (npy_intp[1]) {(t_end - t_start)}, PyArray_DescrFromType(nptpe), 0);                        \
            array = PyArray_DATA((PyArrayObject*)vararray);                                                                         \
            for(j = t_start; j < t_end; j++) {                                                                                      \
                array[j - t_start] = ((mtpe*) Tloc(bat, BUNfirst(bat)))[j];                                                         \
            }                                                                                                                       \
        } else {                                                                                                                    \
            vararray = PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {(t_end-t_start)},                                               \
            nptpe, NULL, &((mtpe*) Tloc(bat, BUNfirst(bat)))[t_start], 0,                                                           \
            NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);                                                                        \
        }
#else
#define BAT_TO_NP(bat, mtpe, nptpe)                                                                                                 \
        vararray = PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {(t_end-t_start)},                                                   \
            nptpe, NULL, &((mtpe*) Tloc(bat, BUNfirst(bat)))[t_start], 0,                                                           \
            NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);
#endif

// This #define creates a new BAT with the internal data and mask from a Numpy array, without copying the data
// 'bat' is a BAT* pointer, which will contain the new BAT. TYPE_'mtpe' is the BAT type, and 'batstore' is the heap storage type of the BAT (this should be STORE_CMEM or STORE_SHARED)
#define CREATE_BAT_ZEROCOPY(bat, mtpe, batstore) {                                                                      \
        bat = BATnew(TYPE_void, TYPE_##mtpe, 0, TRANSIENT);                                                             \
        BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                                   \
        bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;                                                           \
        /*Change nil values to the proper values, if they exist*/                                                       \
        if (mask != NULL)                                                                                               \
        {                                                                                                               \
            for (iu = 0; iu < ret->count; iu++)                                                                         \
            {                                                                                                           \
                if (mask[index_offset * ret->count + iu] == TRUE)                                                       \
                {                                                                                                       \
                    (*(mtpe*)(&data[(index_offset * ret->count + iu) * ret->memory_size])) = mtpe##_nil;                \
                    bat->T->nil = 1;                                                                                    \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        bat->T->nonil = 1 - bat->T->nil;                                                                                \
        /*When we create a BAT a small part of memory is allocated, free it*/                                           \
        GDKfree(bat->T->heap.base);                                                                                     \
                                                                                                                        \
        bat->T->heap.base = &data[(index_offset * ret->count) * ret->memory_size];                                      \
        bat->T->heap.size = ret->count * ret->memory_size;                                                              \
        bat->T->heap.free = bat->T->heap.size;  /*There are no free places in the array*/                               \
        /*If index_offset > 0, we are mapping part of a multidimensional array.*/                                       \
        /*The entire array will be cleared when the part with index_offset=0 is freed*/                                 \
        /*So we set this part of the mapping to 'NOWN'*/                                                                \
        if (index_offset > 0) bat->T->heap.storage = STORE_NOWN;                                                        \
        else bat->T->heap.storage = batstore;                                                                           \
        bat->T->heap.newstorage = STORE_MEM;                                                                            \
        bat->S->count = ret->count;                                                                                     \
        bat->S->capacity = ret->count;                                                                                  \
        bat->S->copiedtodisk = false;                                                                                   \
                                                                                                                        \
        /*Take over the data from the numpy array*/                                                                     \
        if (ret->numpy_array != NULL) PyArray_CLEARFLAGS((PyArrayObject*)ret->numpy_array, NPY_ARRAY_OWNDATA);          \
    }

// This #define converts a Numpy Array to a BAT by copying the internal data to the BAT. It assumes the BAT 'bat' is already created with the proper size.
// This should only be used with integer data that can be cast. It assumes the Numpy Array has an internal array of type 'mtpe_from', and the BAT has an internal array of type 'mtpe_to'.
// it then does the cast by simply doing BAT[i] = (mtpe_to) ((mtpe_from*)NUMPY_ARRAY[i]), which only works if both mtpe_to and mtpe_from are integers
#define NP_COL_BAT_LOOP(bat, mtpe_to, mtpe_from) {                                                                                               \
    if (mask == NULL)                                                                                                                            \
    {                                                                                                                                            \
        for (iu = 0; iu < ret->count; iu++)                                                                                                      \
        {                                                                                                                                        \
            ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = (mtpe_to)(*(mtpe_from*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));    \
        }                                                                                                                                        \
    }                                                                                                                                            \
    else                                                                                                                                         \
    {                                                                                                                                            \
        for (iu = 0; iu < ret->count; iu++)                                                                                                      \
        {                                                                                                                                        \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                    \
            {                                                                                                                                    \
                bat->T->nil = 1;                                                                                                                 \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = mtpe_to##_nil;                                                                       \
            }                                                                                                                                    \
            else                                                                                                                                 \
            {                                                                                                                                    \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = (mtpe_to)(*(mtpe_from*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));\
            }                                                                                                                                    \
        }                                                                                                                                        \
    } }

// This #define converts a Numpy Array to a BAT by copying the internal data to the BAT. It converts the data from the Numpy Array to the BAT using a function
// This function has to have the prototype 'bool function(void *data, size_t memory_size, mtpe_to *resulting_value)', and either return False (if conversion fails)
//  or write the value into the 'resulting_value' pointer. This is used convertring strings/unicodes/python objects to numeric values.
#define NP_COL_BAT_LOOP_FUNC(bat, mtpe_to, func, ptrtpe) {                                                                                                    \
    mtpe_to value;                                                                                                                                    \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (!func((ptrtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                          \
            {                                                                                                                                         \
                msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to);     \
                goto wrapup;                                                                                                                          \
            }                                                                                                                                         \
            ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                        \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    else                                                                                                                                              \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                         \
            {                                                                                                                                         \
                bat->T->nil = 1;                                                                                                                      \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = mtpe_to##_nil;                                                                            \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                if (!func((ptrtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                      \
                {                                                                                                                                     \
                    msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to); \
                    goto wrapup;                                                                                                                      \
                }                                                                                                                                     \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                    \
            }                                                                                                                                         \
        }                                                                                                                                             \
    } }


// This #define is for converting a numeric numpy array into a string BAT. 'conv' is a function that turns a numeric value of type 'mtpe' to a char* array.
#define NP_COL_BAT_STR_LOOP(bat, mtpe, fmt)                                                                                                           \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            snprintf(utf8_string, utf8string_minlength, fmt, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                   \
            BUNappend(bat, utf8_string, FALSE);                                                                                                       \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    else                                                                                                                                              \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                         \
            {                                                                                                                                         \
                bat->T->nil = 1;                                                                                                                      \
                BUNappend(b, str_nil, FALSE);                                                                                                         \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                snprintf(utf8_string, utf8string_minlength, fmt, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));               \
                BUNappend(bat, utf8_string, FALSE);                                                                                                   \
            }                                                                                                                                         \
        }                                                                                                                                             \
    }

// This is here so we can remove the option_zerocopyoutput from the zero copy conditionals if testing is disabled
#ifdef _PYAPI_TESTING_
#define ZEROCOPY_OUTPUT option_zerocopyoutput &&
#else
#define ZEROCOPY_OUTPUT
#endif

// This very big #define combines all the previous #defines for one big #define that is responsible for converting a Numpy array (described in the PyReturn object 'ret')
// to a BAT of type 'mtpe'. This should only be used for numeric BATs (but can be used for any Numpy Array). The resulting BAT will be stored in 'bat'.
#define NP_CREATE_BAT(bat, mtpe) {                                                                                                                             \
        bool *mask = NULL;                                                                                                                                     \
        char *data = NULL;                                                                                                                                     \
        if (ret->mask_data != NULL)                                                                                                                            \
        {                                                                                                                                                      \
            mask = (bool*) ret->mask_data;                                                                                                                     \
        }                                                                                                                                                      \
        if (ret->array_data == NULL)                                                                                                                           \
        {                                                                                                                                                      \
            msg = createException(MAL, "pyapi.eval", "No return value stored in the structure.\n");                                                            \
            goto wrapup;                                                                                                                                       \
        }                                                                                                                                                      \
        data = (char*) ret->array_data;                                                                                                                        \
        if (ZEROCOPY_OUTPUT ret->count > 0 && TYPE_##mtpe == PyType_ToBat(ret->result_type) && (ret->count * ret->memory_size < BUN_MAX) &&                    \
            (ret->numpy_array == NULL || PyArray_FLAGS((PyArrayObject*)ret->numpy_array) & NPY_ARRAY_OWNDATA))                                                 \
        {                                                                                                                                                      \
            /*We can only create a direct map if the numpy array type and target BAT type*/                                                                    \
            /*are identical, otherwise we have to do a conversion.*/                                                                                           \
            assert(ret->array_data != NULL);                                                                                                                   \
            if (ret->numpy_array == NULL)                                                                                                                      \
            {                                                                                                                                                  \
                /*shared memory return*/                                                                                                                       \
                VERBOSE_MESSAGE("- Zero copy (shared memory)!\n");                                                                                             \
                CREATE_BAT_ZEROCOPY(bat, mtpe, STORE_SHARED);                                                                                                  \
                ret->array_data = NULL;                                                                                                                        \
            }                                                                                                                                                  \
            else                                                                                                                                               \
            {                                                                                                                                                  \
                VERBOSE_MESSAGE("- Zero copy!\n");                                                                                                             \
                CREATE_BAT_ZEROCOPY(bat, mtpe, STORE_CMEM);                                                                                                    \
            }                                                                                                                                                  \
        }                                                                                                                                                      \
        else                                                                                                                                                   \
        {                                                                                                                                                      \
            bat = BATnew(TYPE_void, TYPE_##mtpe, ret->count, TRANSIENT);                                                                                       \
            BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                                                                      \
            if (TYPE_##mtpe != TYPE_hge  && TYPE_##mtpe != PyType_ToBat(ret->result_type)) WARNING_MESSAGE("!PERFORMANCE WARNING: You are returning a Numpy Array of type %s, which has to be converted to a BAT of type %s. If you return a Numpy\
Array of type %s no copying will be needed.\n", PyType_Format(ret->result_type), BatType_Format(TYPE_##mtpe), PyType_Format(BatType_ToPyType(TYPE_##mtpe)));   \
            bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;                                                                                              \
            switch(ret->result_type)                                                                                                                           \
            {                                                                                                                                                  \
                case NPY_BOOL:       NP_COL_BAT_LOOP(bat, mtpe, bit); break;                                                                                   \
                case NPY_BYTE:       NP_COL_BAT_LOOP(bat, mtpe, bte); break;                                                                                   \
                case NPY_SHORT:      NP_COL_BAT_LOOP(bat, mtpe, sht); break;                                                                                   \
                case NPY_INT:        NP_COL_BAT_LOOP(bat, mtpe, int); break;                                                                                   \
                case NPY_LONG:                                                                                                                                 \
                case NPY_LONGLONG:   NP_COL_BAT_LOOP(bat, mtpe, lng); break;                                                                                   \
                case NPY_UBYTE:      NP_COL_BAT_LOOP(bat, mtpe, unsigned char); break;                                                                         \
                case NPY_USHORT:     NP_COL_BAT_LOOP(bat, mtpe, unsigned short); break;                                                                        \
                case NPY_UINT:       NP_COL_BAT_LOOP(bat, mtpe, unsigned int); break;                                                                          \
                case NPY_ULONG:                                                                                                                                \
                case NPY_ULONGLONG:  NP_COL_BAT_LOOP(bat, mtpe, unsigned long); break;                                                                         \
                case NPY_FLOAT16:                                                                                                                              \
                case NPY_FLOAT:      NP_COL_BAT_LOOP(bat, mtpe, flt); break;                                                                                   \
                case NPY_DOUBLE:                                                                                                                               \
                case NPY_LONGDOUBLE: NP_COL_BAT_LOOP(bat, mtpe, dbl); break;                                                                                   \
                case NPY_STRING:     NP_COL_BAT_LOOP_FUNC(bat, mtpe, str_to_##mtpe, char); break;                                                                    \
                case NPY_UNICODE:    NP_COL_BAT_LOOP_FUNC(bat, mtpe, unicode_to_##mtpe, Py_UNICODE); break;                                                                \
                case NPY_OBJECT:     NP_COL_BAT_LOOP_FUNC(bat, mtpe, pyobject_to_##mtpe, PyObject*); break;                                                               \
                default:                                                                                                                                       \
                    msg = createException(MAL, "pyapi.eval", "Unrecognized type. Could not convert to %s.\n", BatType_Format(TYPE_##mtpe));                    \
                    goto wrapup;                                                                                                                               \
            }                                                                                                                                                  \
            bat->T->nonil = 1 - bat->T->nil;                                                                                                                   \
            BATsetcount(bat, ret->count);                                                                                                                      \
            BATsettrivprop(bat);                                                                                                                               \
        }                                                                                                                                                      \
    }

str
PyAPIeval(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped, bit mapped);

str
PyAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    return PyAPIeval(cntxt, mb, stk, pci, 0, 0);
}

str
PyAPIevalStdMap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    return PyAPIeval(cntxt, mb, stk, pci, 0, 1);
}

str
PyAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    return PyAPIeval(cntxt, mb, stk, pci, 1, 0);
}

str
PyAPIevalAggrMap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
    return PyAPIeval(cntxt, mb, stk, pci, 1, 1);
}

static char *PyError_CreateException(char *error_text, char *pycall);

//! The main PyAPI function, this function does everything PyAPI related
//! It takes as argument a bunch of input BATs, a python function, and outputs a number of BATs
//! This function follows the following pipeline
//! [PARSE_CODE] Step 1: It parses the Python source code and corrects any wrong indentation, or converts the source code into a PyCodeObject if the source code is encoded as such
//! [CONVERT_BAT] Step 2: It converts the input BATs into Numpy Arrays
//! [EXECUTE_CODE] Step 3: It executes the Python code using the Numpy arrays as arguments
//! [RETURN_VALUES] Step 4: It collects the return values and converts them back into BATs
//! If 'mapped' is set to True, it will fork a separate process at [FORK_PROCESS] that executes Step 1-3, the process will then write the return values into Shared memory [SHARED_MEMORY] and exit, then Step 4 is executed by the main process
str PyAPIeval(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped, bit mapped) {
    sql_func * sqlfun = *(sql_func**) getArgReference(stk, pci, pci->retc);
    str exprStr = *getArgReference_str(stk, pci, pci->retc + 1);

    int i = 1, ai = 0;
    char* pycall = NULL;
    str *args;
    char *msg = MAL_SUCCEED;
    BAT *b = NULL;
    node * argnode;
    int seengrp = FALSE;
    PyObject *pArgs = NULL, *pColumns = NULL, *pColumnTypes = NULL, *pConnection, *pResult = NULL; // this is going to be the parameter tuple
    PyObject *code_object = NULL;
    PyReturn *pyreturn_values = NULL;
    PyInput *pyinput_values = NULL;
    int seqbase = 0;
#ifndef WIN32
    int shm_id = -1;
    size_t memory_size = 0;
    bool child_process = false;
#endif
#ifdef _PYAPI_TESTING_
    time_storage start_time, end_time;
    bool disable_testing = false;
    unsigned long long peak_memory_usage = 0;
#endif
    PyGILState_STATE gstate = PyGILState_LOCKED;
    bit varres = sqlfun ? sqlfun->varres : 0;
    int retcols = !varres ? pci->retc : -1;
    bool holds_gil = !mapped;

    (void) cntxt;

    if (!PyAPIEnabled()) {
        throw(MAL, "pyapi.eval",
              "Embedded Python has not been enabled. Start server with --set %s=true",
              pyapi_enableflag);
    }

#ifdef _PYAPI_TESTING_
    {
        char *string = GDKzalloc((strlen(exprStr) + 1) * sizeof(char));
        strcpy(string, exprStr);
        exprStr = string;
    }
    // If the source starts with 'NOTEST' we disable testing, kind of a band-aid solution but they don't pay me enough for anything better
    if (exprStr[1] == 'N' && exprStr[2] == 'O' && exprStr[3] == 'T' && exprStr[4] == 'E' && exprStr[5] == 'S' && exprStr[6] == 'T') {
        VERBOSE_MESSAGE("Disable testing!\n");
        for(iu = 7; iu < strlen(exprStr); iu++) {
            exprStr[iu - 6] = exprStr[iu];
        }
        exprStr[iu - 6] = '\0';
        disable_testing = true;
    }

    if (benchmark_output != NULL) {
        reset_hook();
        if (!disable_testing && !option_disablemalloctracking && !mapped) init_hook();
        timer(&start_time);
    }
#endif

    VERBOSE_MESSAGE("PyAPI Start\n");

    args = (str*) GDKzalloc(pci->argc * sizeof(str));
    pyreturn_values = GDKzalloc(pci->retc * sizeof(PyReturn));
    if (args == NULL || pyreturn_values == NULL) {
        throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
    }

    if ((pci->argc - (pci->retc + 2)) * sizeof(PyInput) > 0) {
        pyinput_values = GDKzalloc((pci->argc - (pci->retc + 2)) * sizeof(PyInput));

        if (pyinput_values == NULL) {
            GDKfree(args); GDKfree(pyreturn_values);
            throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        }
    }

    // Analyse the SQL_Func structure to get the parameter names
    if (sqlfun != NULL && sqlfun->ops->cnt > 0) {
        int cargs = pci->retc + 2;
        argnode = sqlfun->ops->h;
        while (argnode) {
            char* argname = ((sql_arg*) argnode->data)->name;
            args[cargs] = GDKstrdup(argname);
            cargs++;
            argnode = argnode->next;
        }
    }

    // We name all the unknown arguments, if grouping is enabled there is one unknown argument that is the group variable, we name this 'aggr_group'
    for (i = pci->retc + 2; i < pci->argc; i++) {
        if (args[i] == NULL) {
            if (!seengrp && grouped) {
                args[i] = GDKstrdup("aggr_group");
                seengrp = TRUE;
            } else {
                char argbuf[64];
                snprintf(argbuf, sizeof(argbuf), "arg%i", i - pci->retc - 1);
                args[i] = GDKstrdup(argbuf);
            }
        }
    }

    // Construct PyInput objects, we do this before any multiprocessing because there is some locking going on in there, and locking + forking = bad idea (a thread can fork while another process is in the lock, which means we can get stuck permanently)
    for (i = pci->retc + 2; i < pci->argc; i++)
    {
        PyInput *inp = &pyinput_values[i - (pci->retc + 2)];
        if (!isaBatType(getArgType(mb,pci,i))) {
            inp->scalar = true;
            inp->bat_type = getArgType(mb, pci, i);
            inp->count = 1;
            if (inp->bat_type == TYPE_str) {
                inp->dataptr = getArgReference_str(stk, pci, i);
            }
            else {
                inp->dataptr = getArgReference(stk, pci, i);
            }
        }
        else
        {
            b = BATdescriptor(*getArgReference_bat(stk, pci, i));
            if (b == NULL) {
                msg = createException(MAL, "pyapi.eval", "The BAT passed to the function (argument #%d) is NULL.\n", i - (pci->retc + 2) + 1);
                goto wrapup;
            }
            seqbase = b->H->seq;
            inp->count = BATcount(b);
            inp->bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));
            inp->bat = b;
        }
    }
    if (!mapped) {
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        if (python_call_active) {
            mapped = true;
            holds_gil = false;
        }
        else python_call_active = true;
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
    }

    /*[FORK_PROCESS]*/
    if (mapped)
    {
#ifdef WIN32
        // We haven't implemented multiprocessing on Windows yet
        msg = createException(MAL, "pyapi.eval", "Please visit http://www.linux.com/directory/Distributions to download a Linux distro.\n");
        goto wrapup;
#else
        lng pid;
        char *ptr = NULL;

        //create initial shared memory
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        shm_id = get_unique_shared_memory_id(1 + pci->retc * 2); //we need 1 + pci->retc * 2 shared memory spaces, the first is for the header information, the second pci->retc * 2 is one for each return BAT, and one for each return mask array
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");

        VERBOSE_MESSAGE("Creating multiple processes.\n");

        memory_size = pci->retc * sizeof(ReturnBatDescr); //the memory size for the header files, each process has one per return value

        VERBOSE_MESSAGE("Initializing shared memory.\n");

        assert(memory_size > 0);
        //create the shared memory for the header
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        msg = create_shared_memory(shm_id, memory_size, (void**) &ptr);
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
        if (msg != MAL_SUCCEED)
        {
            goto wrapup;
        }

        VERBOSE_MESSAGE("Waiting to fork.\n");
        //fork
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        VERBOSE_MESSAGE("Start forking.\n");
        if ((pid = fork()) < 0)
        {
            msg = createException(MAL, "pyapi.eval", "Failed to fork process");
            MT_lock_unset(&pyapiLock, "pyapi.evaluate");

            release_shared_memory(ptr);

            goto wrapup;
        }
        else if (pid == 0)
        {
            forked_process = 1;
            GDKnr_threads = 1;
            child_process = true;
#ifdef _PYAPI_TESTING_
            if (!disable_testing && !option_disablemalloctracking && benchmark_output != NULL) { init_hook(); }
#endif
        }

        if (!child_process)
        {
            //main process
            int status;
            bool success = true;
            //wait for child processes
            MT_lock_unset(&pyapiLock, "pyapi.evaluate");

            waitpid(pid, &status, 0);

            if (status != 0)
                success = false;

            if (!success)
            {
                //a child failed, get the error message from the child
                ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[0]);
                char *err_ptr;

                if (descr->bat_size == 0) {
                    msg = createException(MAL, "pyapi.eval", "Failure in child process with unknown error.");
                } else {
                    MT_lock_set(&pyapiLock, "pyapi.evaluate");
                    msg = get_shared_memory(shm_id + 1, descr->bat_size, (void**) &err_ptr);
                    MT_lock_unset(&pyapiLock, "pyapi.evaluate");
                    if (msg == MAL_SUCCEED)
                    {
                        msg = createException(MAL, "pyapi.eval", "%s", err_ptr);
                        release_shared_memory(err_ptr);
                    }
                }

                release_shared_memory(ptr);

                goto wrapup;
            }
            VERBOSE_MESSAGE("Finished waiting for child process.\n");

            //collect return values
            for(i = 0; i < pci->retc; i++)
            {
                PyReturn *ret = &pyreturn_values[i];
                ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[i]);
                size_t total_size = 0;
                bool has_mask = false;
                ret->count = 0;
                ret->memory_size = 0;
                ret->result_type = 0;

                //first get header information
#ifdef _PYAPI_TESTING_
                peak_memory_usage = descr->peak_memory_usage;
#endif
                ret->count += descr->bat_count;
                total_size += descr->bat_size;

                ret->memory_size = descr->element_size;
                ret->result_type = descr->npy_type;
                has_mask = has_mask || descr->has_mask;

                //get the shared memory address for this return value
                VERBOSE_MESSAGE("Parent requesting memory at id %d of size %zu\n", shm_id + (i + 1), total_size);

                assert(total_size > 0);
                MT_lock_set(&pyapiLock, "pyapi.evaluate");
                msg = get_shared_memory(shm_id + (i + 1), total_size, &ret->array_data);
                MT_lock_unset(&pyapiLock, "pyapi.evaluate");

                if (msg != MAL_SUCCEED)
                {
                    release_shared_memory(ptr);
                    goto wrapup;
                }
                ret->mask_data = NULL;
                ret->numpy_array = NULL;
                ret->numpy_mask = NULL;
                ret->multidimensional = FALSE;
                if (has_mask)
                {
                    size_t mask_size = ret->count * sizeof(bool);

                    assert(mask_size > 0);
                    MT_lock_set(&pyapiLock, "pyapi.evaluate");
                    msg = get_shared_memory(shm_id + pci->retc + (i + 1), mask_size, (void**) &ret->mask_data);
                    MT_lock_unset(&pyapiLock, "pyapi.evaluate");

                    if (msg != MAL_SUCCEED)
                    {
                        release_shared_memory(ptr);
                        release_shared_memory(ret->array_data);
                        goto wrapup;
                    }
                }
            }
            msg = MAL_SUCCEED;

            if (ptr != NULL) release_shared_memory(ptr);

            goto returnvalues;
        }
#endif
    }

    //After this point we will execute Python Code, so we need to acquire the GIL
    gstate = PyGILState_Ensure();

    /*[PARSE_CODE]*/
    VERBOSE_MESSAGE("Formatting python code.\n");
    pycall = FormatCode(exprStr, args, pci->argc, 4, &code_object, &msg);
    if (pycall == NULL && code_object == NULL) {
        if (msg == NULL) { msg = createException(MAL, "pyapi.eval", "Error while parsing Python code."); }
        goto wrapup;
    }

    /*[CONVERT_BAT]*/
    VERBOSE_MESSAGE("Loading data from the database into Python.\n");

    // Now we will do the input handling (aka converting the input BATs to numpy arrays)
    // We will put the python arrays in a PyTuple object, we will use this PyTuple object as the set of arguments to call the Python function
    pArgs = PyTuple_New(pci->argc - (pci->retc + 2) + (code_object == NULL ? 3 : 0));
    pColumns = PyDict_New();
    pColumnTypes = PyDict_New();
    pConnection = Py_Connection_Create(cntxt);

    // Now we will loop over the input BATs and convert them to python objects
    for (i = pci->retc + 2; i < pci->argc; i++) {
        PyObject *result_array;
         // t_start and t_end hold the part of the BAT we will convert to a Numpy array, by default these hold the entire BAT [0 - BATcount(b)]
        size_t t_start = 0, t_end = pyinput_values[i - (pci->retc + 2)].count;

        // There are two possibilities, either the input is a BAT, or the input is a scalar
        // If the input is a scalar we will convert it to a python scalar
        // If the input is a BAT, we will convert it to a numpy array
        if (pyinput_values[i - (pci->retc + 2)].scalar) {
            result_array = PyArrayObject_FromScalar(&pyinput_values[i - (pci->retc + 2)], &msg);
        } else {
#ifdef _PYAPI_TESTING_
            if (option_lazyarray) {
                result_array = PyLazyArray_FromBAT(pyinput_values[i - (pci->retc + 2)].bat);
            } else
#endif
            {
                result_array = PyMaskedArray_FromBAT(&pyinput_values[i - (pci->retc + 2)], t_start, t_end, &msg);
            }
        }
        if (result_array == NULL) {
            if (msg == MAL_SUCCEED) {
                msg = createException(MAL, "pyapi.eval", "Failed to create Numpy Array from BAT.");
            }
            goto wrapup;
        }
        if (code_object == NULL) {
            PyObject *arg_type = PyString_FromString(BatType_Format(pyinput_values[i - (pci->retc + 2)].bat_type));
            PyDict_SetItemString(pColumns, args[i], result_array);
            PyDict_SetItemString(pColumnTypes, args[i], arg_type);
            Py_DECREF(arg_type);
        }
        PyTuple_SetItem(pArgs, ai++, result_array);
    }
    if (code_object == NULL) {
        PyTuple_SetItem(pArgs, ai++, pColumns);
        PyTuple_SetItem(pArgs, ai++, pColumnTypes);
        PyTuple_SetItem(pArgs, ai++, pConnection);
    }

    /*[EXECUTE_CODE]*/
    VERBOSE_MESSAGE("Executing python code.\n");

    // Now it is time to actually execute the python code
    {
        PyObject *pFunc, *pModule, *v, *d;

        // First we will load the main module, this is required
        pModule = PyImport_AddModule("__main__");
        if (!pModule) {
            msg = PyError_CreateException("Failed to load module", NULL);
            goto wrapup;
        }

        // Now we will add the UDF to the main module
        d = PyModule_GetDict(pModule);
        if (code_object == NULL) {
            v = PyRun_StringFlags(pycall, Py_file_input, d, d, NULL);
            if (v == NULL) {
                msg = PyError_CreateException("Could not parse Python code", pycall);
                goto wrapup;
            }
            Py_DECREF(v);

            // Now we need to obtain a pointer to the function, the function is called "pyfun"
            pFunc = PyObject_GetAttrString(pModule, "pyfun");
            if (!pFunc || !PyCallable_Check(pFunc)) {
                msg = PyError_CreateException("Failed to load function", NULL);
                goto wrapup;
            }
        } else {
            pFunc = PyFunction_New(code_object, d);
            if (!pFunc || !PyCallable_Check(pFunc)) {
                msg = PyError_CreateException("Failed to load function", NULL);
                goto wrapup;
            }
        }


        // The function has been successfully created/compiled, all that remains is to actually call the function
        pResult = PyObject_CallObject(pFunc, pArgs);

        Py_DECREF(pFunc);
        Py_DECREF(pArgs);
        Py_DECREF(pColumns);
        Py_DECREF(pColumnTypes);

        if (PyErr_Occurred()) {
            msg = PyError_CreateException("Python exception", pycall);
            if (code_object == NULL) { PyRun_SimpleString("del pyfun"); }
            goto wrapup;
        }

        //if (code_object == NULL) { PyRun_SimpleString("del pyfun"); }

        if (PyDict_Check(pResult)) { // Handle dictionary returns
            // For dictionary returns we need to map each of the (key,value) pairs to the proper return value
            // We first analyze the SQL Function structure for a list of return value names
            char **retnames = NULL;
            if (!varres)
            {
                if (sqlfun != NULL) {
                    retnames = GDKzalloc(sizeof(char*) * sqlfun->res->cnt);
                    argnode = sqlfun->res->h;
                    for(i = 0; i < sqlfun->res->cnt; i++) {
                        retnames[i] = ((sql_arg*)argnode->data)->name;
                        argnode = argnode->next;
                    }
                } else {
                    msg = createException(MAL, "pyapi.eval", "Return value is a dictionary, but there is no sql function object, so we don't know the return value names and mapping cannot be done.");
                    goto wrapup;
                }
            } else {
                // If there are a variable number of return types, we take the column names from the dictionary
                PyObject *keys = PyDict_Keys(pResult);
                retcols = PyList_Size(keys);
                retnames = GDKzalloc(sizeof(char*) * retcols);
                for(i = 0; i < retcols; i++) {
                    PyObject *colname = PyList_GetItem(keys, i);
                    if (!PyString_CheckExact(colname)) {
                        msg = createException(MAL, "pyapi.eval", "Expected a string key in the dictionary, but received an object of type %s", colname->ob_type->tp_name);
                        goto wrapup;
                    }
                    retnames[i] = ((PyStringObject*)colname)->ob_sval;
                }
            }
            pResult = PyDict_CheckForConversion(pResult, retcols, retnames, &msg);
            if (retnames != NULL) GDKfree(retnames);
        } else if (varres) {
            msg = createException(MAL, "pyapi.eval", "Expected a variable number return values, but the return type was not a dictionary. We require the return type to be a dictionary for column naming purposes.");
            goto wrapup;
        }
        else {
            // Now we need to do some error checking on the result object, because the result object has to have the correct type/size
            // We will also do some converting of result objects to a common type (such as scalar -> [[scalar]])
            pResult = PyObject_CheckForConversion(pResult, retcols, NULL, &msg);
        }
        if (pResult == NULL) {
            goto wrapup;
        }
    }
    VERBOSE_MESSAGE("Collecting return values.\n");

    if (varres) {
        GDKfree(pyreturn_values);
        pyreturn_values = GDKzalloc(retcols * sizeof(PyReturn));
    }

    // Now we have executed the Python function, we have to collect the return values and convert them to BATs
    // We will first collect header information about the Python return objects and extract the underlying C arrays
    // We will store this header information in a PyReturn object

    // The reason we are doing this as a separate step is because this preprocessing requires us to call the Python API
    // Whereas the actual returning does not require us to call the Python API
    // This means we can do the actual returning without holding the GIL
    if (!PyObject_PreprocessObject(pResult, pyreturn_values, retcols, &msg)) {
        goto wrapup;
    }


#ifndef WIN32
    /*[SHARED_MEMORY]*/
    // This is where the child process stops executing
    // We have successfully executed the Python function and converted the result object to a C array
    // Now all that is left is to copy the C array to shared memory so the main process can read it and return it
    if (mapped && child_process) {
        char *shm_ptr;
        ReturnBatDescr *ptr;

        // First we will fill in the header information, we will need to get a pointer to the header data first
        // The main process has already created the header data for the child process
        VERBOSE_MESSAGE("Getting shared memory.\n");
        msg = get_shared_memory(shm_id, memory_size, (void**) &shm_ptr);
        if (msg != MAL_SUCCEED) {
            goto wrapup;
        }

        VERBOSE_MESSAGE("Writing headers.\n");

#ifdef _PYAPI_TESTING_
        if (!disable_testing && !option_disablemalloctracking && benchmark_output != NULL) { revert_hook(); }
#endif
        // Now we will write data about our result (memory size, type, number of elements) to the header
        ptr = (ReturnBatDescr*)shm_ptr;
        for (i = 0; i < retcols; i++)
        {
            PyReturn *ret = &pyreturn_values[i];
            ReturnBatDescr *descr = &ptr[i];

            if (ret->result_type == NPY_OBJECT) {
                // We can't deal with NPY_OBJECT arrays, because these are 'arrays of pointers', so we can't just copy the content of the array into shared memory
                // So if we're dealing with a NPY_OBJECT array, we convert them to a Numpy Array of type NPY_<TYPE> that corresponds with the desired BAT type
                // WARNING: Because we could be converting to a NPY_STRING or NPY_UNICODE array (if the desired type is TYPE_str or TYPE_hge), this means that memory usage can explode
                //   because NPY_STRING/NPY_UNICODE arrays are 2D string arrays with fixed string length (so if there's one very large string the size explodes quickly)
                //   if someone has some problem with memory size exploding when using PYTHON_MAP but it being fine in regular PYTHON this is probably the issue
                int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));
                PyObject *new_array = PyArray_FromAny(ret->numpy_array, PyArray_DescrFromType(BatType_ToPyType(bat_type)), 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
                if (new_array == NULL) {
                    msg = createException(MAL, "pyapi.eval", "Could not convert the returned NPY_OBJECT array to the desired array of type %s.\n", BatType_Format(bat_type));
                    goto wrapup;
                }
                Py_DECREF(ret->numpy_array); //do we really care about cleaning this up, considering this only happens in a separate process that will be exited soon anyway?
                ret->numpy_array = new_array;
                ret->result_type = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->type_num;
                ret->memory_size = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->elsize;
                ret->count = PyArray_DIMS((PyArrayObject*)ret->numpy_array)[0];
                ret->array_data = PyArray_DATA((PyArrayObject*)ret->numpy_array);
            }

            descr->npy_type = ret->result_type;
            descr->element_size =   ret->memory_size;
            descr->bat_count = ret->count;
            descr->bat_size = ret->memory_size * ret->count;
            descr->has_mask = ret->mask_data != NULL;
#ifdef _PYAPI_TESTING_
            descr->peak_memory_usage = GET_MEMORY_PEAK();
#endif
            if (ret->count > 0)
            {
                int memory_size = ret->memory_size * ret->count;
                char *mem_ptr;
                //now create shared memory for the return value and copy the actual values
                assert(memory_size > 0);
                if (create_shared_memory(shm_id + (i + 1), memory_size, (void**)&mem_ptr) != MAL_SUCCEED)
                {
                    msg = createException(MAL, "pyapi.eval", "Failed to allocate shared memory for returning data.\n");
                    goto wrapup;
                }
                memcpy(mem_ptr, PyArray_DATA((PyArrayObject*)ret->numpy_array), memory_size);

                if (descr->has_mask)
                {
                    bool *mask_ptr;
                    int mask_size = ret->count * sizeof(bool);
                    assert(mask_size > 0);
                    if (create_shared_memory(shm_id + retcols + (i + 1), mask_size, (void**)&mask_ptr) != MAL_SUCCEED) //create a memory space for the mask
                    {
                        msg = createException(MAL, "pyapi.eval", "Failed to allocate shared memory for returning mask.\n");
                        goto wrapup;
                    }
                    assert(mask_ptr != NULL);
                    memcpy(mask_ptr, ret->mask_data, mask_size);
                }
            }
        }
        // Exit child process without an error code
        exit(0);
    }
#endif
    // We are done executing Python code (aside from cleanup), so we can release the GIL
    PyGILState_Release(gstate);
    gstate = PyGILState_LOCKED;

#ifndef WIN32 // This goto is only used for multiprocessing, which isn't supported on Windows yet
returnvalues:
#endif
    /*[RETURN_VALUES]*/
    VERBOSE_MESSAGE("Returning values.\n");

    for (i = 0; i < retcols; i++)
    {
        PyReturn *ret = &pyreturn_values[i];
        int bat_type = TYPE_any;
        if (!varres) {
            bat_type = getColumnType(getArgType(mb,pci,i));

            if (bat_type == TYPE_any || bat_type == TYPE_void) {
                bat_type = PyType_ToBat(ret->result_type);
                getArgType(mb,pci,i) = bat_type;
            }
        } else {
            bat_type = PyType_ToBat(ret->result_type);
        }

        b = PyObject_ConvertToBAT(ret, bat_type, i, seqbase, &msg);
        if (b == NULL) {
            goto wrapup;
        }

        if (isaBatType(getArgType(mb,pci,i)))
        {
            *getArgReference_bat(stk, pci, i) = b->batCacheid;
            BBPkeepref(b->batCacheid);
        }
        else
        { // single value return, only for non-grouped aggregations
            VALinit(&stk->stk[pci->argv[i]], bat_type, Tloc(b, BUNfirst(b)));
        }
        msg = MAL_SUCCEED;
    }
  wrapup:

#ifndef WIN32
    if (mapped && child_process)
    {
        // If we get here, something went wrong in a child process
        char *shm_ptr, *error_mem, *tmp_msg;
        ReturnBatDescr *ptr;

        // Now we exit the program with an error code
        VERBOSE_MESSAGE("Failure in child process: %s\n", msg);

        assert(memory_size > 0);
        tmp_msg = get_shared_memory(shm_id, memory_size, (void**) &shm_ptr);
        if (tmp_msg != MAL_SUCCEED) {
            VERBOSE_MESSAGE("Failed to get shared memory in child process: %s\n", tmp_msg);
            exit(1);
        }

        // To indicate that we failed, we will write information to our header
        ptr = (ReturnBatDescr*)shm_ptr;
        for (i = 0; i < retcols; i++) {
            ReturnBatDescr *descr = &ptr[i];
            // We will write descr->npy_type to -1, so other processes can see that we failed
            descr->npy_type = -1;
            // We will write the memory size of our error message to the bat_size, so the main process can access the shared memory
            descr->bat_size = (strlen(msg) + 1) * sizeof(char);
        }

        // Now create the shared memory to write our error message to
        // We can simply use the slot shm_id + 1, even though this is normally used for return values
        // This is because, if the process fails, no values will be returned
        tmp_msg = create_shared_memory(shm_id + 1, (strlen(msg) + 1) * sizeof(char), (void**) &error_mem);
        if (tmp_msg != MAL_SUCCEED) {
            VERBOSE_MESSAGE("Failed to create shared memory in child process: %s\n", tmp_msg);
            exit(1);
        }

        strcpy(error_mem, msg);
        exit(1);
    }
#endif

    VERBOSE_MESSAGE("Cleaning up.\n");

    if (holds_gil){
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        python_call_active = false;
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
    }

    // Actual cleanup
    // Cleanup input BATs
    for (i = pci->retc + 2; i < pci->argc; i++)
    {
        PyInput *inp = &pyinput_values[i - (pci->retc + 2)];
        if (inp->bat != NULL) BBPunfix(inp->bat->batCacheid);
    }
    if (pResult != NULL && gstate == PyGILState_LOCKED) {
        //if there is a pResult here, we are running single threaded (LANGUAGE PYTHON),
        //thus we need to free python objects, thus we need to obtain the GIL
        gstate = PyGILState_Ensure();
    }
    for (i = 0; i < retcols; i++) {
        PyReturn *ret = &pyreturn_values[i];
        // First clean up any return values
        if (!ret->multidimensional) {
            // Clean up numpy arrays, if they are there
            if (ret->numpy_array != NULL) Py_DECREF(ret->numpy_array);
            if (ret->numpy_mask != NULL) Py_DECREF(ret->numpy_mask);
        }
        // If there is no numpy array, but there is array data, then that array data must be shared memory
        if (ret->numpy_array == NULL && ret->array_data != NULL) {
            release_shared_memory(ret->array_data);
        }
        if (ret->numpy_mask == NULL && ret->mask_data != NULL) {
            release_shared_memory(ret->mask_data);
        }
    }
    if (pResult != NULL) {
        Py_DECREF(pResult);
    }
    if (gstate != PyGILState_LOCKED) {
        PyGILState_Release(gstate);
    }

    // Now release some GDK memory we allocated for strings and input values
    GDKfree(pyreturn_values);
    GDKfree(pyinput_values);
    for (i = 0; i < pci->argc; i++)
        if (args[i] != NULL)
            GDKfree(args[i]);
    GDKfree(args);
    GDKfree(pycall);

#ifdef _PYAPI_TESTING_
    GDKfree(exprStr);
    if (benchmark_output != NULL) {
        FILE *f = NULL;
        if (!mapped && !disable_testing && !option_disablemalloctracking) {
            revert_hook();
            peak_memory_usage = GET_MEMORY_PEAK();
        }
        timer(&end_time);
        VERBOSE_MESSAGE("End Time: %f\n", GET_ELAPSED_TIME(start_time, end_time));

        // We lock file access for when mapped is set
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        f = fopen(benchmark_output, "a");
        if (f != NULL) {
            fprintf(f, "%llu\t%f\n", peak_memory_usage, GET_ELAPSED_TIME(start_time, end_time));
            if (msg != MAL_SUCCEED) {
                fprintf(f, "Error: %s\n", msg);
            }
            fclose(f);
        } else {
            perror("Error");
        }
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
    }
#endif
    VERBOSE_MESSAGE("Finished cleaning up.\n");
    return msg;
}

str
 PyAPIprelude(void *ret) {
    (void) ret;
#ifndef _EMBEDDED_MONETDB_MONETDB_LIB_
    MT_lock_init(&pyapiLock, "pyapi_lock");
    if (PyAPIEnabled()) {
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        if (!pyapiInitialized) {
            char* iar = NULL;
            Py_Initialize();
            PyRun_SimpleString("import numpy");
            PyByteArray_Override();
            import_array1(iar);
            initialize_shared_memory();
            lazyarray_init();
            _connection_init();
            marshal_module = PyImport_Import(PyString_FromString("marshal"));
            if (marshal_module == NULL) {
                return createException(MAL, "pyapi.eval", "Failed to load Marshal module.");
            }
            marshal_loads = PyObject_GetAttrString(marshal_module, "loads");
            if (marshal_loads == NULL) {
                return createException(MAL, "pyapi.eval", "Failed to load function \"loads\" from Marshal module.");
            }
            PyEval_SaveThread();
            pyapiInitialized++;
        }
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
        fprintf(stdout, "# MonetDB/Python module loaded\n");
    }
#else
    if (!pyapiInitialized) {
        char* iar = NULL;
        import_array1(iar);
        pyapiInitialized++;
    }
#endif
#ifdef _PYAPI_VERBOSE_
    option_verbose = GDKgetenv_isyes(verbose_enableflag) || GDKgetenv_istrue(verbose_enableflag);
#endif
#ifdef _PYAPI_DEBUG_
    option_debug = GDKgetenv_isyes(debug_enableflag) || GDKgetenv_istrue(debug_enableflag);
    (void) option_debug;
#endif
#ifdef _PYAPI_WARNINGS_
    option_warning = GDKgetenv_isyes(warning_enableflag) || GDKgetenv_istrue(warning_enableflag);
#endif
#ifdef _PYAPI_TESTING_
    //These flags are for testing purposes, they shouldn't be used for normal purposes
    option_zerocopyinput = !(GDKgetenv_isyes(zerocopyinput_disableflag) || GDKgetenv_istrue(zerocopyinput_disableflag));
    option_zerocopyoutput = !(GDKgetenv_isyes(zerocopyoutput_disableflag) || GDKgetenv_istrue(zerocopyoutput_disableflag));
    option_numpy_string_array = !(GDKgetenv_isyes(numpy_string_array_disableflag) || GDKgetenv_istrue(numpy_string_array_disableflag));
    option_bytearray = GDKgetenv_isyes(bytearray_enableflag) || GDKgetenv_istrue(bytearray_enableflag);
    option_oldnullmask = GDKgetenv_isyes(oldnullmask_enableflag) || GDKgetenv_istrue(oldnullmask_enableflag);
    option_lazyarray = GDKgetenv_isyes(lazyarray_enableflag) || GDKgetenv_istrue(lazyarray_enableflag);
    option_alwaysunicode = (GDKgetenv_isyes(alwaysunicode_enableflag) || GDKgetenv_istrue(alwaysunicode_enableflag));
    benchmark_output = GDKgetenv(benchmark_output_flag);
    option_disablemalloctracking = (GDKgetenv_isyes(disable_malloc_tracking) || GDKgetenv_istrue(disable_malloc_tracking));
    fprintf(stdout, "# MonetDB/Python testing enabled.\n");
#endif
    return MAL_SUCCEED;
}

//Returns true if the type of [object] is a scalar (i.e. numeric scalar or string, basically "not an array but a single value")
bool PyType_IsPyScalar(PyObject *object)
{
    if (object == NULL) return false;
    return (PyArray_CheckScalar(object) || PyInt_Check(object) || PyFloat_Check(object) || PyLong_Check(object) || PyString_Check(object) || PyBool_Check(object) || PyUnicode_Check(object) || PyByteArray_Check(object));
}


static char *PyError_CreateException(char *error_text, char *pycall)
{
    PyObject *py_error_type = NULL, *py_error_value = NULL, *py_error_traceback = NULL;
    char *py_error_string = NULL;
    lng line_number = -1;

    PyErr_Fetch(&py_error_type, &py_error_value, &py_error_traceback);
    if (py_error_value) {
        PyObject *error;
        PyErr_NormalizeException(&py_error_type, &py_error_value, &py_error_traceback);
        error = PyObject_Str(py_error_value);

        py_error_string = PyString_AS_STRING(error);
        Py_XDECREF(error);
        if (pycall != NULL && strlen(pycall) > 0) {
            if (py_error_traceback == NULL) {
                //no traceback info, this means we are dealing with a parsing error
                //line information should be in the error message
                sscanf(py_error_string, "%*[^0-9]"LLFMT, &line_number);
                if (line_number < 0) goto finally;
            } else {
                line_number = ((PyTracebackObject*)py_error_traceback)->tb_lineno;
            }

            // Now only display the line numbers around the error message, we display 5 lines around the error message
            {
                char linenr[32];
                size_t nrpos, pos, i, j;
                char lineinformation[5000]; //we only support 5000 characters for 5 lines of the program, should be enough
                nrpos = 0; // Current line number
                pos = 0; //Current position in the lineinformation result array
                for(i = 0; i < strlen(pycall); i++) {
                    if (pycall[i] == '\n' || i == 0) {
                        // Check if we have arrived at a new line, if we have increment the line count
                        nrpos++;
                        // Now check if we should display this line
                        if (nrpos >= ((size_t)line_number - 2) && nrpos <= ((size_t)line_number + 2) && pos < 4997) {
                            // We shouldn't put a newline on the first line we encounter, only on subsequent lines
                            if (nrpos > ((size_t)line_number - 2)) lineinformation[pos++] = '\n';
                            if ((size_t)line_number == nrpos) {
                                // If this line is the 'error' line, add an arrow before it, otherwise just add spaces
                                lineinformation[pos++] = '>';
                                lineinformation[pos++] = ' ';
                            } else {
                                lineinformation[pos++] = ' ';
                                lineinformation[pos++] = ' ';
                            }
                            snprintf(linenr, 32, SZFMT, nrpos);
                            for(j = 0; j < strlen(linenr); j++) {
                                lineinformation[pos++] = linenr[j];
                            }
                            lineinformation[pos++] = '.';
                            lineinformation[pos++] = ' ';
                        }
                    }
                    if (pycall[i] != '\n' && nrpos >= (size_t)line_number - 2 && nrpos <= (size_t)line_number + 2 && pos < 4999) {
                        // If we are on a line number that we have to display, copy the text from this line for display
                        lineinformation[pos++] = pycall[i];
                    }
                }
                lineinformation[pos] = '\0';
                return createException(MAL, "pyapi.eval", "%s\n%s\n%s", error_text, lineinformation, py_error_string);
            }
        }
    }
    else {
        py_error_string = "";
    }
finally:
    if (pycall == NULL) return createException(MAL, "pyapi.eval", "%s\n%s", error_text, py_error_string);
    return createException(MAL, "pyapi.eval", "%s\n%s\n%s", error_text, pycall, py_error_string);
}

PyObject *PyArrayObject_FromScalar(PyInput* inp, char **return_message)
{
    PyObject *vararray = NULL;
    char *msg = NULL;
    assert(inp->scalar); //input has to be a scalar
    VERBOSE_MESSAGE("- Loading a scalar of type %s (%i)", BatType_Format(inp->bat_type), inp->bat_type);

    switch(inp->bat_type)
    {
        case TYPE_bit:
            vararray = PyInt_FromLong((long)(*(bit*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %ld]\n", (long)(*(bit*)inp->dataptr));
            break;
        case TYPE_bte:
            vararray = PyInt_FromLong((long)(*(bte*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %ld]\n", (long)(*(bte*)inp->dataptr));
            break;
        case TYPE_sht:
            vararray = PyInt_FromLong((long)(*(sht*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %ld]\n", (long)(*(sht*)inp->dataptr));
            break;
        case TYPE_int:
            vararray = PyInt_FromLong((long)(*(int*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %ld]\n", (long)(*(int*)inp->dataptr));
            break;
        case TYPE_lng:
            vararray = PyLong_FromLong((long)(*(lng*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %ld]\n", (long)(*(lng*)inp->dataptr));
            break;
        case TYPE_flt:
            vararray = PyFloat_FromDouble((double)(*(flt*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %lf]\n", (double)(*(flt*)inp->dataptr));
            break;
        case TYPE_dbl:
            vararray = PyFloat_FromDouble((double)(*(dbl*)inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %lf]\n", (double)(*(dbl*)inp->dataptr));
            break;
#ifdef HAVE_HGE
        case TYPE_hge:
            vararray = PyLong_FromHge(*((hge *) inp->dataptr));
            VERBOSE_MESSAGE(" [Value: Huge]\n");
            break;
#endif
        case TYPE_str:
            vararray = PyUnicode_FromString(*((char**) inp->dataptr));
            VERBOSE_MESSAGE(" [Value: %s]\n", *((char**) inp->dataptr));
            break;
        default:
            VERBOSE_MESSAGE(" [Value: Unknown]\n");
            msg = createException(MAL, "pyapi.eval", "Unsupported scalar type %i.", inp->bat_type);
            goto wrapup;
    }
    if (vararray == NULL)
    {
        msg = createException(MAL, "pyapi.eval", "Something went wrong converting the MonetDB scalar to a Python scalar.");
        goto wrapup;
    }
wrapup:
    *return_message = msg;
    return vararray;
}

PyObject *PyMaskedArray_FromBAT(PyInput *inp, size_t t_start, size_t t_end, char **return_message)
{
    BAT *b = inp->bat;
    char *msg;
    PyObject *vararray = PyArrayObject_FromBAT(inp, t_start, t_end, return_message);
    if (vararray == NULL) {
        return NULL;
    }
    // To deal with null values, we use the numpy masked array structure
    // The masked array structure is an object with two arrays of equal size, a data array and a mask array
    // The mask array is a boolean array that has the value 'True' when the element is NULL, and 'False' otherwise
    // If the BAT has Null values, we construct this masked array
    if (!(b->T->nil == 0 && b->T->nonil == 1))
    {
        PyObject *mask;
        PyObject *mafunc = PyObject_GetAttrString(PyImport_Import(PyString_FromString("numpy.ma")), "masked_array");
        PyObject *maargs;
        PyObject *nullmask = PyNullMask_FromBAT(b, t_start, t_end);

        if (nullmask == Py_None) {
            maargs = PyTuple_New(1);
            PyTuple_SetItem(maargs, 0, vararray);
        } else {
            maargs = PyTuple_New(2);
            PyTuple_SetItem(maargs, 0, vararray);
            PyTuple_SetItem(maargs, 1, (PyObject*) nullmask);
        }

        // Now we will actually construct the mask by calling the masked array constructor
        mask = PyObject_CallObject(mafunc, maargs);
        if (!mask) {
            msg = PyError_CreateException("Failed to create mask", NULL);
            goto wrapup;
        }
        Py_DECREF(maargs);
        Py_DECREF(mafunc);

        vararray = mask;
    }
    return vararray;
wrapup:
    *return_message = msg;
    return NULL;
}

PyObject *PyArrayObject_FromBAT(PyInput *inp, size_t t_start, size_t t_end, char **return_message)
{
    // This variable will hold the converted Python object
    PyObject *vararray = NULL;
    char *msg;
    size_t j = 0;
    BUN p = 0, q = 0;
    BATiter li;
    BAT *b = inp->bat;

    assert(!inp->scalar); //input has to be a BAT

    if (b == NULL)
    {
        // No BAT was found, we can't do anything in this case
        msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        goto wrapup;
    }

    VERBOSE_MESSAGE("- Loading a BAT of type %s (%d) [Size: %lu]\n", BatType_Format(inp->bat_type), inp->bat_type, inp->count);

    switch (inp->bat_type) {
    case TYPE_bte:
        BAT_TO_NP(b, bte, NPY_INT8);
        break;
    case TYPE_sht:
        BAT_TO_NP(b, sht, NPY_INT16);
        break;
    case TYPE_int:
        BAT_TO_NP(b, int, NPY_INT32);
        break;
    case TYPE_lng:
        BAT_TO_NP(b, lng, NPY_INT64);
        break;
    case TYPE_flt:
        BAT_TO_NP(b, flt, NPY_FLOAT32);
        break;
    case TYPE_dbl:
        BAT_TO_NP(b, dbl, NPY_FLOAT64);
        break;
    case TYPE_str:
#ifdef _PYAPI_TESTING_
        if (option_numpy_string_array)
#endif
        {
            bool unicode = false;
            size_t maxsize = 0;
            li = bat_iterator(b);

            //we first loop over all the strings in the BAT to find the maximum length of a single string
            //this is because NUMPY only supports strings with a fixed maximum length
            j = 0;
            BATloop(b, p, q) {
                if (j >= t_start) {
                    bool ascii;
                    const char *t = (const char *) BUNtail(li, p);
                    size_t length;
                    if (strcmp(t, str_nil) == 0) {
                        length = 1;
                    } else {
                        length = utf8_strlen(t, &ascii); //get the amount of UTF-8 characters in the string
                        unicode = !ascii || unicode; //if even one string is unicode we have to store the entire array as unicode
                    }
                    if (length > maxsize)
                        maxsize = length;
                }
                if (j == t_end) break;
                j++;
            }
            if (unicode) {
                VERBOSE_MESSAGE("- Unicode string!\n");
                //create a NPY_UNICODE array object
                vararray = PyArray_New(
                    &PyArray_Type,
                    1,
                    (npy_intp[1]) {t_end - t_start},
                    NPY_UNICODE,
                    NULL,
                    NULL,
                    maxsize * 4,  //we have to do maxsize*4 because NPY_UNICODE is stored as UNICODE-32 (i.e. 4 bytes per character)
                    0,
                    NULL);
                //fill the NPY_UNICODE array object using the PyArray_SETITEM function
                j = 0;
                BATloop(b, p, q)
                {
                    if (j >= t_start) {
                        char *t = (char *) BUNtail(li, p);
                        PyObject *obj;
                        if (strcmp(t, str_nil) == 0) {
                             //str_nil isn't a valid UTF-8 character (it's 0x80), so we can't decode it as UTF-8 (it will throw an error)
                            obj = PyUnicode_FromString("-");
                        }
                        else {
                            //otherwise we can just decode the string as UTF-8
                            obj = PyUnicode_FromString(t);
                        }

                        if (obj == NULL)
                        {
                            PyErr_Print();
                            msg = createException(MAL, "pyapi.eval", "Failed to decode string as UTF-8.");
                            goto wrapup;
                        }
                        PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j - t_start), obj);
                        Py_DECREF(obj);
                    }
                    if (j == t_end) break;
                    j++;
                }
            } else {
                VERBOSE_MESSAGE("- ASCII string!\n");
                //create a NPY_STRING array object
                vararray = PyArray_New(
                    &PyArray_Type,
                    1,
                    (npy_intp[1]) {t_end - t_start},
                    NPY_STRING,
                    NULL,
                    NULL,
                    maxsize,
                    0,
                    NULL);
                j = 0;
                BATloop(b, p, q)
                {
                    if (j >= t_start) {
                        char *t = (char *) BUNtail(li, p);
                        PyObject *obj = PyString_FromString(t);

                        if (obj == NULL)
                        {
                            msg = createException(MAL, "pyapi.eval", "Failed to create string.");
                            goto wrapup;
                        }
                        PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j - t_start), obj);
                        Py_DECREF(obj);
                    }
                    if (j == t_end) break;
                    j++;
                }
            }
        }
#ifdef _PYAPI_TESTING_
        else
        {
            bool unicode = option_alwaysunicode;
            li = bat_iterator(b);
            //create a NPY_OBJECT array object
            vararray = PyArray_New(
                &PyArray_Type,
                1,
                (npy_intp[1]) {t_end - t_start},
                NPY_OBJECT,
                NULL,
                NULL,
                0,
                0,
                NULL);

            if (!option_alwaysunicode)
            {
                j = 0;
                BATloop(b, p, q) {
                    if (j >= t_start) {
                        bool ascii;
                        const char *t = (const char *) BUNtail(li, p);
                        if (strcmp(t, str_nil) == 0) continue;
                        utf8_strlen(t, &ascii);
                        unicode = !ascii || unicode;
                        if (unicode) break;
                    }
                    if (j == t_end) break;
                    j++;
                }
            }

            j = 0;
            BATloop(b, p, q)
            {
                if (j >= t_start) {
                    char *t = (char *) BUNtail(li, p);
                    PyObject *obj;
                    if (unicode)
                    {
                        if (strcmp(t, str_nil) == 0) {
                             //str_nil isn't a valid UTF-8 character (it's 0x80), so we can't decode it as UTF-8 (it will throw an error)
                            obj = PyUnicode_FromString("-");
                        }
                        else {
                            //otherwise we can just decode the string as UTF-8
                            obj = PyUnicode_FromString(t);
                        }
                    } else {
                        if (option_bytearray) obj = PyByteArray_FromString(t);
                        else
                        {
                            obj = PyString_FromString(t);
                        }
                    }

                    if (obj == NULL)
                    {
                        msg = createException(MAL, "pyapi.eval", "Failed to create string.");
                        goto wrapup;
                    }
                    ((PyObject**)PyArray_DATA((PyArrayObject*)vararray))[j - t_start] = obj;
                }
                if (j == t_end) break;
                j++;
            }
        }
#endif
        break;
#ifdef HAVE_HGE
    case TYPE_hge:
    {
        li = bat_iterator(b);

        //create a NPY_OBJECT array to hold the huge type
        vararray = PyArray_New(
            &PyArray_Type,
            1,
            (npy_intp[1]) { t_end - t_start },
            NPY_OBJECT,
            NULL,
            NULL,
            0,
            0,
            NULL);

        j = 0;
        WARNING_MESSAGE("!PERFORMANCE WARNING: Type \"hge\" (128 bit) is unsupported by Numpy. The numbers are instead converted to python objects of type \"PyLong\". This means a python object is constructed for every huge integer and the entire column is copied.\n");
        BATloop(b, p, q) {
            if (j >= t_start) {
                PyObject *obj;
                const hge *t = (const hge *) BUNtail(li, p);
                obj = PyLong_FromHge(*t);
                ((PyObject**)PyArray_DATA((PyArrayObject*)vararray))[j - t_start] = obj;
            }
            if (j == t_end) break;
            j++;
        }
        break;
    }
#endif
    default:
        msg = createException(MAL, "pyapi.eval", "unknown argument type ");
        goto wrapup;
    }
    if (vararray == NULL) {
        msg = PyError_CreateException("Failed to convert BAT to Numpy array.", NULL);
        goto wrapup;
    }
    return vararray;
wrapup:
    *return_message = msg;
    return NULL;
}

#define CreateNullMask(tpe)                                        \
    for(j = 0; j < count; j++) {                                   \
        mask_data[j] = *((tpe*)BUNtail(bi, BUNfirst(b) + j)) == tpe##_nil;  \
        found_nil = found_nil || mask_data[j];                     \
    }

PyObject *PyNullMask_FromBAT(BAT *b, size_t t_start, size_t t_end)
{
    // We will now construct the Masked array, we start by setting everything to False
    size_t count = t_end - t_start;
    PyArrayObject* nullmask = (PyArrayObject*) PyArray_ZEROS(1, (npy_intp[1]) {( count )}, NPY_BOOL, 0);
    const void *nil = ATOMnilptr(b->ttype);
    size_t j;
    bool found_nil = false;
    BATiter bi = bat_iterator(b);
    bool *mask_data = (bool*)PyArray_DATA(nullmask);

#ifdef _PYAPI_TESTING_
    if (!option_oldnullmask) {
#endif
    switch(ATOMstorage(getColumnType(b->T->type)))
    {
        case TYPE_bit: CreateNullMask(bit); break;
        case TYPE_bte: CreateNullMask(bte); break;
        case TYPE_sht: CreateNullMask(sht); break;
        case TYPE_int: CreateNullMask(int); break;
        case TYPE_lng: CreateNullMask(lng); break;
        case TYPE_flt: CreateNullMask(flt); break;
        case TYPE_dbl: CreateNullMask(dbl); break;
#ifdef HAVE_HGE
        case TYPE_hge: CreateNullMask(hge); break;
#endif
        case TYPE_str:
        {
            int (*atomcmp)(const void *, const void *) = ATOMcompare(b->ttype);
            for (j = 0; j < count; j++) {
                mask_data[j] = (*atomcmp)(BUNtail(bi, BUNfirst(b) + j), nil) == 0;
                found_nil = found_nil || mask_data[j];
            }
            break;
        }
        default:
            //todo: do something with the error?
            return NULL;
    }
#ifdef _PYAPI_TESTING_
    } else {
        int (*atomcmp)(const void *, const void *) = ATOMcompare(b->ttype);
        for (j = 0; j < count; j++) {
            if ((*atomcmp)(BUNtail(bi, BUNfirst(b) + j), nil) == 0) {
                ((bool*)PyArray_DATA(nullmask))[j] = true;
                found_nil = true;
            }
        }
    }
#endif


    if (!found_nil) {
        Py_DECREF(nullmask);
        Py_RETURN_NONE;
    }

    return (PyObject*)nullmask;
}


PyObject *PyDict_CheckForConversion(PyObject *pResult, int expected_columns, char **retcol_names, char **return_message)
{
    char *msg = MAL_SUCCEED;
    PyObject *result = PyList_New(expected_columns), *keys = PyDict_Keys(pResult);
    int i;

    if (PyList_Size(keys) != expected_columns) {
#ifdef _PYAPI_WARNINGS_
        if (PyList_Size(keys) > expected_columns) {
            WARNING_MESSAGE("WARNING: Expected %d return values, but a dictionary with "SSZFMT" values was returned instead.\n", expected_columns, PyList_Size(keys));
        }
#endif
    }

    for(i = 0; i < expected_columns; i++) {
        PyObject *object = PyDict_GetItemString(pResult, retcol_names[i]);
        if (object == NULL) {
            msg = createException(MAL, "pyapi.eval", "Expected a return value with name \"%s\", but this key was not present in the dictionary.", retcol_names[i]);
            goto wrapup;
        }
        object = PyObject_CheckForConversion(object, 1, NULL, return_message);
        if (object == NULL) {
            msg = createException(MAL, "pyapi.eval", "Error converting dict return value \"%s\": %s.", retcol_names[i], *return_message);
            GDKfree(*return_message);
            goto wrapup;
        }
        if (PyList_CheckExact(object)) {
            PyList_SetItem(result, i, PyList_GetItem(object, 0));
        } else {
            msg = createException(MAL, "pyapi.eval", "Why is this not a list?");
            goto wrapup;
        }
    }
    return result;
wrapup:
    *return_message = msg;
    Py_DECREF(result);
    Py_DECREF(keys);
    return NULL;
}

PyObject *PyObject_CheckForConversion(PyObject *pResult, int expected_columns, int *actual_columns, char **return_message)
{
    char *msg;
    int columns = 0;
    if (pResult) {
        PyObject * pColO = NULL;
        if (PyType_IsPandasDataFrame(pResult)) {
            //the result object is a Pandas data frame
            //we can convert the pandas data frame to a numpy array by simply accessing the "values" field (as pandas dataframes are numpy arrays internally)
            pResult = PyObject_GetAttrString(pResult, "values");
            if (pResult == NULL) {
                msg = createException(MAL, "pyapi.eval", "Invalid Pandas data frame.");
                goto wrapup;
            }
            //we transpose the values field so it's aligned correctly for our purposes
            pResult = PyObject_GetAttrString(pResult, "T");
            if (pResult == NULL) {
                msg = createException(MAL, "pyapi.eval", "Invalid Pandas data frame.");
                goto wrapup;
            }
        }

        if (PyType_IsPyScalar(pResult)) { //check if the return object is a scalar
            if (expected_columns == 1 || expected_columns <= 0)  {
                //if we only expect a single return value, we can accept scalars by converting it into an array holding an array holding the element (i.e. [[pResult]])
                PyObject *list = PyList_New(1);
                PyList_SetItem(list, 0, pResult);
                pResult = list;

                list = PyList_New(1);
                PyList_SetItem(list, 0, pResult);
                pResult = list;

                columns = 1;
            }
            else {
                //the result object is a scalar, yet we expect more than one return value. We can only convert the result into a list with a single element, so the output is necessarily wrong.
                msg = createException(MAL, "pyapi.eval", "A single scalar was returned, yet we expect a list of %d columns. We can only convert a single scalar into a single column, thus the result is invalid.", expected_columns);
                goto wrapup;
            }
        }
        else {
            //if it is not a scalar, we check if it is a single array
            bool IsSingleArray = TRUE;
            PyObject *data = pResult;
            if (PyType_IsNumpyMaskedArray(data)) {
                data = PyObject_GetAttrString(pResult, "data");
                if (data == NULL) {
                    msg = createException(MAL, "pyapi.eval", "Invalid masked array.");
                    goto wrapup;
                }
            }
            if (PyType_IsNumpyArray(data)) {
                if (PyArray_NDIM((PyArrayObject*)data) != 1) {
                    IsSingleArray = FALSE;
                }
                else {
                    pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, 0));
                    IsSingleArray = PyType_IsPyScalar(pColO);
                }
            }
            else if (PyList_Check(data)) {
                pColO = PyList_GetItem(data, 0);
                IsSingleArray = PyType_IsPyScalar(pColO);
            }
            else if (PyLazyArray_CheckExact(data)) {
                pColO = data;
                IsSingleArray = TRUE;
            } else if (!PyType_IsNumpyMaskedArray(data)) {
                //it is neither a python array, numpy array or numpy masked array, thus the result is unsupported! Throw an exception!
                msg = createException(MAL, "pyapi.eval", "Unsupported result object. Expected either a list, dictionary, a numpy array, a numpy masked array or a pandas data frame, but received an object of type \"%s\"", PyString_AsString(PyObject_Str(PyObject_Type(data))));
                goto wrapup;
            }

            if (IsSingleArray) {
                if (expected_columns == 1 || expected_columns <= 0) {
                    //if we only expect a single return value, we can accept a single array by converting it into an array holding an array holding the element (i.e. [pResult])
                    PyObject *list = PyList_New(1);
                    PyList_SetItem(list, 0, pResult);
                    pResult = list;

                    columns = 1;
                }
                else {
                    //the result object is a single array, yet we expect more than one return value. We can only convert the result into a list with a single array, so the output is necessarily wrong.
                    msg = createException(MAL, "pyapi.eval", "A single array was returned, yet we expect a list of %d columns. The result is invalid.", expected_columns);
                    goto wrapup;
                }
            }
            else {
                //the return value is an array of arrays, all we need to do is check if it is the correct size
                int results = 0;
                if (PyList_Check(data)) results = PyList_Size(data);
                else results = PyArray_DIMS((PyArrayObject*)data)[0];
                columns = results;
                if (results != expected_columns && expected_columns > 0) {
                    //wrong return size, we expect pci->retc arrays
                    msg = createException(MAL, "pyapi.eval", "An array of size %d was returned, yet we expect a list of %d columns. The result is invalid.", results, expected_columns);
                    goto wrapup;
                }
            }
        }
    } else {
        msg = createException(MAL, "pyapi.eval", "Invalid result object. No result object could be generated.");
        goto wrapup;
    }

    if (actual_columns != NULL) *actual_columns = columns;
    return pResult;
wrapup:
    if (actual_columns != NULL) *actual_columns = columns;
    *return_message = msg;
    return NULL;
}


bool PyObject_PreprocessObject(PyObject *pResult, PyReturn *pyreturn_values, int column_count, char **return_message)
{
    int i;
    char *msg;
    for (i = 0; i < column_count; i++) {
        // Refers to the current Numpy mask (if it exists)
        PyObject *pMask = NULL;
        // Refers to the current Numpy array
        PyObject * pColO = NULL;
        // This is the PyReturn header information for the current return value, we will fill this now
        PyReturn *ret = &pyreturn_values[i];

        ret->multidimensional = FALSE;
        // There are three possibilities (we have ensured this right after executing the Python call by calling PyObject_CheckForConversion)
        // 1: The top level result object is a PyList or Numpy Array containing pci->retc Numpy Arrays
        // 2: The top level result object is a (pci->retc x N) dimensional Numpy Array [Multidimensional]
        // 3: The top level result object is a (pci->retc x N) dimensional Numpy Masked Array [Multidimensional]
        if (PyList_Check(pResult)) {
            // If it is a PyList, we simply get the i'th Numpy array from the PyList
            pColO = PyList_GetItem(pResult, i);
        }
        else {
            // If it isn't, the result object is either a Nump Masked Array or a Numpy Array
            PyObject *data = pResult;
            if (PyType_IsNumpyMaskedArray(data)) {
                data = PyObject_GetAttrString(pResult, "data"); // If it is a Masked array, the data is stored in the masked_array.data attribute
                pMask = PyObject_GetAttrString(pResult, "mask");
            }

            // We can either have a multidimensional numpy array, or a single dimensional numpy array
            if (PyArray_NDIM((PyArrayObject*)data) != 1) {
                // If it is a multidimensional numpy array, we have to convert the i'th dimension to a NUMPY array object
                ret->multidimensional = TRUE;
                ret->result_type = PyArray_DESCR((PyArrayObject*)data)->type_num;
            }
            else {
                // If it is a single dimensional Numpy array, we get the i'th Numpy array from the Numpy Array
                pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, i));
            }
        }

        // Now we have to do some preprocessing on the data
        if (ret->multidimensional) {
            // If it is a multidimensional Numpy array, we don't need to do any conversion, we can just do some pointers
            ret->count = PyArray_DIMS((PyArrayObject*)pResult)[1];
            ret->numpy_array = pResult;
            ret->numpy_mask = pMask;
            ret->array_data = PyArray_DATA((PyArrayObject*)ret->numpy_array);
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA((PyArrayObject*)ret->numpy_mask);
            ret->memory_size = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->elsize;
        }
        else {
            if (PyLazyArray_CheckExact(pColO)) {
                // To handle returning of lazy arrays, we just convert them to a Numpy array. This is slow and could be done much faster, but since this can only happen if we directly return one of the input arguments this should be a rare situation anyway.
                pColO = PyLazyArray_AsNumpyArray(pColO);
                if (pColO == NULL) {
                    msg = PyError_CreateException("Failed to convert lazy array to numpy array.\n", NULL);
                    goto wrapup;
                }
            }
            // If it isn't we need to convert pColO to the expected Numpy Array type
            ret->numpy_array = PyArray_FromAny(pColO, NULL, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
            if (ret->numpy_array == NULL) {
                msg = createException(MAL, "pyapi.eval", "Could not create a Numpy array from the return type.\n");
                goto wrapup;
            }

            ret->result_type = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->type_num; // We read the result type from the resulting array
            ret->memory_size = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->elsize;
            ret->count = PyArray_DIMS((PyArrayObject*)ret->numpy_array)[0];
            ret->array_data = PyArray_DATA((PyArrayObject*)ret->numpy_array);
            // If pColO is a Masked array, we convert the mask to a NPY_BOOL numpy array
            if (PyObject_HasAttrString(pColO, "mask")) {
                pMask = PyObject_GetAttrString(pColO, "mask");
                if (pMask != NULL) {
                    ret->numpy_mask = PyArray_FromAny(pMask, PyArray_DescrFromType(NPY_BOOL), 1, 1,  NPY_ARRAY_CARRAY, NULL);
                    if (ret->numpy_mask == NULL || PyArray_DIMS((PyArrayObject*)ret->numpy_mask)[0] != (int)ret->count)
                    {
                        PyErr_Clear();
                        pMask = NULL;
                        ret->numpy_mask = NULL;
                    }
                }
            }
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA((PyArrayObject*)ret->numpy_mask);
        }
    }
    return TRUE;
wrapup:
    *return_message = msg;
    return FALSE;
}

BAT *PyObject_ConvertToBAT(PyReturn *ret, int bat_type, int i, int seqbase, char **return_message)
{
    BAT *b = NULL;
    size_t index_offset = 0;
    char *msg;
    size_t iu;

    if (ret->multidimensional) index_offset = i;
    VERBOSE_MESSAGE("- Returning a Numpy Array of type %s of size %zu and storing it in a BAT of type %s\n", PyType_Format(ret->result_type), ret->count,  BatType_Format(bat_type));
    switch (bat_type)
    {
    case TYPE_bit:
        NP_CREATE_BAT(b, bit);
        break;
    case TYPE_bte:
        NP_CREATE_BAT(b, bte);
        break;
    case TYPE_sht:
        NP_CREATE_BAT(b, sht);
        break;
    case TYPE_int:
        NP_CREATE_BAT(b, int);
        break;
    case TYPE_oid:
        NP_CREATE_BAT(b, oid);
        break;
    case TYPE_lng:
        NP_CREATE_BAT(b, lng);
        break;
    case TYPE_wrd:
        NP_CREATE_BAT(b, wrd);
        break;
    case TYPE_flt:
        NP_CREATE_BAT(b, flt);
        break;
    case TYPE_dbl:
        NP_CREATE_BAT(b, dbl);
        break;
#ifdef HAVE_HGE
    case TYPE_hge:
        NP_CREATE_BAT(b, hge);
        break;
#endif
    case TYPE_str:
        {
            bool *mask = NULL;
            char *data = NULL;
            char *utf8_string = NULL;
            if (ret->mask_data != NULL)
            {
                mask = (bool*)ret->mask_data;
            }
            if (ret->array_data == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "No return value stored in the structure.  n");
                goto wrapup;
            }
            data = (char*) ret->array_data;

            if (ret->result_type != NPY_OBJECT) {
                utf8_string = GDKzalloc(utf8string_minlength + ret->memory_size + 1);
                utf8_string[utf8string_minlength + ret->memory_size] = '\0';
            }

            b = BATnew(TYPE_void, TYPE_str, ret->count, TRANSIENT);
            BATseqbase(b, seqbase); b->T->nil = 0; b->T->nonil = 1;
            b->tkey = 0; b->tsorted = 0; b->trevsorted = 0;
            VERBOSE_MESSAGE("- Collecting return values of type %s.\n", PyType_Format(ret->result_type));
            switch(ret->result_type)
            {
                case NPY_BOOL:      NP_COL_BAT_STR_LOOP(b, bit, "%hhd"); break;
                case NPY_BYTE:      NP_COL_BAT_STR_LOOP(b, bte, "%hhd"); break;
                case NPY_SHORT:     NP_COL_BAT_STR_LOOP(b, sht, "%hd"); break;
                case NPY_INT:       NP_COL_BAT_STR_LOOP(b, int, "%d"); break;
                case NPY_LONG:      NP_COL_BAT_STR_LOOP(b, long, "%ld"); break;
                case NPY_LONGLONG:  NP_COL_BAT_STR_LOOP(b, lng, LLFMT); break;
                case NPY_UBYTE:     NP_COL_BAT_STR_LOOP(b, unsigned char, "%hhu"); break;
                case NPY_USHORT:    NP_COL_BAT_STR_LOOP(b, unsigned short, "%hu"); break;
                case NPY_UINT:      NP_COL_BAT_STR_LOOP(b, unsigned int, "%u"); break;
                case NPY_ULONG:     NP_COL_BAT_STR_LOOP(b, unsigned long, "%lu"); break;
                case NPY_ULONGLONG: NP_COL_BAT_STR_LOOP(b, unsigned long long, ULLFMT); break;
                case NPY_FLOAT16:
                case NPY_FLOAT:     NP_COL_BAT_STR_LOOP(b, flt, "%f"); break;
                case NPY_DOUBLE:
                case NPY_LONGDOUBLE: NP_COL_BAT_STR_LOOP(b, dbl, "%lf"); break;
                case NPY_STRING:
                    for (iu = 0; iu < ret->count; iu++) {
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {
                            b->T->nil = 1;
                            BUNappend(b, str_nil, FALSE);
                        }  else {
                            if (!string_copy(&data[(index_offset * ret->count + iu) * ret->memory_size], utf8_string, ret->memory_size)) {
                                msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");
                                goto wrapup;
                            }
                            BUNappend(b, utf8_string, FALSE);
                        }
                    }
                    break;
                case NPY_UNICODE:
                    for (iu = 0; iu < ret->count; iu++) {
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {
                            b->T->nil = 1;
                            BUNappend(b, str_nil, FALSE);
                        }  else {
                            utf32_to_utf8(0, ret->memory_size / 4, utf8_string, (const Py_UNICODE*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));
                            BUNappend(b, utf8_string, FALSE);
                        }
                    }
                    break;
                case NPY_OBJECT:
                {
                    //The resulting array is an array of pointers to various python objects
                    //Because the python objects can be of any size, we need to allocate a different size utf8_string for every object
                    //we will first loop over all the objects to get the maximum size needed, so we only need to do one allocation
                    size_t utf8_size = utf8string_minlength;
                    for (iu = 0; iu < ret->count; iu++) {
                        size_t size = utf8string_minlength;
                        PyObject *obj;
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) continue;
                        obj = *((PyObject**) &data[(index_offset * ret->count + iu) * ret->memory_size]);
                        if (PyString_CheckExact(obj) || PyByteArray_CheckExact(obj)) {
                            size = Py_SIZE(obj);     //Normal strings are 1 string per character
                        } else if (PyUnicode_CheckExact(obj)) {
                            size = Py_SIZE(obj) * 4; //UTF32 is 4 bytes per character
                        }
                        if (size > utf8_size) utf8_size = size;
                    }
                    utf8_string = GDKzalloc(utf8_size);
                    for (iu = 0; iu < ret->count; iu++) {
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {
                            b->T->nil = 1;
                            BUNappend(b, str_nil, FALSE);
                        } else {
                            //we try to handle as many types as possible
                            PyObject *obj = *((PyObject**) &data[(index_offset * ret->count + iu) * ret->memory_size]);
                            if (PyString_CheckExact(obj)) {
                                char *str = ((PyStringObject*)obj)->ob_sval;
                                if (!string_copy(str, utf8_string, strlen(str) + 1)) {
                                    msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");
                                    goto wrapup;
                                }
                            } else if (PyByteArray_CheckExact(obj)) {
                                char *str = ((PyByteArrayObject*)obj)->ob_bytes;
                                if (!string_copy(str, utf8_string, strlen(str) + 1)) {
                                    msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");
                                    goto wrapup;
                                }
                            } else if (PyUnicode_CheckExact(obj)) {
                                Py_UNICODE *str = (Py_UNICODE*)((PyUnicodeObject*)obj)->str;
                                utf32_to_utf8(0, ((PyUnicodeObject*)obj)->length, utf8_string, str);
                            } else if (PyBool_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj) || PyFloat_Check(obj)) {
#ifdef HAVE_HGE
                                hge h;
                                pyobject_to_hge(&obj, 0, &h);
                                hge_to_string(utf8_string, h);
#else
                                lng h;
                                pyobject_to_lng(&obj, 0, &h);
                                snprintf(utf8_string, utf8string_minlength, LLFMT, h);
#endif
                            } else {
                                msg = createException(MAL, "pyapi.eval", "Unrecognized Python object. Could not convert to NPY_UNICODE.\n");
                                goto wrapup;
                            }
                            BUNappend(b, utf8_string, FALSE);
                        }
                    }
                    break;
                }
                default:
                    msg = createException(MAL, "pyapi.eval", "Unrecognized type. Could not convert to NPY_UNICODE.\n");
                    goto wrapup;
            }
            GDKfree(utf8_string);

            b->T->nonil = 1 - b->T->nil;
            BATsetcount(b, ret->count);
            BATsettrivprop(b);
            break;
        }
    default:
        msg = createException(MAL, "pyapi.eval", "Unrecognized BAT type %s.\n", BatType_Format(bat_type));
        goto wrapup;
    }
    return b;
wrapup:
    *return_message = msg;
    return NULL;
}
