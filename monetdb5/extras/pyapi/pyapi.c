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
 
#undef _GNU_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define PYAPI_VERBOSE
#define _PYAPI_DEBUG_

#include <string.h>

 #include <semaphore.h>

const char* pyapi_enableflag = "embedded_py";
char *NPYConstToString(int);
char *BATConstToString(int);
bool IsPyScalar(PyObject *object);
bool IsNPYArray(PyObject *object);
bool IsNPYMaskedArray(PyObject *object);
bool IsPandasDataFrame(PyObject *object);
int snprintf_huge(char *, int, hge );
bool string_to_hge(char* , hge *);
bool PyType_IsInteger(int);
bool PyType_IsFloat(int);
bool PyType_IsDouble(int);

int PyAPIEnabled(void) {
    return (GDKgetenv_istrue(pyapi_enableflag)
            || GDKgetenv_isyes(pyapi_enableflag));
}

#ifdef WIN32
//windows
#else
//linux
#endif

#ifdef PYAPI_VERBOSE
#define VERBOSE_MESSAGE(args...) { \
    printf(args); \
    fflush(stdout); \
}
#else
#define VERBOSE_MESSAGE(args...) ((void) 0)
#endif


// TODO: exclude pyapi from mergetable, too
// TODO: add to SQL layer
// TODO: can we call the Python interpreter in a multi-thread environment?
static MT_Lock pyapiLock;
static MT_Lock pyapiSluice;
static int pyapiInitialized = FALSE;


#define BAT_TO_NP(bat, mtpe, nptpe)                                   \
        PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {BATcount(bat)},  \
        nptpe, NULL, (mtpe*) Tloc(bat, BUNfirst(bat)), 0,             \
        NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);

#define NP_PREPARE_DATA_COL(mtpe, check) {                            \
        if (!check(resultType))                                       \
        {                                                             \
            printf("!WARNING: Storing an array of type \"%s\" into a BAT of type \"%s\" without converting, this will likely result in unwanted behavior (this should probably be an error).\n", NPYConstToString(resultType), #mtpe); \
        }                                                             \
        ret->count = PyArray_DIMS((PyArrayObject*)pResult)[1];        \
        ret->numpy_array = (PyArrayObject*)pResult;                   \
        ret->numpy_mask = (PyArrayObject*)pMask;                      \
        ret->memory_size = PyArray_DESCR(ret->numpy_array)->elsize;   \
    }

#define NP_PREPARE_DATA(nptpe) {                                      \
        ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO,    \
            PyArray_DescrFromType(nptpe), 1, 1, NPY_ARRAY_CARRAY |    \
            NPY_ARRAY_FORCECAST, NULL);                               \
        if (ret->numpy_array == NULL)                                 \
        {                                                             \
            ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO, NULL, 1, 1,  NPY_ARRAY_CARRAY, NULL);  \
            if (ret->numpy_array != NULL && nptpe == NPY_UNICODE && PyArray_DTYPE(ret->numpy_array)->type_num == NPY_STRING) \
            {                                                         \
                msg = createException(MAL, "pyapi.eval", "Could not convert the string array to UTF-8. We currently only support UTF-8 formatted strings."); \
                goto wrapup;                                          \
            }                                                         \
            else                                                      \
            {                                                         \
                msg = createException(MAL, "pyapi.eval", "Wrong return type in python function. Expected an array of type \"%s\" as return value, but the python function returned an array of type \"%s\".", #nptpe, NPYConstToString(PyArray_DTYPE(ret->numpy_array)->type_num));       \
                goto wrapup;                                          \
            }                                                         \
        }                                                             \
        ret->count = PyArray_DIMS(ret->numpy_array)[0];               \
        if (PyObject_HasAttrString(pColO, "mask"))                    \
        {                                                             \
            pMask = PyObject_GetAttrString(pColO, "mask");            \
            if (pMask != NULL)                                        \
            {                                                         \
                ret->numpy_mask = (PyArrayObject*) PyArray_FromAny(pMask, PyArray_DescrFromType(NPY_BOOL), 1, 1,  NPY_ARRAY_CARRAY, NULL); \
                if (ret->numpy_mask == NULL || PyArray_DIMS(ret->numpy_mask)[0] != (int)ret->count)                                  \
                {                                                     \
                    pMask = NULL;                                     \
                    ret->numpy_mask = NULL;                           \
                    /*msg = createException(MAL, "pyapi.eval", "A masked array was returned, but the mask does not have the same length as the array.");*/  \
                    /*goto wrapup;*/                                  \
                }                                                     \
            }                                                         \
        }                                                             \
    }

#define NP_TO_BAT(bat, mtpe, nptpe) {                                 \
        pCol = (PyArrayObject*) PyArray_FromAny(pColO, \
            PyArray_DescrFromType(nptpe), 1, 1, NPY_ARRAY_CARRAY |    \
            NPY_ARRAY_FORCECAST, NULL);                               \
        if (pCol == NULL)                                             \
        {                                                             \
            pCol = (PyArrayObject*) PyArray_FromAny(pColO, NULL, 1, 1,  NPY_ARRAY_CARRAY, NULL);  \
            if (pCol != NULL && nptpe == NPY_UNICODE && PyArray_DTYPE(pCol)->type_num == NPY_STRING) \
            {                                                                   \
                msg = createException(MAL, "pyapi.eval", "Could not convert the string array to UTF-8. We currently only support UTF-8 formatted strings."); \
                goto wrapup;                                          \
            }                                                           \
            else                                                        \
            {                                                           \
                msg = createException(MAL, "pyapi.eval", "Wrong return type in python function. Expected an array of type \"%s\" as return value, but the python function returned an array of type \"%s\".", #nptpe, NPYConstToString(PyArray_DTYPE(pCol)->type_num));       \
                goto wrapup;                                              \
            }                                                           \
        }                                                             \
        count = PyArray_DIMS(pCol)[0];                                \
        bat = BATnew(TYPE_void, TYPE_##mtpe, count, TRANSIENT);         \
        if (PyObject_HasAttrString(pColO, "mask"))                      \
        {                                                                 \
            pMask = PyObject_GetAttrString(pColO, "mask");                \
            if (pMask != NULL)                                            \
            {                                                             \
                pMaskArray = (PyArrayObject*) PyArray_FromAny(pMask, PyArray_DescrFromType(NPY_BOOL), 1, 1,  NPY_ARRAY_CARRAY, NULL); \
                if (pMaskArray == NULL || PyArray_DIMS(pMaskArray)[0] != (int)count)                                  \
                {                                                         \
                    pMask = NULL; \
                    pMaskArray = NULL; \
                    /*msg = createException(MAL, "pyapi.eval", "A masked array was returned, but the mask does not have the same length as the array.");*/  \
                    /*goto wrapup;*/                                       \
                }                                                         \
            }                                                             \
        }                                                                  \
    }

#define NP_CREATE_BAT(bat, mtpe, nptpe) {                               \
        bool *mask = NULL; \
        mtpe *data = NULL; \
        if (ret->numpy_mask != NULL) \
        { \
            mask = (bool*)PyArray_DATA(ret->numpy_mask); \
        } \
        if (ret->numpy_array == NULL) \
        { \
            msg = createException(MAL, "pyapi.eval", "No return value stored in the structure.\n");       \
            goto wrapup;    \
        } \
        bat = BATnew(TYPE_void, TYPE_##mtpe, ret->count, TRANSIENT);         \
        data = (mtpe*) PyArray_DATA(ret->numpy_array); \
        BATseqbase(bat, 0); bat->T->nil = 0; bat->T->nonil = 1;       \
        bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;         \
        for (j =0; j < ret->count; j++)                                      \
        {                                                             \
            ((mtpe*) Tloc(bat, BUNfirst(bat)))[j] = (data[j]); \
            if (mask != NULL && (mask[j]) == TRUE) \
            {                                                         \
                bat->T->nil = 1;                                       \
                ((mtpe*) Tloc(bat, BUNfirst(bat)))[j] = mtpe##_nil;    \
                                                                      \
            }                                                         \
        } bat->T->nonil = 1 - bat->T->nil;                            \
        BATsetcount(bat, ret->count); \
        BATsettrivprop(bat); }

#define NP_CREATE_BAT_COL(bat, mtpe, nptpe) {                               \
        bool *mask = NULL; \
        char *data = NULL; \
        if (ret->numpy_mask != NULL) \
        { \
            mask = (bool*)PyArray_DATA(ret->numpy_mask); \
        } \
        if (ret->numpy_array == NULL) \
        { \
            msg = createException(MAL, "pyapi.eval", "No return value stored in the structure.\n");       \
            goto wrapup;    \
        } \
        bat = BATnew(TYPE_void, TYPE_##mtpe, ret->count, TRANSIENT);         \
        data = (char*) PyArray_DATA(ret->numpy_array); \
        BATseqbase(bat, 0); bat->T->nil = 0; bat->T->nonil = 1;       \
        bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;         \
        for (j =0; j < ret->count; j++)                                      \
        {                                                             \
            ((mtpe*) Tloc(bat, BUNfirst(bat)))[j] = *(mtpe*)(&data[(i*ret->count + j) * ret->memory_size]); \
            if (mask != NULL && (mask[i*ret->count + j]) == TRUE) \
            {                                                         \
                bat->T->nil = 1;                                       \
                ((mtpe*) Tloc(bat, BUNfirst(bat)))[j] = mtpe##_nil;                                 \
                                                                      \
            }                                                         \
        } bat->T->nonil = 1 - bat->T->nil;                            \
        BATsetcount(bat, ret->count); \
        BATsettrivprop(bat); }

str PyAPIeval(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped, bit mapped);

str 
PyAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 0, 0);
}

str 
PyAPIevalStdMap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 0, 1);
}

str 
PyAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 1, 0);
}

str 
PyAPIevalAggrMap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 1, 1);
}

typedef enum {
    NORMAL, SEENNL, INQUOTES, ESCAPED
} pyapi_scan_state;


struct _PyReturn{
    PyArrayObject *numpy_array;
    PyArrayObject *numpy_mask;
    size_t count;
    size_t memory_size;
    BAT *bat_return;
    bool multidimensional;
};
#define PyReturn struct _PyReturn
static bool Initialized = false;
static bool SemaphoreInitialized = false;
static int PassedSemaphore = 0;
static int MaxProcesses = 0;
static int Processes = 0;
sem_t execute_semaphore;

PyGILState_STATE AcquireLock(bool *holds_gil, bool first);
void ReleaseLock(PyGILState_STATE gstate, bool *holds_gil, bool final);

str PyAPIeval(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped, bit mapped) {
    sql_func * sqlfun = *(sql_func**) getArgReference(stk, pci, pci->retc);
    str exprStr = *getArgReference_str(stk, pci, pci->retc + 1);

    int i = 1, ai = 0;
    char argbuf[64];
    char argnames[1000] = "";
    size_t pos;
    char* pycall = NULL;
    char *expr_ind = NULL;
    size_t pycalllen, expr_ind_len;
    str *args;
    char *msg = MAL_SUCCEED;
    BAT *b = NULL;
    node * argnode;
    int seengrp = FALSE;
    PyObject *pArgs, *pResult; // this is going to be the parameter tuple
    BUN p = 0, q = 0;
    BATiter li;
    PyGILState_STATE gstate;
    bool holds_gil = FALSE;
    PyReturn *pyreturn_values = NULL;

    size_t count;
    size_t maxsize;
    size_t j;

    if (!PyAPIEnabled()) {
        throw(MAL, "pyapi.eval",
              "Embedded Python has not been enabled. Start server with --set %s=true",
              pyapi_enableflag);
    }

    VERBOSE_MESSAGE("PyAPI Start\n");

    pycalllen = strlen(exprStr) + sizeof(argnames) + 1000;
    expr_ind_len = strlen(exprStr) + 1000;

    pycall =      GDKzalloc(pycalllen);
    expr_ind =    GDKzalloc(expr_ind_len);
    args = (str*) GDKzalloc(sizeof(str) * pci->argc);
    pyreturn_values = GDKzalloc(sizeof(PyReturn) * pci->retc);

    if (args == NULL || pycall == NULL || pyreturn_values == NULL) {
        throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        // TODO: free args and rcall
    }


    // first argument after the return contains the pointer to the sql_func structure
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

    // the first unknown argument is the group, we don't really care for the rest.
    for (i = pci->retc + 2; i < pci->argc; i++) {
        if (args[i] == NULL) {
            if (!seengrp && grouped) {
                args[i] = GDKstrdup("aggr_group");
                seengrp = TRUE;
            } else {
                snprintf(argbuf, sizeof(argbuf), "arg%i", i - pci->retc - 1);
                args[i] = GDKstrdup(argbuf);
            }
        }
    }


    VERBOSE_MESSAGE("Formatting python code.\n");

    // create argument list
    pos = 0;
    for (i = pci->retc + 2; i < pci->argc && pos < sizeof(argnames); i++) {
        pos += snprintf(argnames + pos, sizeof(argnames) - pos, "%s%s",
                        args[i], i < pci->argc - 1 ? ", " : "");
    }
    if (pos >= sizeof(argnames)) {
        msg = createException(MAL, "pyapi.eval", "Command too large");
        goto wrapup;
    }

    {

        // indent every line in the expression by one level,
        // if we find newline-tab, use tab, space otherwise
        // two passes, first inserts null placeholder, second replaces
        // need to be careful, newline might be in a quoted string
        // this does not handle multi-line strings starting with """ (yet?)
        pyapi_scan_state state = SEENNL;
        char indentchar = 0;
        size_t py_pos, py_ind_pos = 0;

        if (strlen(exprStr) > 0 && exprStr[0] == '{')
            exprStr[0] = ' ';
        if (strlen(exprStr) > 2 && exprStr[strlen(exprStr) - 2] == '}')
            exprStr[strlen(exprStr) - 2] = ' ';

        for (py_pos = 0; py_pos < strlen(exprStr); py_pos++) 
        {
            if (exprStr[py_pos] == ';')
                exprStr[py_pos] = ' ';
        }

        for (py_pos = 0; py_pos < strlen(exprStr); py_pos++) {
            // +1 because we need space for the \0 we append below.
            if (py_ind_pos + 1 > expr_ind_len) {
                msg = createException(MAL, "pyapi.eval", "Overflow in re-indentation");
                goto wrapup;
            }
            switch(state) {
                case NORMAL:
                    if (exprStr[py_pos] == '\'' || exprStr[py_pos] == '"') {
                        state = INQUOTES;
                    }
                    if (exprStr[py_pos] == '\n') {
                        state = SEENNL;
                    }
                    break;

                case INQUOTES:
                    if (exprStr[py_pos] == '\\') {
                        state = ESCAPED;
                    }
                    if (exprStr[py_pos] == '\'' || exprStr[py_pos] == '"') {
                        state = NORMAL;
                    }
                    break;

                case ESCAPED:
                    state = INQUOTES;
                    break;

                case SEENNL:
                    if (exprStr[py_pos] == ' ' || exprStr[py_pos] == '\t') {
                        indentchar = exprStr[py_pos];
                    }
                    expr_ind[py_ind_pos++] = 0;
                    state = NORMAL;
                    break;
            }
            expr_ind[py_ind_pos++] = exprStr[py_pos];
        }
        if (indentchar == 0) {
            indentchar = ' ';
        }
        for (py_pos = 0; py_pos < py_ind_pos; py_pos++) {
            if (expr_ind[py_pos] == 0) {
                expr_ind[py_pos] = indentchar;
            }
        
}        // make sure this is terminated.
        expr_ind[py_ind_pos++] = 0;
    }

    if (snprintf(pycall, pycalllen,
         "def pyfun(%s):\n%s",
         argnames, expr_ind) >= (int) pycalllen) {
        msg = createException(MAL, "pyapi.eval", "Command too large");
        goto wrapup;
    }

    if (mapped)
    {
        MT_lock_set(&pyapiSluice, "pyapi.sluice");
        if (SemaphoreInitialized == FALSE)
        {
            MaxProcesses = 8; //todo: replace 8 with #number of processes
            PassedSemaphore = 0;

            sem_init(&execute_semaphore, 0, 0); //initialize execute semaphore

            SemaphoreInitialized = TRUE;
        }
        MT_lock_unset(&pyapiSluice, "pyapi.sluice");
    }

    gstate = AcquireLock(&holds_gil, true);

    VERBOSE_MESSAGE("Loading data from the database into Python.\n");

    // create function argument tuple, we pass a tuple of numpy arrays
    pArgs = PyTuple_New(pci->argc-(pci->retc + 2));

    // for each input column (BAT):
    for (i = pci->retc + 2; i < pci->argc; i++) {
        PyObject *vararray = NULL;
        // turn scalars into one-valued BATs
        // TODO: also do this for Python? Or should scalar values be 'simple' variables?
        if (!isaBatType(getArgType(mb,pci,i))) 
        {

            VERBOSE_MESSAGE("- Loading a scalar of type %s (%i)\n", BATConstToString(getArgType(mb,pci,i)), getArgType(mb,pci,i));

            //this is old code that converts a single input scalar into a BAT -> this is replaced by the code below, but I'll leave this here in case this might be preferable
            //the argument is a scalar, check which scalar type it is
            /*b = BATnew(TYPE_void, getArgType(mb, pci, i), 0, TRANSIENT);
            if (b == NULL) {
                msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                goto wrapup;
            }
            if ( getArgType(mb,pci,i) == TYPE_str)
                BUNappend(b, *getArgReference_str(stk, pci, i), FALSE);
            else
            {
                BUNappend(b, getArgReference(stk, pci, i), FALSE);
            }
            BATsetcount(b, 1);
            BATseqbase(b, 0);
            BATsettrivprop(b);
        } else {
            b = BATdescriptor(*getArgReference_bat(stk, pci, i));
            if (b == NULL) {
                msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                goto wrapup;
            }*/
            switch(getArgType(mb,pci,i))
            {
                case TYPE_bit:
                    vararray = PyInt_FromLong((long)(*(bit*)getArgReference(stk, pci, i)));
                    break;
                case TYPE_bte:
                    vararray = PyInt_FromLong((long)(*(bte*)getArgReference(stk, pci, i)));
                    break;
                case TYPE_sht:
                    vararray = PyInt_FromLong((long)(*(sht*)getArgReference(stk, pci, i))); 
                    break;
                case TYPE_int:
                    vararray = PyInt_FromLong((long)(*(int*)getArgReference(stk, pci, i)));
                    break;
                case TYPE_lng:
                    vararray = PyLong_FromLong((long)(*(lng*)getArgReference(stk, pci, i)));
                    break;
                case TYPE_flt:
                    vararray = PyFloat_FromDouble((double)(*(flt*)getArgReference(stk, pci, i)));
                    break;
                case TYPE_dbl:
                    vararray = PyFloat_FromDouble((double)(*(dbl*)getArgReference(stk, pci, i)));
                    break;
                case TYPE_hge:
                    {
                        char hex[40];
                        const hge *t = (const hge *) getArgReference(stk, pci, i);
                        snprintf_huge(hex, 40, *t);
                        //then we create a PyLong from that string by parsing it
                        vararray = PyLong_FromString(hex, NULL, 16);
                    }
                    break;
                case TYPE_str:
                    vararray = PyString_FromString((char*)getArgReference_str(stk, pci, i));
                    break;
                default:
                    msg = createException(MAL, "pyapi.eval", "Unsupported scalar type %i.", getArgType(mb,pci,i));
                    goto wrapup;
            }
            if (vararray == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Something went wrong converting the MonetDB scalar to a Python scalar.");
                goto wrapup;
            }
            PyTuple_SetItem(pArgs, ai++, vararray);
        }
        else
        {
            b = BATdescriptor(*getArgReference_bat(stk, pci, i));
            if (b == NULL) 
            {
                msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                goto wrapup;
            }

            VERBOSE_MESSAGE("- Loading a BAT of type %s (%i)\n", BATConstToString(ATOMstorage(getColumnType(getArgType(mb,pci,i)))), ATOMstorage(getColumnType(getArgType(mb,pci,i))));

            //the argument is a BAT, convert it to a numpy array
            switch (ATOMstorage(getColumnType(getArgType(mb,pci,i)))) {
            case TYPE_bte:
                vararray = BAT_TO_NP(b, bte, NPY_INT8);
                break;
            case TYPE_sht:
                vararray = BAT_TO_NP(b, sht, NPY_INT16);
                break;
            case TYPE_int:
                vararray = BAT_TO_NP(b, int, NPY_INT32);
                break;
            case TYPE_lng:
                vararray = BAT_TO_NP(b, lng, NPY_INT64);
                break;
            case TYPE_flt:
                vararray = BAT_TO_NP(b, flt, NPY_FLOAT32);
                break;
            case TYPE_dbl:
                vararray = BAT_TO_NP(b, dbl, NPY_FLOAT64);
                break;
            case TYPE_str:
                li = bat_iterator(b);

                //we first loop over all the strings in the BAT to find the maximum length of a single string
                //this is because NUMPY only supports strings with a fixed maximum length
                maxsize = 0;
                count = BATcount(b);
                BATloop(b, p, q)
                {
                    const char *t = (const char *) BUNtail(li, p);
                    const size_t length = (const size_t) strlen(t);

                    if (strlen(t) > maxsize)
                        maxsize = length;
                }

                //create a NPY_UNICODE array object
                vararray = PyArray_New(
                    &PyArray_Type, 
                    1, 
                    (npy_intp[1]) {count},  
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
                    const char *t = (const char *) BUNtail(li, p);
                    PyObject *obj;
                    if (strcmp(t, str_nil) == 0) 
                    {
                         //str_nil isn't a valid UTF-8 character (it's 0x80), so we need to decode it as Latin1
                        obj = PyUnicode_DecodeLatin1(t, strlen(t), "strict");
                    }
                    else
                    {
                        obj = PyUnicode_DecodeUTF8(t, strlen(t), "strict");
                    }
                    if (obj == NULL)
                    {
                        PyErr_Print();
                        msg = createException(MAL, "pyapi.eval", "Failed to decode string as UTF-8.");
                        goto wrapup;
                    }
                    PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
                    j++;
                }
                break;
            case TYPE_hge:
                li = bat_iterator(b);
                count = BATcount(b);

                //create a NPY_OBJECT array to hold the huge type
                vararray = PyArray_New(
                    &PyArray_Type, 
                    1, 
                    (npy_intp[1]) {count},  
                    NPY_OBJECT, 
                    NULL, 
                    NULL, 
                    128,          //128 bits per value
                    0, 
                    NULL);

                j = 0;
                printf("!WARNING: Type \"hge\" (128 bit) is unsupported by Numpy. The numbers are instead converted to python objects of type \"long\". This is likely very slow.\n");
                BATloop(b, p, q)
                {
                    //we first convert the huge to a string in hex format
                    char hex[40];
                    PyObject *obj;
                    const hge *t = (const hge *) BUNtail(li, p);
                    snprintf_huge(hex, 40, *t);
                    //then we create a PyLong from that string by parsing it
                    obj = PyLong_FromString(hex, NULL, 16);
                    if (obj == NULL)
                    {
                        PyErr_Print();
                        msg = createException(MAL, "pyapi.eval", "Failed to convert huge array.");
                        goto wrapup;
                    }
                    PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
                    j++;
                }
                break;

                /*
                //Convert huge to double, this might be preferrable so I'll leave this code here.//
                li = bat_iterator(b);
                count = BATcount(b);
                vararray = PyArray_New(
                    &PyArray_Type, 
                    1, 
                    (npy_intp[1]) {count},  
                    NPY_DOUBLE, 
                    NULL, 
                    NULL, 
                    0,          
                    0, 
                    NULL);
                j = 0;
                BATloop(b, p, q)
                {

                    const hge *t = (const hge *) BUNtail(li, p);
                    PyObject *obj = PyFloat_FromDouble((double) *t);
                    if (obj == NULL)
                    {
                        PyErr_Print();
                        msg = createException(MAL, "pyapi.eval", "Failed to convert huge array.");
                        goto wrapup;
                    }
                    PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
                    j++;
                }
                printf("!WARNING: BATs of type \"hge\" (128 bits) are converted to type \"double\" (64 bits), information might be lost.\n");
                break;*/
            default:
                msg = createException(MAL, "pyapi.eval", "unknown argument type ");
                goto wrapup;
            }

               
            // we use numpy.ma to deal with possible NULL values in the data
            // once numpy comes with proper NA support, this will change
            if (b->T->nil)
            {
                PyObject *mafunc = PyObject_GetAttrString(PyImport_Import(
                        PyString_FromString("numpy.ma")), "masked_array");
                PyObject *maargs = PyTuple_New(2);
                PyArrayObject* nullmask = (PyArrayObject*) PyArray_ZEROS(1,
                                (npy_intp[1]) {BATcount(b)}, NPY_BOOL, 0);

                const void *nil = ATOMnilptr(b->ttype);
                int (*atomcmp)(const void *, const void *) = ATOMcompare(b->ttype);
                BATiter bi = bat_iterator(b);

                if (b->T->nil) 
                {
                    size_t j;
                    for (j = 0; j < BATcount(b); j++) {
                        if ((*atomcmp)(BUNtail(bi, BUNfirst(b) + j), nil) == 0) {
                            // Houston we have a NULL
                            PyArray_SETITEM(nullmask, PyArray_GETPTR1(nullmask, j), Py_True);
                        }
                    }
                }

                PyTuple_SetItem(maargs, 0, vararray);
                PyTuple_SetItem(maargs, 1, (PyObject*) nullmask);
                    
                vararray = PyObject_CallObject(mafunc, maargs);
                if (!vararray) {
                    msg = createException(MAL, "pyapi.eval", "UUUH");
                            goto wrapup;
                }
            }
            PyTuple_SetItem(pArgs, ai++, vararray);

            // TODO: we cannot clean this up just yet, there may be a shallow copy referenced in python.
            // TODO: do this later

            BBPunfix(b->batCacheid);

            //msg = createException(MAL, "pyapi.eval", "unknown argument type ");
            //goto wrapup;
        }
    }


    VERBOSE_MESSAGE("Executing python code.\n");

    {
        int pyret = 0;
        PyObject *pFunc, *pModule;

        // TODO: does this create overhead?, see if we can share the import
        pModule = PyImport_Import(PyString_FromString("__main__"));
        if (!Initialized)
        {
            VERBOSE_MESSAGE("Initializing function.\n");
            pyret = PyRun_SimpleString(pycall);

            Initialized = true;
        }
        pFunc = PyObject_GetAttrString(pModule, "pyfun");

        

        //fprintf(stdout, "%s\n", pycall);
        if (pyret != 0 || !pModule || !pFunc || !PyCallable_Check(pFunc)) {
            msg = createException(MAL, "pyapi.eval", "could not parse Python code %s", pycall);
            goto wrapup;
        }

        if (mapped)
        {
            PyObject *pMult, *pPoolClass, *pPoolProcesses, *pPoolObject, *pApply, *pPoolArgs, *pOutput, *pGet;

            pMult = PyImport_Import(PyString_FromString("multiprocessing"));
            if (pMult == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Failed to load module Multiprocessing");
                goto wrapup;
            }
            pPoolClass = PyObject_GetAttrString(pMult, "Pool");
            if (pPoolClass == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Failed to load Pool class");
                goto wrapup;
            }

            VERBOSE_MESSAGE("Create pool process.\n");
            pPoolProcesses = PyTuple_New(1);
            PyTuple_SetItem(pPoolProcesses, 0, PyInt_FromLong(1));
            pPoolObject = PyObject_CallObject(pPoolClass, pPoolProcesses);
            if (pPoolObject == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Failed to construct pool.");
                goto wrapup;
            }
            pApply = PyObject_GetAttrString(pPoolObject, "apply_async");
            if (pApply == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Failed.");
                goto wrapup;
            }
            pPoolArgs = PyTuple_New(2);

            PyTuple_SetItem(pPoolArgs, 0, pFunc);
            PyTuple_SetItem(pPoolArgs, 1, pArgs);

            pOutput = PyObject_CallObject(pApply, pPoolArgs);

            pGet = PyObject_GetAttrString(pOutput, "get");
            if (pGet == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "No get?");
                goto wrapup;
            }

            PassedSemaphore++;
            ReleaseLock(gstate, &holds_gil, false);

            if (PassedSemaphore != MaxProcesses)
            {
                VERBOSE_MESSAGE("Start waiting, %i processes passed semaphore\n", PassedSemaphore);
                sem_wait(&execute_semaphore);
            }

            gstate = AcquireLock(&holds_gil, false);

            pResult = PyObject_CallObject(pGet, NULL);
        }
        else
        {
            pResult = PyObject_CallObject(pFunc, pArgs);
        }
        if (PyErr_Occurred()) {
            PyObject *pErrType, *pErrVal, *pErrTb;
            PyErr_Fetch(&pErrType, &pErrVal, &pErrTb);
            if (pErrVal) {
                msg = createException(MAL, "pyapi.eval", "Python exception: %s", PyString_AS_STRING(PyObject_Str(pErrVal)));
            } else {
                msg = createException(MAL, "pyapi.eval", "Python exception: ?");
            }
            goto wrapup; // shudder
        }

        if (pResult) 
        {
            PyObject * pColO = NULL;

            if (IsPandasDataFrame(pResult))
            {
                //the result object is a Pandas data frame
                //we can convert the pandas data frame to a numpy array by simply accessing the "values" field (as pandas dataframes are numpy arrays internally)
                pResult = PyObject_GetAttrString(pResult, "values"); 
                if (pResult == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Invalid Pandas data frame.");
                    goto wrapup; 
                }
                //we transpose the values field so it's aligned correctly for our purposes
                pResult = PyObject_GetAttrString(pResult, "T");
                if (pResult == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Invalid Pandas data frame.");
                    goto wrapup; 
                }
            }

            if (IsPyScalar(pResult)) //check if the return object is a scalar
            {
                if (pci->retc == 1) 
                {
                    //if we only expect a single return value, we can accept scalars by converting it into an array holding an array holding the element (i.e. [[pResult]])
                    PyObject *list = PyList_New(1);
                    PyList_SetItem(list, 0, pResult);
                    pResult = list;

                    list = PyList_New(1);
                    PyList_SetItem(list, 0, pResult);
                    pResult = list;
                }
                else
                {
                    //the result object is a scalar, yet we expect more than one return value. We can only convert the result into a list with a single element, so the output is necessarily wrong.
                    msg = createException(MAL, "pyapi.eval", "A single scalar was returned, yet we expect a list of %d columns. We can only convert a single scalar into a single column, thus the result is invalid.", pci->retc);
                    goto wrapup;
                }
            }
            else
            {
                //if it is not a scalar, we check if it is a single array
                bool IsSingleArray = TRUE;
                PyObject *data = pResult;
                if (IsNPYMaskedArray(data))
                {
                    data = PyObject_GetAttrString(pResult, "data");   
                    if (data == NULL)
                    {
                        msg = createException(MAL, "pyapi.eval", "Invalid masked array.");
                        goto wrapup;
                    }           
                }

                if (IsNPYArray(data)) 
                {
                    if (PyArray_NDIM((PyArrayObject*)data) != 1)
                    {
                        IsSingleArray = FALSE;
                    }
                    else
                    {
                        pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, 0));
                        IsSingleArray = IsPyScalar(pColO);
                    }
                }
                else if (PyList_Check(data)) 
                {
                    pColO = PyList_GetItem(data, 0);
                    IsSingleArray = IsPyScalar(pColO);
                }
                else if (!IsNPYMaskedArray(data))
                {
                    //it is neither a python array, numpy array or numpy masked array, thus the result is unsupported! Throw an exception!
                    msg = createException(MAL, "pyapi.eval", "Unsupported result object. Expected either an array, a numpy array, a numpy masked array or a pandas data frame, but received an object of type \"%s\"", PyString_AsString(PyObject_Str(PyObject_Type(data))));
                    goto wrapup;
                }

                if (IsSingleArray)
                {
                    if (pci->retc == 1)
                     {
                        //if we only expect a single return value, we can accept a single array by converting it into an array holding an array holding the element (i.e. [pResult])
                        PyObject *list = PyList_New(1);
                        PyList_SetItem(list, 0, pResult);
                        pResult = list;
                    }
                    else
                    {
                        //the result object is a single array, yet we expect more than one return value. We can only convert the result into a list with a single array, so the output is necessarily wrong.
                        msg = createException(MAL, "pyapi.eval", "A single array was returned, yet we expect a list of %d columns. The result is invalid.", pci->retc);
                        goto wrapup;
                    }
                }
                else
                {
                    //the return value is an array of arrays, all we need to do is check if it is the correct size
                    int results = 0;
                    if (PyList_Check(data)) results = PyList_Size(data);
                    else results = PyArray_DIMS((PyArrayObject*)data)[0];
                    if (results != pci->retc)
                    {
                        //wrong return size, we expect pci->retc arrays
                        msg = createException(MAL, "pyapi.eval", "An array of size %d was returned, yet we expect a list of %d columns. The result is invalid.", results, pci->retc);
                        goto wrapup;
                    }
                }
            }
        }
        else
        {
            msg = createException(MAL, "pyapi.eval", "Invalid result object. No result object could be generated.");
            goto wrapup;
        }
        // delete the function again
        //PyRun_SimpleString("try:\n\tdel pyfun\nexcept NameError:\n\tpass");
    }

    VERBOSE_MESSAGE("Collecting return values.\n");

    for (i = 0; i < pci->retc; i++) 
    {
        PyArrayObject *pCol = NULL;
        PyArrayObject *pMaskArray = NULL;
        PyObject *pMask = NULL;
        PyObject * pColO = NULL;
        PyReturn *ret = &pyreturn_values[i];
        int resultType = 0;
        int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));

        ret->bat_return = NULL;
        ret->multidimensional = FALSE;

        if (PyList_Check(pResult)) 
        {
            //if it is a PyList, get the i'th array from the PyList
            pColO = PyList_GetItem(pResult, i);
        }
        else 
        {
            //if it isn't it's either a NPYArray or a NPYMaskedArray
            PyObject *data = pResult;
            if (IsNPYMaskedArray(data))
            {
                data = PyObject_GetAttrString(pResult, "data");   
                pMask = PyObject_GetAttrString(pResult, "mask");    
            }


            //we can either have a multidimensional numpy array, or a single dimensional numpy array 
            if (PyArray_NDIM((PyArrayObject*)data) != 1)
            {
                //if it is a multidimensional numpy array, we have to convert the i'th dimension to a NUMPY array object
                ret->multidimensional = TRUE;
                resultType = PyArray_DESCR((PyArrayObject*)data)->type_num;
            }
            else
            {
                //if it is a single dimensional numpy array, we get the i'th array from the NPYArray (this is a list)
                pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, i));
            }
        }
        switch (bat_type) 
        {
        case TYPE_bte:
            if (ret->multidimensional) 
            {
                NP_PREPARE_DATA_COL(bit, PyType_IsInteger);
            }
            else 
            {
                NP_PREPARE_DATA(NPY_INT8);
            }
            break;
        case TYPE_sht:
            if (ret->multidimensional) 
            {
                NP_PREPARE_DATA_COL(sht, PyType_IsInteger);
            }
            else 
            {
                NP_PREPARE_DATA(NPY_INT16);
            }
            break;
        case TYPE_int:
            if (ret->multidimensional) 
            {
                NP_PREPARE_DATA_COL(int, PyType_IsInteger);
            }
            else 
            {
                NP_PREPARE_DATA(NPY_INT32);
            }
            break;
        case TYPE_lng:
            if (ret->multidimensional) 
            {
                NP_PREPARE_DATA_COL(lng, PyType_IsInteger);
            }
            else 
            {
                NP_PREPARE_DATA(NPY_INT64);
            }
            break;
        case TYPE_flt:
            if (ret->multidimensional) 
            {
                NP_PREPARE_DATA_COL(flt, PyType_IsFloat);
            }
            else 
            {
                NP_PREPARE_DATA(NPY_FLOAT);
            }
            break;
        case TYPE_dbl:
            if (ret->multidimensional) 
            {
                NP_PREPARE_DATA_COL(dbl, PyType_IsDouble);
            }
            else 
            {
                NP_PREPARE_DATA(NPY_DOUBLE);
            }
            break;
        case TYPE_hge:
            if (ret->multidimensional)
            {
                count = PyArray_DIMS((PyArrayObject*)pResult)[1];
                b = BATnew(TYPE_void, TYPE_hge, count, TRANSIENT);

                BATseqbase(b, 0); 
                b->T->nil = 0; b->tkey = 0; b->tsorted = 0; b->trevsorted = 0; b->tdense = 0; b->T->nonil = 1;       

                for (j = 0; j < count; j++) 
                {
                    PyObject *obj = PyArray_GETITEM((PyArrayObject*)pResult, PyArray_GETPTR2((PyArrayObject*)pResult, i, j));
                    hge h;
                    if (!string_to_hge(PyString_AsString(PyObject_Str(obj)), &h))
                    {
                        msg = createException(MAL, "pyapi.eval", "Could not convert the string \"%s\" to a hge value.", PyString_AsString(PyObject_Str(obj)));
                        goto wrapup;
                    }   
                    BUNappend(b, &h, FALSE);
                }      
                BATsetcount(b, count); 
            }
            else
            {
                NP_TO_BAT(b, hge, NPY_OBJECT);

                b = BATnew(TYPE_void, TYPE_hge, count, TRANSIENT);

                BATseqbase(b, 0); 
                b->T->nil = 0; b->tkey = 0; b->tsorted = 0; b->trevsorted = 0; b->tdense = 0;

                for (j = 0; j < count; j++) 
                {
                    //first check if the masked array contains 'TRUE' here
                    if (pMaskArray != NULL && PyArray_GETITEM(pMaskArray, PyArray_GETPTR1(pMaskArray, j)) == Py_True) 
                    {                                  
                        //if it is, we have found a NULL value, append str_nil to the BAT                   
                        b->T->nil = 1; 
                        BUNappend(b, &hge_nil, FALSE);  
                    }
                    else
                    {
                        //if the masked array contains FALSE, then append the actual string to the BAT
                        PyObject *obj = PyArray_GETITEM(pCol, PyArray_GETPTR1(pCol, j));
                        hge h;
                        if (!string_to_hge(PyString_AsString(PyObject_Str(obj)), &h))
                        {
                            msg = createException(MAL, "pyapi.eval", "Could not convert the string \"%s\" to a hge value.", PyString_AsString(PyObject_Str(obj)));
                            goto wrapup;
                        }   
                        BUNappend(b, &h, FALSE);
                    }
                }     
                b->T->nonil = 1 - b->T->nil;        
                BATsetcount(b, count); 
            }
            ret->bat_return = b;
            break;
        case TYPE_str:
            if (ret->multidimensional) 
            {
                count = PyArray_DIMS((PyArrayObject*)pResult)[1];
                b = BATnew(TYPE_void, TYPE_str, count, TRANSIENT);

                BATseqbase(b, 0); 
                b->T->nil = 0; b->tkey = 0; b->tsorted = 0; b->trevsorted = 0; b->tdense = 0; b->T->nonil = 1;

                for (j = 0; j < count; j++) 
                {
                    if (PyArray_DTYPE((PyArrayObject*)pResult)->type_num == NPY_UNICODE)
                    {
                        PyObject *obj = PyUnicode_AsUTF8String(PyArray_GETITEM((PyArrayObject*)pResult, PyArray_GETPTR2((PyArrayObject*)pResult, i, j)));
                        BUNappend(b, PyString_AsString(PyObject_Str(obj)), FALSE);
                    }
                    else
                    {
                        const char *str = PyString_AsString(PyObject_Str(PyArray_GETITEM((PyArrayObject*)pResult, PyArray_GETPTR2((PyArrayObject*)pResult, i, j))));
                        PyObject *test = PyUnicode_FromString(str);
                        if (test == NULL)
                        {
                            msg = createException(MAL, "pyapi.eval", "Could not convert the string \"%s\" to UTF-8. We currently only support UTF-8 formatted strings.", str);
                            goto wrapup;
                        }
                        BUNappend(b, str, FALSE);
                    }
                }     
                BATsetcount(b, count); 
            }
            else 
            {
                NP_TO_BAT(b, str, NPY_UNICODE);

                b = BATnew(TYPE_void, TYPE_str, count, TRANSIENT);

                BATseqbase(b, 0); 
                b->T->nil = 0; b->tkey = 0; b->tsorted = 0; b->trevsorted = 0; b->tdense = 0;

                for (j = 0; j < count; j++) 
                {
                    //first check if the masked array contains 'TRUE' here
                    if (pMaskArray != NULL && PyArray_GETITEM(pMaskArray, PyArray_GETPTR1(pMaskArray, j)) == Py_True) 
                    {                                  
                        //if it is, we have found a NULL value, append str_nil to the BAT                   
                        b->T->nil = 1; 
                        BUNappend(b, str_nil, FALSE);  
                    }
                    else
                    {
                        //if the masked array contains FALSE, then append the actual string to the BAT
                        PyObject *obj = PyUnicode_AsUTF8String(PyArray_GETITEM(pCol, PyArray_GETPTR1(pCol, j)));
                        BUNappend(b, PyString_AsString(PyObject_Str(obj)), FALSE);
                    }
                }     
                b->T->nonil = 1 - b->T->nil;        
                BATsetcount(b, count); 
            }
            ret->bat_return = b;
            break;
        default:
            msg = createException(MAL, "pyapi.eval",
                                  "unknown return type for return argument %d: %d", i,
                                  bat_type);
            goto wrapup;
        }
    }


    ReleaseLock(gstate, &holds_gil, true);

    VERBOSE_MESSAGE("Returning values.\n");

    for (i = 0; i < pci->retc; i++) 
    {
        PyReturn *ret = &pyreturn_values[i];
        int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));
        b = ret->bat_return;    


        switch (bat_type) 
        {
        case TYPE_bte:
            if (ret->multidimensional) 
            {
                NP_CREATE_BAT_COL(b, bit, NPY_INT8);
            }
            else 
            {
                NP_CREATE_BAT(b, bit, NPY_INT8);
            }
            break;
        case TYPE_sht:
            if (ret->multidimensional) 
            {
                NP_CREATE_BAT_COL(b, sht, NPY_INT16);
            }
            else 
            {
                NP_CREATE_BAT(b, sht, NPY_INT16);
            }
            break;
        case TYPE_int:
            if (ret->multidimensional) 
            {
                NP_CREATE_BAT_COL(b, int, NPY_INT32);
            }
            else 
            {
                NP_CREATE_BAT(b, int, NPY_INT32);
            }
            break;
        case TYPE_lng:
            if (ret->multidimensional) 
            {
                NP_CREATE_BAT_COL(b, lng, NPY_INT64);
            }
            else 
            {
                NP_CREATE_BAT(b, lng, NPY_INT64);
            }
            break;
        case TYPE_flt:
            if (ret->multidimensional) 
            {
                NP_CREATE_BAT_COL(b, flt, NPY_FLOAT32);
            }
            else 
            {
                NP_CREATE_BAT(b, flt, NPY_FLOAT32);
            }
            break;
        case TYPE_dbl:
            if (ret->multidimensional) 
            {
                NP_CREATE_BAT_COL(b, dbl, NPY_FLOAT64);
            }
            else 
            {
                NP_CREATE_BAT(b, dbl, NPY_FLOAT64);
            }
            break;
        }

        if (isaBatType(getArgType(mb,pci,i))) {
            *getArgReference_bat(stk, pci, i) = b->batCacheid;
            BBPkeepref(b->batCacheid);
        } else { // single value return, only for non-grouped aggregations
            VALinit(&stk->stk[pci->argv[i]], bat_type, Tloc(b, BUNfirst(b)));
        }
        msg = MAL_SUCCEED;
    }
  wrapup:

    VERBOSE_MESSAGE("Cleaning up.\n");
    ReleaseLock(gstate, &holds_gil, true);

    GDKfree(pyreturn_values);
    GDKfree(args);
    GDKfree(pycall);
    GDKfree(expr_ind);

    VERBOSE_MESSAGE("Finished cleaning up.\n");
    return msg;
}

str PyAPIprelude(void *ret) {
    (void) ret;
    MT_lock_init(&pyapiLock, "pyapi_lock");
    MT_lock_init(&pyapiSluice, "pyapi_sluice");
    if (PyAPIEnabled()) {
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        if (!pyapiInitialized) {
            char* iar = NULL;
            Py_Initialize();
            PyEval_InitThreads();
            import_array1(iar);
            PyRun_SimpleString("import numpy");
            PyEval_SaveThread();
            //PyEval_ReleaseLock();
            pyapiInitialized++;
        }
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
        fprintf(stdout, "# MonetDB/Python module loaded\n");
    }
    return MAL_SUCCEED;
}


char *NPYConstToString(int type)
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

char *BATConstToString(int type)
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
        case NPY_ULONGLONG: return TRUE;
        default: return FALSE;
    }
}

bool PyType_IsFloat(int type)
{
    switch (type)
    {
        case NPY_FLOAT16: 
        case NPY_FLOAT: return TRUE;
        default: return FALSE;
    }
}

bool PyType_IsDouble(int type)
{
    switch (type)
    {
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE: return TRUE;
        default: return FALSE;
    }
}

//Returns TRUE if the type of [object] is a scalar (i.e. numeric scalar or string, basically "not an array but a single value")
bool IsPyScalar(PyObject *object)
{
    PyArray_Descr *descr;

    if (object == NULL) return FALSE;
    if (PyList_Check(object)) return FALSE;
    if (PyObject_HasAttrString(object, "mask")) return FALSE;

    descr = PyArray_DescrFromScalar(object);
    if (descr == NULL) return FALSE;
    if (descr->type_num != NPY_OBJECT) return TRUE; //check if the object is a numpy scalar
    if (PyInt_Check(object) || PyFloat_Check(object) || PyLong_Check(object) || PyString_Check(object) || PyBool_Check(object) || PyUnicode_Check(object)) return TRUE;

    return FALSE;
}

bool IsPandasDataFrame(PyObject *object)
{
    return (strcmp(PyString_AsString(PyObject_Str(PyObject_Type(object))), "<class 'pandas.core.frame.DataFrame'>") == 0);
}

bool IsNPYArray(PyObject *object)
{
    return (strcmp(PyString_AsString(PyObject_Str(PyObject_Type(object))), "<type 'numpy.ndarray'>") == 0);
}

bool IsNPYMaskedArray(PyObject *object)
{
    return (strcmp(PyString_AsString(PyObject_Str(PyObject_Type(object))), "<class 'numpy.ma.core.MaskedArray'>") == 0);
}
   
int snprintf_huge(char * str, int size, hge x)
{
    int i = 0;
    for(i = 0; i < size - 1; i++) str[i] = '0';
    if (x < 0) 
    {
        x *= -1;
        str[0] = '-';
    }
    str[size - 1] = '\0';
    while(x > 0)
    {
        int v = x % 16;
        i--;
        if (i < 0) return FALSE;
        if (v == 0)       str[i] = '0';
        else if (v == 1)  str[i] = '1';
        else if (v == 2)  str[i] = '2';
        else if (v == 3)  str[i] = '3';
        else if (v == 4)  str[i] = '4';
        else if (v == 5)  str[i] = '5';
        else if (v == 6)  str[i] = '6';
        else if (v == 7)  str[i] = '7';
        else if (v == 8)  str[i] = '8';
        else if (v == 9)  str[i] = '9';
        else if (v == 10) str[i] = 'a';
        else if (v == 11) str[i] = 'b';
        else if (v == 12) str[i] = 'c';
        else if (v == 13) str[i] = 'd';
        else if (v == 14) str[i] = 'e';
        else if (v == 15) str[i] = 'f';
        x = x / 16;
    }
    return TRUE;
}

bool string_to_hge(char* str, hge *h)
{
    hge j = 1;
    int i;

    if (h == NULL) return FALSE;
    *h = 0;

    for(i = strlen(str) - 1; i >= 0; i--)
    {
        if (str[i] == '1') *h += j;
        else if (str[i] == '2') *h += j * 2;
        else if (str[i] == '3') *h += j * 3;
        else if (str[i] == '4') *h += j * 4;
        else if (str[i] == '5') *h += j * 5;
        else if (str[i] == '6') *h += j * 6;
        else if (str[i] == '7') *h += j * 7;
        else if (str[i] == '8') *h += j * 8;
        else if (str[i] == '9') *h += j * 9;
        else if (str[i] == '0') ;
        else if (str[i] == ',' || str[i] == '.') { *h = 0; j = 1; continue; }
        else if (str[i] == '-') *h *= -1;
        else return FALSE; //invalid string
        j *= 10;
    } 
    return TRUE;
}


PyGILState_STATE AcquireLock(bool *holds_gil, bool first)
{
    PyGILState_STATE gstate;
    if (*holds_gil == TRUE) 
    {
        VERBOSE_MESSAGE("Process already holds GIL!\n");
        return 0;
    }

    MT_lock_set(&pyapiLock, "pyapi.evaluate");
    //gstate = 1;
    gstate = PyGILState_Ensure();

    VERBOSE_MESSAGE("Acquired GIL lock.\n");

    *holds_gil = true;
    if (first)
    {
        Processes++;
        VERBOSE_MESSAGE("Processes: %i\n", Processes);
    }
    return gstate;
}

void ReleaseLock(PyGILState_STATE gstate, bool *holds_gil, bool final)
{
    if (*holds_gil == FALSE) return;

    if (final)
    {
        Processes--;
        VERBOSE_MESSAGE("Processes: %i\n", Processes);
        if (Processes == 0) 
        {
            Initialized = false;
            SemaphoreInitialized = false;
            PassedSemaphore = 0;
            PyRun_SimpleString("del pyfun");
        }
    }
    
    VERBOSE_MESSAGE("Releasing GIL lock.\n");

    PyGILState_Release(gstate);
    MT_lock_unset(&pyapiLock, "pyapi.evaluate");
    *holds_gil = FALSE;

    if (final)
    {
        sem_post(&execute_semaphore);
    }
}
