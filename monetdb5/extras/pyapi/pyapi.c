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

#include "unicode.h"
#include "pytypes.h"
#include "type_conversion.h"
#include "shared_memory.h"

//#define _PYAPI_VERBOSE_
#define _PYAPI_DEBUG_

#include <stdint.h>

#include <stdio.h>
#include <string.h>

#ifdef WIN32

#else
#include <sys/types.h>
#include <sys/wait.h>
#endif

const char* pyapi_enableflag = "embedded_py";

#ifdef _PYAPI_VERBOSE_
#define VERBOSE_MESSAGE(...) {   \
    if (shm_id > 0) printf("%d: ", shm_id); \
    printf(__VA_ARGS__);        \
    fflush(stdout);                   \
}
#else
#define VERBOSE_MESSAGE(...) ((void) 0)
#endif

#define GDK_Alloc(var, size) { \
    var = GDKzalloc(size);  \
    if (var == NULL) \
    { \
        msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL); \
        goto wrapup; \
    } \
}

#define GDK_Free(var) { \
    if (var != NULL) \
        GDKfree(var); \
}

struct _ReturnBatDescr
{
    int npy_type;                        //npy type 
    size_t element_size;                 //element size in bytes
    size_t bat_count;                     //number of elements in bat
    size_t bat_size;                     //bat size in bytes
    size_t bat_start;                    //start position of bat
    bool has_mask;                       //if the return value has a mask or not
};
#define ReturnBatDescr struct _ReturnBatDescr

struct _PyInput{
    void *dataptr;
    BAT *bat;
    int bat_type;
    size_t count;
    bool scalar;
};
#define PyInput struct _PyInput

struct _PyReturn{
    PyArrayObject *numpy_array;
    PyArrayObject *numpy_mask;
    void *array_data;
    bool *mask_data;
    size_t count;
    size_t memory_size;
    BAT *bat_return;
    int result_type;
    bool multidimensional;
};
#define PyReturn struct _PyReturn

int PyAPIEnabled(void) {
    return (GDKgetenv_istrue(pyapi_enableflag)
            || GDKgetenv_isyes(pyapi_enableflag));
}


// TODO: exclude pyapi from mergetable, too
static MT_Lock pyapiLock;
static MT_Lock pyapiSluice;
static int pyapiInitialized = FALSE;

#define BAT_TO_NP(bat, mtpe, nptpe)                                                                     \
        PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {(t_end-t_start)},                                  \
        nptpe, NULL, &((mtpe*) Tloc(bat, BUNfirst(bat)))[t_start], 0,                                   \
        NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);

#define BAT_MMAP(bat, mtpe, batstore) {                                                                 \
        bat = BATnew(TYPE_void, TYPE_##mtpe, 0, TRANSIENT);                                             \
        BATseqbase(bat, 0); bat->T->nil = 0; bat->T->nonil = 1;                                         \
        bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;                                           \
        /*Change nil values to the proper values, if they exist*/                                       \
        if (mask != NULL)                                                                               \
        {                                                                                               \
            for (iu = 0; iu < ret->count; iu++)                                                         \
            {                                                                                           \
                if (mask[index_offset * ret->count + iu] == TRUE)                                       \
                {                                                                                       \
                    (*(mtpe*)(&data[(index_offset * ret->count + iu) * ret->memory_size])) = mtpe##_nil;\
                    bat->T->nil = 1;                                                                    \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        bat->T->nonil = 1 - bat->T->nil;                                                                \
        /*When we create a BAT a small part of memory is allocated, free it*/                           \
        GDKfree(bat->T->heap.base);                                                                     \
                                                                                                        \
        bat->T->heap.base = &data[(index_offset * ret->count) * ret->memory_size];                      \
        bat->T->heap.size = ret->count * ret->memory_size;                                              \
        bat->T->heap.free = bat->T->heap.size;  /*There are no free places in the array*/               \
        /*If index_offset > 0, we are mapping part of a multidimensional array.*/                       \
        /*The entire array will be cleared when the part with index_offset=0 is freed*/                 \
        /*So we set this part of the mapping to 'NOWN'*/                                                \
        if (index_offset > 0) bat->T->heap.storage = STORE_NOWN;                                        \
        else bat->T->heap.storage = batstore;                                                           \
        bat->T->heap.newstorage = STORE_MEM;                                                            \
        bat->S->count = ret->count;                                                                     \
        bat->S->capacity = ret->count;                                                                  \
        bat->S->copiedtodisk = false;                                                                   \
                                                                                                        \
        /*Take over the data from the numpy array*/                                                     \
        if (ret->numpy_array != NULL) PyArray_CLEARFLAGS(ret->numpy_array, NPY_ARRAY_OWNDATA);          \
    }

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

#define NP_COL_BAT_LOOP_FUNC(bat, mtpe_to, func) {                                                                                                    \
    mtpe_to value;                                                                                                                                    \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (!func(&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                          \
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
                if (!func(&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                      \
                {                                                                                                                                     \
                    msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to); \
                    goto wrapup;                                                                                                                      \
                }                                                                                                                                     \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                    \
            }                                                                                                                                         \
        }                                                                                                                                             \
    } }
    

#define NP_COL_BAT_STR_LOOP(bat, mtpe, conv)                                                                                                          \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            conv(utf8_string, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                                                  \
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
                conv(utf8_string, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                                              \
                BUNappend(bat, utf8_string, FALSE);                                                                                                   \
            }                                                                                                                                         \
        }                                                                                                                                             \
    }

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
        if (TYPE_##mtpe == PyType_ToBat(ret->result_type) && (ret->count * ret->memory_size < BUN_MAX) &&                                                      \
            (ret->numpy_array == NULL || PyArray_FLAGS(ret->numpy_array) & NPY_ARRAY_OWNDATA))                                                                 \
        {                                                                                                                                                      \
            /*We can only create a direct map if the numpy array type and target BAT type*/                                                                    \
            /*are identical, otherwise we have to do a conversion.*/                                                                                           \
            if (ret->numpy_array == NULL)                                                                                                                      \
            {                                                                                                                                                  \
                /*shared memory return*/                                                                                                                       \
                VERBOSE_MESSAGE("Shared memory map!\n");                                                                                                       \
                BAT_MMAP(bat, mtpe, STORE_SHARED);                                                                                                             \
                ret->array_data = NULL;                                                                                                                        \
            }                                                                                                                                                  \
            else                                                                                                                                               \
            {                                                                                                                                                  \
                VERBOSE_MESSAGE("Memory map!\n");                                                                                                              \
                BAT_MMAP(bat, mtpe, STORE_CMEM);                                                                                                               \
            }                                                                                                                                                  \
        }                                                                                                                                                      \
        else                                                                                                                                                   \
        {                                                                                                                                                      \
            bat = BATnew(TYPE_void, TYPE_##mtpe, ret->count, TRANSIENT);                                                                                       \
            BATseqbase(bat, 0); bat->T->nil = 0; bat->T->nonil = 1;                                                                                            \
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
                case NPY_STRING:     NP_COL_BAT_LOOP_FUNC(bat, mtpe, str_to_##mtpe); break;                                                                    \
                case NPY_UNICODE:    NP_COL_BAT_LOOP_FUNC(bat, mtpe, unicode_to_##mtpe); break;                                                                \
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
PyAPIeval(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped, bit mapped);

str 
PyAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) 
{
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 0, 0);
}

str 
PyAPIevalStdMap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) 
{
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 0, 1);
}

str 
PyAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) 
{
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 1, 0);
}

str 
PyAPIevalAggrMap(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) 
{
    (void) cntxt;
    return PyAPIeval(mb, stk, pci, 1, 1);
}

typedef enum {
    NORMAL, SEENNL, INQUOTES, ESCAPED
} pyapi_scan_state;

bool PyType_IsPyScalar(PyObject *object);

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
    PyObject *pArgs, *pResult = NULL; // this is going to be the parameter tuple
    BUN p = 0, q = 0;
    BATiter li;
    PyReturn *pyreturn_values = NULL;
    PyInput *pyinput_values = NULL;

#ifndef WIN32
    bool single_fork = mapped == 1;
    int shm_id = -1;
    int sem_id = -1;
    int process_id = 0;
    int memory_size;
    int process_count;
#endif

    size_t count;
    size_t maxsize;
    int j;
    size_t iu;

    if (!PyAPIEnabled()) 
    {
        throw(MAL, "pyapi.eval",
              "Embedded Python has not been enabled. Start server with --set %s=true",
              pyapi_enableflag);
    }

    VERBOSE_MESSAGE("PyAPI Start\n");
    pycalllen = strlen(exprStr) + sizeof(argnames) + 1000;
    expr_ind_len = strlen(exprStr) + 1000;

    pycall =      GDKzalloc(pycalllen);
    expr_ind =    GDKzalloc(expr_ind_len);
    args = (str*) GDKzalloc(pci->argc * sizeof(str));
    pyreturn_values = GDKzalloc(pci->retc * sizeof(PyReturn));
    pyinput_values = GDKzalloc((pci->argc - (pci->retc + 2)) * sizeof(PyInput));

    if (args == NULL || pycall == NULL || pyreturn_values == NULL) 
    {
        throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        // TODO: free args and rcall
    }

    // first argument after the return contains the pointer to the sql_func structure
    if (sqlfun != NULL && sqlfun->ops->cnt > 0) 
    {
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
    for (i = pci->retc + 2; i < pci->argc; i++) 
    {
        if (args[i] == NULL) 
        {
            if (!seengrp && grouped) 
            {
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
    for (i = pci->retc + 2; i < pci->argc && pos < sizeof(argnames); i++) 
    {
        pos += snprintf(argnames + pos, sizeof(argnames) - pos, "%s%s", args[i], i < pci->argc - 1 ? ", " : "");
    }
    if (pos >= sizeof(argnames)) 
    {
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
                    if (exprStr[py_pos] == '\'' || exprStr[py_pos] == '"') 
                    {
                        state = INQUOTES;
                    }
                    if (exprStr[py_pos] == '\n') 
                    {
                        state = SEENNL;
                    }
                    break;

                case INQUOTES:
                    if (exprStr[py_pos] == '\\') 
                    {
                        state = ESCAPED;
                    }
                    if (exprStr[py_pos] == '\'' || exprStr[py_pos] == '"') 
                    {
                        state = NORMAL;
                    }
                    break;

                case ESCAPED:
                    state = INQUOTES;
                    break;

                case SEENNL:
                    if (exprStr[py_pos] == ' ' || exprStr[py_pos] == '\t') 
                    {
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
        for (py_pos = 0; py_pos < py_ind_pos; py_pos++) 
        {
            if (expr_ind[py_pos] == 0) 
            {
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

    //input analysis
    for (i = pci->retc + 2; i < pci->argc; i++) 
    {
        PyInput *inp = &pyinput_values[i - (pci->retc + 2)];
        if (!isaBatType(getArgType(mb,pci,i))) 
        {
            inp->scalar = true;
            inp->bat_type = getArgType(mb, pci, i);
            inp->count = 1;
            if (inp->bat_type == TYPE_str)
            {
                inp->dataptr = getArgReference_str(stk, pci, i);
            }
            else
            {
                inp->dataptr = getArgReference(stk, pci, i);
            }
        }
        else
        {
            b = BATdescriptor(*getArgReference_bat(stk, pci, i));

            inp->count = BATcount(b);
            inp->bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));
            inp->bat = b;
        }
    }

    if (mapped)
    {
#ifdef WIN32
        msg = createException(MAL, "pyapi.eval", "Please visit http://www.linux.com/directory/Distributions to download a Linux distro.\n");
        goto wrapup;
#else
        lng *pids = NULL;
        char *ptr = NULL;
        if (single_fork)
        {
            process_count = 1;
        }
        else
        {
            process_count = 8;
        }

        //create initial shared memory
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        shm_id = get_unique_shared_memory_id(1 + pci->retc * 2); //we need 1 + pci->retc * 2 shared memory spaces, the first is for the header information, the second pci->retc * 2 is one for each return BAT, and one for each return mask array    
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");

        VERBOSE_MESSAGE("Creating multiple processes.\n");

        pids = GDKzalloc(sizeof(lng) * process_count);

        memory_size = pci->retc * process_count * sizeof(ReturnBatDescr); //the memory size for the header files, each process has one per return value

        VERBOSE_MESSAGE("Initializing shared memory.\n");

        //create the shared memory for the header
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        ptr = create_shared_memory(shm_id, memory_size); 
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
        if (ptr == NULL) 
        {
            msg = createException(MAL, "pyapi.eval", "Failed to initialize shared memory");
            GDKfree(pids);
            process_id = 0;
            goto wrapup;
        }

        if (process_count > 1)
        {
            //initialize cross-process semaphore, we use two semaphores
            //the semaphores are used as follows:
            //we set the first semaphore to process_count, and the second semaphore to 0
            //every process first passes the first semaphore (decreasing the value), then tries to pass the second semaphore (which will block, because it is set to 0)
            //when the final process passes the first semaphore, it checks the value of the first semaphore (which is then equal to 0)
            //the final process will then set the value of the second semaphore to process_count, allowing all processes to pass

            //this means processes will only start returning values once all the processes are finished, this is done because we want to have one big shared memory block for each return value
            //and we can only create that block when we know how many return values there are, which we only know when all the processes have returned
            
            sem_id = create_process_semaphore(shm_id, 2);
            change_semaphore_value(sem_id, 0, process_count);
        }

        VERBOSE_MESSAGE("Waiting to fork.\n");
        //fork
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        VERBOSE_MESSAGE("Start forking.\n");
        for(i = 0; i < process_count; i++)
        {
            if ((pids[i] = fork()) < 0)
            {
                msg = createException(MAL, "pyapi.eval", "Failed to fork process");
                MT_lock_unset(&pyapiLock, "pyapi.evaluate");

                if (process_count > 1) release_process_semaphore(sem_id);
                release_shared_memory(ptr);
                GDKfree(pids);

                process_id = 0;
                goto wrapup;
            }
            else if (pids[i] == 0)
            {
                break;
            }
        }

        process_id = i + 1;
        if (i == process_count)
        {
            //main process
            int failedprocess = 0;
            int current_process = process_count;
            bool success = true;

            //wait for child processes
            MT_lock_unset(&pyapiLock, "pyapi.evaluate");
            while(current_process > 0)
            {
                int status;
                waitpid(pids[current_process - 1], &status, 0);
                if (status != 0)
                {
                    failedprocess = current_process - 1;
                    success = false;
                }
                current_process--;
            }

            if (!success)
            {
                //a child failed, get the error message from the child
                ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[failedprocess * pci->retc + 0]);

                char *err_ptr = get_shared_memory(shm_id + 1, descr->bat_size);
                if (err_ptr != NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "%s", err_ptr);
                    release_shared_memory(err_ptr);
                }
                else
                {
                    msg = createException(MAL, "pyapi.eval", "Error in child process, but no exception was thrown.");
                }

                if (process_count > 1) release_process_semaphore(sem_id);
                release_shared_memory(ptr);
                GDKfree(pids);

                process_id = 0;
                goto wrapup;
            }
            VERBOSE_MESSAGE("Finished waiting for child processes.\n");

            //collect return values
            for(i = 0; i < pci->retc; i++)
            {
                PyReturn *ret = &pyreturn_values[i];
                int total_size = 0;
                bool has_mask = false;
                ret->count = 0;
                ret->memory_size = 0;
                ret->result_type = 0;

                //first get header information 
                for(j = 0; j < process_count; j++)
                {
                    ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[j * pci->retc + i]);
                    ret->count += descr->bat_count;
                    total_size += descr->bat_size;
                    if (j > 0)
                    {
                        //if these asserts fail the processes are returning different BAT types, which shouldn't happen
                        assert(ret->memory_size == descr->element_size);
                        assert(ret->result_type == descr->npy_type);
                    }
                    ret->memory_size = descr->element_size;
                    ret->result_type = descr->npy_type;
                    has_mask = has_mask || descr->has_mask;
                }

                //get the shared memory address for this return value
                VERBOSE_MESSAGE("Parent requesting memory at id %d of size %d\n", shm_id + (i + 1), total_size);

                MT_lock_set(&pyapiLock, "pyapi.evaluate");
                ret->array_data = get_shared_memory(shm_id + (i + 1), total_size);
                MT_lock_unset(&pyapiLock, "pyapi.evaluate");

                if (ret->array_data == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Shared memory does not exist.\n");
                    if (process_count > 1) release_process_semaphore(sem_id);
                    release_shared_memory(ptr);
                    GDKfree(pids);
                    goto wrapup;
                }
                ret->mask_data = NULL;
                ret->numpy_array = NULL;
                ret->numpy_mask = NULL;
                ret->multidimensional = FALSE;
                if (has_mask)
                {
                    int mask_size = ret->count * sizeof(bool);

                    MT_lock_set(&pyapiLock, "pyapi.evaluate");
                    ret->mask_data = get_shared_memory(shm_id + pci->retc + (i + 1), mask_size);
                    MT_lock_unset(&pyapiLock, "pyapi.evaluate");

                    if (ret->mask_data == NULL)
                    {
                        msg = createException(MAL, "pyapi.eval", "Shared memory does not exist.\n");
                        if (process_count > 1) release_process_semaphore(sem_id);
                        release_shared_memory(ptr);
                        release_shared_memory(ret->array_data);
                        GDKfree(pids);
                        goto wrapup;
                    }
                }
            }
            msg = MAL_SUCCEED;
        
            if (sem_id >= 0) release_process_semaphore(sem_id);    
            if (ptr != NULL) release_shared_memory(ptr);
            if (pids != NULL) GDKfree(pids);
            process_id = 0;

            goto returnvalues;
        }
#endif
    }

    //VERBOSE_MESSAGE("Attempt to acquire GIL.\n");
    //gstate = AcquireLock(&holds_gil);


    VERBOSE_MESSAGE("Loading data from the database into Python.\n");

    // create function argument tuple, we pass a tuple of numpy arrays
    pArgs = PyTuple_New(pci->argc-(pci->retc + 2));

    // for each input column (BAT):
    for (i = pci->retc + 2; i < pci->argc; i++) {
        PyObject *vararray = NULL;
        PyInput *inp = &pyinput_values[i - (pci->retc + 2)];
        // turn scalars into one-valued BATs
        // TODO: also do this for Python? Or should scalar values be 'simple' variables?
        if (inp->scalar) 
        {
            VERBOSE_MESSAGE("- Loading a scalar of type %s (%i)\n", BatType_Format(getArgType(mb,pci,i)), getArgType(mb,pci,i));
            
            switch(inp->bat_type)
            {
                case TYPE_bit:
                    vararray = PyInt_FromLong((long)(*(bit*)inp->dataptr));
                    break;
                case TYPE_bte:
                    vararray = PyInt_FromLong((long)(*(bte*)inp->dataptr));
                    break;
                case TYPE_sht:
                    vararray = PyInt_FromLong((long)(*(sht*)inp->dataptr)); 
                    break;
                case TYPE_int:
                    vararray = PyInt_FromLong((long)(*(int*)inp->dataptr));
                    break;
                case TYPE_lng:
                    vararray = PyLong_FromLong((long)(*(lng*)inp->dataptr));
                    break;
                case TYPE_flt:
                    vararray = PyFloat_FromDouble((double)(*(flt*)inp->dataptr));
                    break;
                case TYPE_dbl:
                    vararray = PyFloat_FromDouble((double)(*(dbl*)inp->dataptr));
                    break;
#ifdef HAVE_HGE
                case TYPE_hge:
                    {
                        char hex[40];
                        const hge *t = (const hge *) inp->dataptr;
                        hge_to_string(hex, 40, *t);
                        //then we create a PyLong from that string by parsing it
                        vararray = PyLong_FromString(hex, NULL, 16);
                    }
                    break;
#endif
                case TYPE_str:
                    vararray = PyUnicode_FromString(*((char**) inp->dataptr));
                    break;
                default:
                    msg = createException(MAL, "pyapi.eval", "Unsupported scalar type %i.", inp->bat_type);
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
            int t_start = 0, t_end = 0;

            b = inp->bat;
            if (b == NULL) 
            {
                msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                goto wrapup;
            }
            t_end = inp->count;

#ifndef WIN32
            if (mapped && process_id && process_count > 1)
            {
                double chunk = process_id - 1;
                double totalchunks = process_count;
                double count = BATcount(b);
                t_start = ceil((count * chunk) / totalchunks);
                t_end = floor((count * (chunk + 1)) / totalchunks);
                if (((int)count) / 2 * 2 == (int)count) t_end--;
            }
#endif
            VERBOSE_MESSAGE("Start: %d, End: %d, Count: %d\n", t_start, t_end, t_end - t_start);

            VERBOSE_MESSAGE("- Loading a BAT of type %s (%d)\n", BatType_Format(inp->bat_type), inp->bat_type);

            //the argument is a BAT, convert it to a numpy array
            switch (inp->bat_type) {
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
                count = inp->count;
                BATloop(b, p, q)
                {
                    const char *t = (const char *) BUNtail(li, p);
                    const size_t length = utf8_strlen(t); //get the amount of UTF-8 characters in the string

                    if (length > maxsize)
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
                         //str_nil isn't a valid UTF-8 character (it's 0x80), so we can't decode it as UTF-8 (it will throw an error)
                        obj = PyUnicode_FromString("-");
                    }
                    else
                    {
                        //otherwise we can just decode the string as UTF-8
                        obj = PyUnicode_FromString(t);
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
#ifdef HAVE_HGE
            case TYPE_hge:
                li = bat_iterator(b);
                count = inp->count;

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
                fprintf(stderr, "!WARNING: Type \"hge\" (128 bit) is unsupported by Numpy. The numbers are instead converted to python objects of type \"long\". This is likely very slow.\n");
                BATloop(b, p, q)
                {
                    char hex[40];
                    //we first convert the huge to a string in hex format
                    PyObject *obj;
                    const hge *t = (const hge *) BUNtail(li, p);
                    hge_to_string(hex, 40, *t);
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
#endif
            default:
                msg = createException(MAL, "pyapi.eval", "unknown argument type ");
                goto wrapup;
            }

            VERBOSE_MESSAGE("Masked array check\n");
            // we use numpy.ma to deal with possible NULL values in the data
            // once numpy comes with proper NA support, this will change
            if (b->T->nil)
            {
                PyObject *mask;
                PyObject *mafunc = PyObject_GetAttrString(PyImport_Import(PyString_FromString("numpy.ma")), "masked_array");
                PyObject *maargs = PyTuple_New(2);
                PyArrayObject* nullmask = (PyArrayObject*) PyArray_ZEROS(1,
                                (npy_intp[1]) {(t_end - t_start)}, NPY_BOOL, 0);

                const void *nil = ATOMnilptr(b->ttype);
                int (*atomcmp)(const void *, const void *) = ATOMcompare(b->ttype);
                BATiter bi = bat_iterator(b);

                for (j = 0; j < t_end - t_start; j++) 
                {
                    if ((*atomcmp)(BUNtail(bi, BUNfirst(b) + t_start + j), nil) == 0) 
                    {
                        // Houston we have a NULL
                        PyArray_SETITEM(nullmask, PyArray_GETPTR1(nullmask, j), Py_True);
                    }
                }

                PyTuple_SetItem(maargs, 0, vararray);
                PyTuple_SetItem(maargs, 1, (PyObject*) nullmask);
                    
                mask = PyObject_CallObject(mafunc, maargs);
                if (!mask) 
                {
                    PyErr_Print();
                    msg = createException(MAL, "pyapi.eval", "UUUH");
                    goto wrapup;
                }
                Py_DECREF(vararray);
                Py_DECREF(nullmask);
                Py_DECREF(mafunc);

                vararray = mask;
            }
            PyTuple_SetItem(pArgs, ai++, vararray);

            //BBPunfix(b->batCacheid);
        }
    }


    VERBOSE_MESSAGE("Executing python code.\n");

    {
        int pyret = 0;
        PyObject *pFunc, *pModule, *str;

        str = PyString_FromString("__main__");
        pModule = PyImport_Import(str);
        Py_CLEAR(str);

        
        VERBOSE_MESSAGE("Initializing function.\n");
        pyret = PyRun_SimpleString(pycall);
        pFunc = PyObject_GetAttrString(pModule, "pyfun");
        
        if (pyret != 0 || !pModule || !pFunc || !PyCallable_Check(pFunc)) {
            PyErr_Print();
            msg = createException(MAL, "pyapi.eval", "could not parse Python code \n%s", pycall);
            goto wrapup;
        }

        pResult = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pFunc);
        Py_DECREF(pArgs);

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

            if (PyType_IsPandasDataFrame(pResult))
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

            if (PyType_IsPyScalar(pResult)) //check if the return object is a scalar
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
                if (PyType_IsNumpyMaskedArray(data))
                {
                    data = PyObject_GetAttrString(pResult, "data");   
                    if (data == NULL)
                    {
                        msg = createException(MAL, "pyapi.eval", "Invalid masked array.");
                        goto wrapup;
                    }           
                }

                if (PyType_IsNumpyArray(data)) 
                {
                    if (PyArray_NDIM((PyArrayObject*)data) != 1)
                    {
                        IsSingleArray = FALSE;
                    }
                    else
                    {
                        pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, 0));
                        IsSingleArray = PyType_IsPyScalar(pColO);
                    }
                }
                else if (PyList_Check(data)) 
                {
                    pColO = PyList_GetItem(data, 0);
                    IsSingleArray = PyType_IsPyScalar(pColO);
                }
                else if (!PyType_IsNumpyMaskedArray(data))
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
            if (!PyList_Check(pResult)) 
            {
                //check if the result is a multi-dimensional numpy array of type NPY_OBJECT
                //if the result object is a multi-dimensional numpy array of type NPY_OBJECT, we convert it to NPY_STRING because we don't know how to handle NPY_OBJECT arrays otherwise (they could contain literally anything)
                if (PyType_IsNumpyMaskedArray(pResult))
                {
                    PyObject *data, *mask;
                    data = PyObject_GetAttrString(pResult, "data");  
                    if (PyArray_NDIM((PyArrayObject*)data) != 1 && PyArray_DESCR((PyArrayObject*)data)->type_num == NPY_OBJECT)
                    {
                        //if it's a masked array we have to copy the mask along with converting the data to NPY_STRING 
                        PyObject *mafunc, *maargs;
                        PyObject *tp = PyArray_FromAny(pResult, PyArray_DescrFromType(NPY_STRING), 0, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
                        mask = PyObject_GetAttrString(pResult, "mask"); 

                        mafunc = PyObject_GetAttrString(PyImport_Import(PyString_FromString("numpy.ma")), "masked_array");
                        maargs = PyTuple_New(2);
                        PyTuple_SetItem(maargs, 0, tp);
                        PyTuple_SetItem(maargs, 1, mask);
                        mask = PyObject_CallObject(mafunc, maargs);
                        Py_DECREF(pResult);
                        Py_DECREF(mafunc);
                        pResult = mask;
                    }  
                }
                else 
                {
                    if (PyArray_NDIM((PyArrayObject*)pResult) != 1 && PyArray_DESCR((PyArrayObject*)pResult)->type_num == NPY_OBJECT)
                    {
                        //if it's not a masked array we just convert the data to NPY_STRING
                        PyObject *tp = PyArray_FromAny(pResult, PyArray_DescrFromType(NPY_STRING), 0, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
                        Py_DECREF(pResult);
                        pResult = tp;
                    }
                }
            }
            PyRun_SimpleString("del pyfun");
        }
        else
        {
            msg = createException(MAL, "pyapi.eval", "Invalid result object. No result object could be generated.");
            goto wrapup;
        }
    }

    VERBOSE_MESSAGE("Collecting return values.\n");

    for (i = 0; i < pci->retc; i++) 
    {
        PyObject *pMask = NULL;
        PyObject * pColO = NULL;
        PyReturn *ret = &pyreturn_values[i];
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
            if (PyType_IsNumpyMaskedArray(data))
            {
                data = PyObject_GetAttrString(pResult, "data");   
                pMask = PyObject_GetAttrString(pResult, "mask");    
            }


            //we can either have a multidimensional numpy array, or a single dimensional numpy array 
            if (PyArray_NDIM((PyArrayObject*)data) != 1)
            {
                //if it is a multidimensional numpy array, we have to convert the i'th dimension to a NUMPY array object
                ret->multidimensional = TRUE;
                ret->result_type = PyArray_DESCR((PyArrayObject*)data)->type_num;
            }
            else
            {
                //if it is a single dimensional numpy array, we get the i'th array from the NPYArray (this is a list)
                pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, i));
            }
        }

        if (ret->multidimensional)
        {
            ret->count = PyArray_DIMS((PyArrayObject*)pResult)[1];        
            ret->numpy_array = (PyArrayObject*)pResult;                   
            ret->numpy_mask = (PyArrayObject*)pMask;   
            ret->array_data = PyArray_DATA(ret->numpy_array);
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA(ret->numpy_mask);                 
            ret->memory_size = PyArray_DESCR(ret->numpy_array)->elsize;   
        }
        else
        {
            ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO, PyArray_DescrFromType(BatType_ToPyType(bat_type)), 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
            if (ret->numpy_array == NULL)
            {
                ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO, NULL, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
                if (ret->numpy_array == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Could not create a Numpy array from the return type.\n");
                    goto wrapup;
                }
            }
            ret->result_type = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->type_num;
            if (ret->result_type == NPY_OBJECT)
            {
                Py_DECREF(ret->numpy_array);
                ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO, PyArray_DescrFromType(NPY_STRING), 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
                ret->result_type = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->type_num;
                if (ret->numpy_array == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Could not create a Numpy array from the return type.\n");
                    goto wrapup;
                }
            }
            ret->memory_size = PyArray_DESCR(ret->numpy_array)->elsize;
            ret->count = PyArray_DIMS(ret->numpy_array)[0];
            ret->array_data = PyArray_DATA(ret->numpy_array);     
            if (PyObject_HasAttrString(pColO, "mask"))
            {
                pMask = PyObject_GetAttrString(pColO, "mask");
                if (pMask != NULL)
                {
                    ret->numpy_mask = (PyArrayObject*) PyArray_FromAny(pMask, PyArray_DescrFromType(NPY_BOOL), 1, 1,  NPY_ARRAY_CARRAY, NULL);
                    if (ret->numpy_mask == NULL || PyArray_DIMS(ret->numpy_mask)[0] != (int)ret->count)
                    {
                        pMask = NULL;
                        ret->numpy_mask = NULL;
                        /*msg = createException(MAL, "pyapi.eval", "A masked array was returned, but the mask does not have the same length as the array.");*/  
                        /*goto wrapup;*/                                  
                    }
                }
            }
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA(ret->numpy_mask); 
        }
    }

    //ReleaseLock(gstate, &holds_gil);

    VERBOSE_MESSAGE("Returning values.\n");

#ifndef WIN32
    if (mapped && process_id)
    {
        int value = 0;
        char *shm_ptr;
        ReturnBatDescr *ptr;


        VERBOSE_MESSAGE("Getting shared memory.\n");
        shm_ptr = get_shared_memory(shm_id, memory_size);
        if (shm_ptr == NULL) 
        {
            msg = createException(MAL, "pyapi.eval", "Failed to allocate shared memory for header data.\n");
            goto wrapup;
        }


        VERBOSE_MESSAGE("Writing headers.\n");
        ptr = (ReturnBatDescr*)shm_ptr;
        //return values
        //first fill in header values
        for (i = 0; i < pci->retc; i++) 
        {
            PyReturn *ret = &pyreturn_values[i];

            ReturnBatDescr *descr = &ptr[(process_id - 1) * pci->retc + i];
            descr->npy_type = ret->result_type;
            descr->element_size =   ret->memory_size;
            descr->bat_count = ret->count;
            descr->bat_size = ret->memory_size * ret->count;
            descr->has_mask = ret->mask_data != NULL;
        }

        if (process_count > 1)
        {
            VERBOSE_MESSAGE("Process %d entering the first semaphore\n", process_id);
            change_semaphore_value(sem_id, 0, -1);
            value = get_semaphore_value(sem_id, 0);
            VERBOSE_MESSAGE("Process %d waiting on semaphore, currently at value %d\n", process_id, value);
        }
        if (value == 0)
        {
            //all processes have passed the semaphore, so we can begin returning values
            //first create the shared memory space for each of the return values
            for (i = 0; i < pci->retc; i++) 
            {
                int return_size = 0;
                int mask_size = 0;
                bool has_mask = false;
                for(j = 0; j < process_count; j++)
                {
                     //for each of the processes, count the size of their return values
                     ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[j * pci->retc + i]);
                     return_size += descr->bat_size;
                     mask_size += descr->bat_count * sizeof(bool);
                     has_mask = has_mask || descr->has_mask;
                }
                //allocate the shared memory for this return value
                VERBOSE_MESSAGE("Child creating shared memory at id %d of size %d\n", shm_id + (i + 1), return_size);
                if (create_shared_memory(shm_id + (i + 1), return_size) == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Failed to allocate shared memory for returning data.\n");
                    goto wrapup;
                }
                if (has_mask)                 
                {
                    if (create_shared_memory(shm_id + pci->retc + (i + 1), mask_size)== NULL) //create a memory space for the mask
                    {
                        msg = createException(MAL, "pyapi.eval", "Failed to allocate shared memory for returning mask.\n");
                        goto wrapup;
                    }
                }
            }

            //now release all the other waiting processes so they can all begin returning values
            if (process_count > 1) change_semaphore_value(sem_id, 1, process_count);
        }

        //we wait here for all the processes to finish, so we can know the size of their return values
        if (process_count > 1) 
        {
            change_semaphore_value(sem_id, 1, -1); 

            //first check if all of the processes successfully completed, if any one of them failed we return with an error code
            for (i = 0; i < pci->retc; i++) 
            {
                for(j = 0; j < process_count; j++)
                {
                    ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[j * pci->retc + i]);
                    if (descr->npy_type < 0)
                    {
                        exit(0);
                    }
                }
            }
        }

        //now we can return the values
        for (i = 0; i < pci->retc; i++) 
        {
            char *mem_ptr;
            PyReturn *ret = &pyreturn_values[i];
            //first we compute the position where we will start writing in shared memory by looking at the processes before us
            int start_size = 0;
            int return_size = 0;
            int mask_size = 0;
            int mask_start = 0;
            bool has_mask = false;
            for(j = 0; j < process_count; j++)
            {
                ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[j * pci->retc + i]);
                if (j < (process_id - 1)) 
                {
                   start_size += descr->bat_size;
                   mask_start += descr->bat_count;
                }
                return_size += descr->bat_size;
                mask_size += descr->bat_count *sizeof(bool);

                has_mask = descr->has_mask || descr->has_mask;
            }
            //now we can copy our return values to the shared memory
            VERBOSE_MESSAGE("Process %d returning values in range %zu-%zu\n", process_id, start_size / ret->memory_size, start_size / ret->memory_size + ret->count);
            mem_ptr = get_shared_memory(shm_id + (i + 1), return_size);
            if (mem_ptr == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Failed to get pointer to shared memory for data.\n");
                goto wrapup;
            }
            memcpy(&mem_ptr[start_size], PyArray_DATA(ret->numpy_array), ret->memory_size * ret->count);

            if (has_mask)
            {
                bool *mask_ptr = (bool*)get_shared_memory(shm_id + pci->retc + (i + 1), mask_size);

                if (mask_ptr == NULL)
                {
                    msg = createException(MAL, "pyapi.eval", "Failed to get pointer to shared memory for pointer.\n");
                    goto wrapup;
                }

                if (ret->numpy_mask == NULL)
                {
                    for(iu = 0; iu < ret->count; iu++)
                    {
                        mask_ptr[mask_start + iu] = false;
                    }
                }
                else
                {
                    for(iu = 0; iu < ret->count; iu++)
                    {
                        mask_ptr[mask_start + iu] = ret->mask_data[iu];
                    }
                }
            }
        }
        exit(0);
    }
returnvalues:
#endif
    //dereference the input BATs
    for (i = pci->retc + 2; i < pci->argc; i++) 
    {
        PyInput *inp = &pyinput_values[i - (pci->retc + 2)];
        if (inp->bat != NULL) BBPunfix(inp->bat->batCacheid);
    }

    for (i = 0; i < pci->retc; i++) 
    {
        PyReturn *ret = &pyreturn_values[i];
        int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));
        size_t index_offset = 0;
        b = ret->bat_return;    

        if (ret->multidimensional) index_offset = i;

        switch (bat_type) 
        {
        case TYPE_bte:
            NP_CREATE_BAT(b, bit);
            break;
        case TYPE_sht:
            NP_CREATE_BAT(b, sht);
            break;
        case TYPE_int:
            NP_CREATE_BAT(b, int);
            break;
        case TYPE_lng:
            NP_CREATE_BAT(b, lng);
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
                char *utf8_string; 
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

                utf8_string = GDKzalloc(64 + ret->memory_size + 1); 
                utf8_string[64 + ret->memory_size] = '\0';       

                b = BATnew(TYPE_void, TYPE_str, ret->count, TRANSIENT);    
                BATseqbase(b, 0); b->T->nil = 0; b->T->nonil = 1;         
                b->tkey = 0; b->tsorted = 0; b->trevsorted = 0;
                switch(ret->result_type)                                                          
                {                                                                                 
                    case NPY_BOOL:      NP_COL_BAT_STR_LOOP(b, bit, lng_to_string); break;
                    case NPY_BYTE:      NP_COL_BAT_STR_LOOP(b, bte, lng_to_string); break;
                    case NPY_SHORT:     NP_COL_BAT_STR_LOOP(b, sht, lng_to_string); break;
                    case NPY_INT:       NP_COL_BAT_STR_LOOP(b, int, lng_to_string); break;
                    case NPY_LONG:      
                    case NPY_LONGLONG:  NP_COL_BAT_STR_LOOP(b, lng, lng_to_string); break;
                    case NPY_UBYTE:     NP_COL_BAT_STR_LOOP(b, unsigned char, lng_to_string); break;
                    case NPY_USHORT:    NP_COL_BAT_STR_LOOP(b, unsigned short, lng_to_string); break;
                    case NPY_UINT:      NP_COL_BAT_STR_LOOP(b, unsigned int, lng_to_string); break;
                    case NPY_ULONG:     NP_COL_BAT_STR_LOOP(b, unsigned long, lng_to_string); break;  
                    case NPY_ULONGLONG: NP_COL_BAT_STR_LOOP(b, unsigned long long, lng_to_string); break;  
                    case NPY_FLOAT16:                                                             
                    case NPY_FLOAT:     NP_COL_BAT_STR_LOOP(b, flt, dbl_to_string); break;             
                    case NPY_DOUBLE:                                                              
                    case NPY_LONGDOUBLE: NP_COL_BAT_STR_LOOP(b, dbl, dbl_to_string); break;                  
                    case NPY_STRING:    
                        for (iu = 0; iu < ret->count; iu++)                                        
                        {              
                            if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE)   
                            {                                                           
                                b->T->nil = 1;    
                                BUNappend(b, str_nil, FALSE);                                                            
                            }    
                            else
                            {
                                if (!string_copy(&data[(index_offset * ret->count + iu) * ret->memory_size], utf8_string, ret->memory_size))
                                {
                                    msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");       
                                    goto wrapup;    
                                }
                                BUNappend(b, utf8_string, FALSE); 
                            }                                                       
                        }    
                        break;
                    case NPY_UNICODE:    
                        for (iu = 0; iu < ret->count; iu++)                                        
                        {              
                            if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE)   
                            {                                                           
                                b->T->nil = 1;    
                                BUNappend(b, str_nil, FALSE);
                            }    
                            else
                            {
                                utf32_to_utf8(0, ret->memory_size / 4, utf8_string, (const uint32_t*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));
                                BUNappend(b, utf8_string, FALSE);
                            }                                                       
                        }    
                        break;
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
    if (mapped && process_id)
    {
        //something went wrong in a child process,

        //to prevent the other processes from stalling, we set the value of the second semaphore to process_count
        //this allows the other processes to exit
        //but first, we set the value of npy_type to -1, this informs the other processes that this process has crashed
        char *shm_ptr, *error_mem;
        ReturnBatDescr *ptr;

        shm_ptr = get_shared_memory(shm_id, memory_size);
        if (shm_ptr == NULL) goto wrapup;

        ptr = (ReturnBatDescr*)shm_ptr;

        for (i = 0; i < pci->retc; i++) 
        {
            ReturnBatDescr *descr = &ptr[(process_id - 1) * pci->retc + i];
            descr->npy_type = -1;
            descr->bat_size = strlen(msg) * sizeof(char);
        }

        error_mem = create_shared_memory(shm_id + 1, strlen(msg) * sizeof(char));
        for(iu = 0; iu < strlen(msg); iu++)
        {
            error_mem[iu] = msg[iu];
        }
        //increase the value of the semaphore
        if (process_count > 1) change_semaphore_value(sem_id, 1, process_count);
        //exit the program with an error code
        VERBOSE_MESSAGE("%s\n", msg);
        exit(1);
    }
#endif

    VERBOSE_MESSAGE("Cleaning up.\n");

    //MT_lock_unset(&pyapiLock, "pyapi.evaluate");
    for (i = 0; i < pci->retc; i++) 
    {
        PyReturn *ret = &pyreturn_values[i];
        if (!ret->multidimensional)
        {
            //AcquireLock(&holds_gil);
            if (ret->numpy_array != NULL) Py_DECREF(ret->numpy_array);                                  
            if (ret->numpy_mask != NULL) Py_DECREF(ret->numpy_mask);       
            //ReleaseLock(gstate, &holds_gil);
        }
        if (ret->numpy_array == NULL && ret->array_data != NULL)
        {
            release_shared_memory(ret->array_data);
        }
        if (ret->numpy_mask == NULL && ret->mask_data != NULL) 
        {
            release_shared_memory(ret->mask_data);
        }
    }
    if (pResult != NULL) 
    { 
        //AcquireLock(&holds_gil);
        Py_DECREF(pResult); 
        pResult = NULL;      
        //ReleaseLock(gstate, &holds_gil);
    }

    GDKfree(pyreturn_values);
    GDKfree(pyinput_values);
    for (i = 0; i < pci->argc; i++)
        if (args[i] != NULL)
            GDKfree(args[i]);
    GDKfree(args);
    GDKfree(pycall);
    GDKfree(expr_ind);

    VERBOSE_MESSAGE("Finished cleaning up.\n");
    return msg;
}

str
 PyAPIprelude(void *ret) {
    (void) ret;
    MT_lock_init(&pyapiLock, "pyapi_lock");
    MT_lock_init(&pyapiSluice, "pyapi_sluice");
    if (PyAPIEnabled()) {
        MT_lock_set(&pyapiLock, "pyapi.evaluate");
        if (!pyapiInitialized) {
            char* iar = NULL;
            Py_Initialize();
            //PyEval_InitThreads();
            import_array1(iar);
            PyRun_SimpleString("import numpy");
            //PyEval_SaveThread();
            initialize_shared_memory();
            //PyEval_ReleaseLock();
            pyapiInitialized++;
        }
        MT_lock_unset(&pyapiLock, "pyapi.evaluate");
        fprintf(stdout, "# MonetDB/Python module loaded\n");
    }
    return MAL_SUCCEED;
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
