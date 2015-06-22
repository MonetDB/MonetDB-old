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
const char* zerocopy_disableflag = "disable_pyzerocopy";
const char* verbose_enableflag = "enable_pyverbose";
const char* debug_enableflag = "enable_pydebug";

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
    if (var == NULL) { \
        msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL); \
        goto wrapup; \
    } \
}

#define GDK_Free(var) { \
    if (var != NULL) \
        GDKfree(var); \
}

const char * pyarg_tabwidth[] = {"TABWIDTH", "MULTIPROCESSING"};

struct _ParseArguments
{
    int tab_width;
    bool multiprocessing;
};
#define ParseArguments struct _ParseArguments

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
    int result_type;
    bool multidimensional;
};
#define PyReturn struct _PyReturn

int PyAPIEnabled(void) {
    return (GDKgetenv_istrue(pyapi_enableflag)
            || GDKgetenv_isyes(pyapi_enableflag));
}

char* FormatCode(char* code, char **args, size_t argcount, size_t tabwidth);

static MT_Lock pyapiLock;
static MT_Lock pyapiSluice;
static int pyapiInitialized = FALSE;

#define BAT_TO_NP(bat, mtpe, nptpe)                                                                     \
        PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {(t_end-t_start)},                                  \
        nptpe, NULL, &((mtpe*) Tloc(bat, BUNfirst(bat)))[t_start], 0,                                   \
        NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);

#define BAT_MMAP(bat, mtpe, batstore) {                                                                 \
        bat = BATnew(TYPE_void, TYPE_##mtpe, 0, TRANSIENT);                                             \
        BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                         \
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
        if (option_zerocopy && ret->count > 0 && TYPE_##mtpe == PyType_ToBat(ret->result_type) && (ret->count * ret->memory_size < BUN_MAX) &&                  \
            (ret->numpy_array == NULL || PyArray_FLAGS(ret->numpy_array) & NPY_ARRAY_OWNDATA))                                                                 \
        {                                                                                                                                                      \
            /*We can only create a direct map if the numpy array type and target BAT type*/                                                                    \
            /*are identical, otherwise we have to do a conversion.*/                                                                                           \
            assert(ret->array_data != NULL);                                                                                                                   \
            if (ret->numpy_array == NULL)                                                                                                                      \
            {                                                                                                                                                  \
                /*shared memory return*/                                                                                                                       \
                VERBOSE_MESSAGE("- Shared memory map!\n");                                                                                                     \
                BAT_MMAP(bat, mtpe, STORE_SHARED);                                                                                                             \
                ret->array_data = NULL;                                                                                                                        \
            }                                                                                                                                                  \
            else                                                                                                                                               \
            {                                                                                                                                                  \
                VERBOSE_MESSAGE("- Memory map!\n");                                                                                                            \
                BAT_MMAP(bat, mtpe, STORE_CMEM);                                                                                                               \
            }                                                                                                                                                  \
        }                                                                                                                                                      \
        else                                                                                                                                                   \
        {                                                                                                                                                      \
            bat = BATnew(TYPE_void, TYPE_##mtpe, ret->count, TRANSIENT);                                                                                       \
            BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                                                                      \
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
                case NPY_OBJECT:     NP_COL_BAT_LOOP_FUNC(bat, mtpe, pyobject_to_##mtpe); break;                                                                \
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
char *PyError_CreateException(char *error_text, char *pycall);

str PyAPIeval(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped, bit mapped) {
    sql_func * sqlfun = *(sql_func**) getArgReference(stk, pci, pci->retc);
    str exprStr = *getArgReference_str(stk, pci, pci->retc + 1);

    int i = 1, ai = 0;
    char* pycall = NULL;
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
    int seqbase = 0;

    bool numpy_string_array = false;
    bool option_verbose = GDKgetenv_isyes(verbose_enableflag) || GDKgetenv_istrue(verbose_enableflag);
    bool option_debug = GDKgetenv_isyes(debug_enableflag) || GDKgetenv_istrue(debug_enableflag);
    bool option_zerocopy = !(GDKgetenv_isyes(zerocopy_disableflag) || GDKgetenv_istrue(zerocopy_disableflag));
    (void) option_verbose; (void) option_debug;
#ifndef WIN32
    bool single_fork = mapped == 1;
    int shm_id = -1;
    int sem_id = -1;
    int process_id = 0;
    int memory_size = 0;
    int process_count = 0;
#endif

    size_t count;
    size_t maxsize;
    int j;
    size_t iu;

    if (!PyAPIEnabled()) {
        throw(MAL, "pyapi.eval",
              "Embedded Python has not been enabled. Start server with --set %s=true",
              pyapi_enableflag);
    }

    VERBOSE_MESSAGE("PyAPI Start\n");


    args = (str*) GDKzalloc(pci->argc * sizeof(str));
    if (args == NULL) {
        throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
    }
    pyreturn_values = GDKzalloc(pci->retc * sizeof(PyReturn));

    if (pyreturn_values == NULL) {
        GDKfree(args);
        throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
    }

    if ((pci->argc - (pci->retc + 2)) * sizeof(PyInput) > 0)
    {
        pyinput_values = GDKzalloc((pci->argc - (pci->retc + 2)) * sizeof(PyInput));

        if (pyinput_values == NULL) {
            GDKfree(args); GDKfree(pyreturn_values);
            throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        }
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
                char argbuf[64];
                snprintf(argbuf, sizeof(argbuf), "arg%i", i - pci->retc - 1);
                args[i] = GDKstrdup(argbuf);
            }
        }
    }


    VERBOSE_MESSAGE("Formatting python code.\n");

    pycall = FormatCode(exprStr, args, pci->argc, 4);
    if (pycall == NULL) {
        throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
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
            seqbase = b->H->seq;
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

    VERBOSE_MESSAGE("Loading data from the database into Python.\n");

    // Now we will do the input handling (aka converting the input BATs to numpy arrays)
    // We will put the python arrays in a PyTuple object, we will use this PyTuple object as the set of arguments to call the Python function
    pArgs = PyTuple_New(pci->argc - (pci->retc + 2));

    // Now we will loop over the input BATs and convert them to python objects
    for (i = pci->retc + 2; i < pci->argc; i++) {
        // This variable will hold the converted Python object
        PyObject *vararray = NULL; 
        // This is the header information of this input BAT, it holds a pointer to the BAT and information about the BAT
        PyInput *inp = &pyinput_values[i - (pci->retc + 2)]; 

        // There are two possibilities, either the input is a BAT, or the input is a scalar
        // If the input is a scalar we will convert it to a python scalar
        // If the input is a BAT, we will convert it to a numpy array
        if (inp->scalar) {
            VERBOSE_MESSAGE("- Loading a scalar of type %s (%i)\n", BatType_Format(getArgType(mb,pci,i)), getArgType(mb,pci,i));
            
            // The input is a scalar, so we will create a Python scalar from the input
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
                    vararray = PyLong_FromHge(*((hge *) inp->dataptr));
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
            PyTuple_SetItem(pArgs, ai++, vararray); // Add the resulting python object to the PyTuple
        }
        else
        {
            // The input is a BAT, we will convert it to a numpy array
            // t_start and t_end hold the part of the BAT we will convert to a Numpy array, by default these hold the entire BAT [0 - BATcount(b)]
            int t_start = 0, t_end = inp->count;

            b = inp->bat;
            if (b == NULL) 
            {
                // No BAT was found, we can't do anything in this case
                msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                goto wrapup;
            }

            VERBOSE_MESSAGE("- Loading a BAT of type %s (%d)\n", BatType_Format(inp->bat_type), inp->bat_type);

#ifndef WIN32
            if (mapped && process_id && process_count > 1)
            {
                // If there are multiple processes, we are responsible for dividing the input among them
                // We set t_start and t_end to the appropriate chunk for this process (depending on the process id and total amount of processes)
                double chunk = process_id - 1;
                double totalchunks = process_count;
                double count = BATcount(b);
                if (count >= process_count) {
                    t_start = ceil((count * chunk) / totalchunks);
                    t_end = floor((count * (chunk + 1)) / totalchunks);
                    if (((int)count) / 2 * 2 == (int)count) t_end--;
                    VERBOSE_MESSAGE("---Start: %d, End: %d, Count: %d\n", t_start, t_end, t_end - t_start);
                }
            }
#endif
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
                if (numpy_string_array) {
                    bool unicode = false;

                    li = bat_iterator(b);

                    //we first loop over all the strings in the BAT to find the maximum length of a single string
                    //this is because NUMPY only supports strings with a fixed maximum length
                    maxsize = 0;
                    count = inp->count;
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
                                PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
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
                                PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
                            }
                            if (j == t_end) break;
                            j++;
                        }
                    }
                }
                else {
                    bool ascii;
                    li = bat_iterator(b);
                    count = inp->count;
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
                    j = 0;
                    BATloop(b, p, q)
                    {
                        if (j >= t_start) {
                            char *t = (char *) BUNtail(li, p);
                            PyObject *obj;
                            utf8_strlen(t, &ascii);
                            if (!ascii) {
                                if (strcmp(t, str_nil) == 0) {
                                     //str_nil isn't a valid UTF-8 character (it's 0x80), so we can't decode it as UTF-8 (it will throw an error)
                                    obj = PyUnicode_FromString("-");
                                }
                                else {
                                    //otherwise we can just decode the string as UTF-8
                                    obj = PyUnicode_FromString(t);
                                }
                            } else {
                                if (strcmp(t, str_nil) == 0) {
                                     //str_nil isn't a valid UTF-8 character (it's 0x80), so we can't decode it as UTF-8 (it will throw an error)
                                    obj = PyString_FromString("-");
                                }
                                else {
                                    //otherwise we can just decode the string as UTF-8
                                    obj = PyString_FromString(t);
                                }
                            }

                            if (obj == NULL)
                            {
                                msg = createException(MAL, "pyapi.eval", "Failed to create string.");
                                goto wrapup;
                            }
                            PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
                            }
                        if (j == t_end) break;
                        j++;
                    }
                }
                break;
#ifdef HAVE_HGE
            case TYPE_hge:
            {
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
                    0,
                    0,
                    NULL);

                j = 0;
                fprintf(stderr, "!WARNING: Type \"hge\" (128 bit) is unsupported by Numpy. The numbers are instead converted to python objects of type \"long\". This is likely very slow.\n");
                BATloop(b, p, q) {
                    PyObject *obj;
                    const hge *t = (const hge *) BUNtail(li, p);
                    obj = PyLong_FromHge(*t);
                    PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), obj);
                    j++;
                }
                break;
            }
#endif
            default:
                msg = createException(MAL, "pyapi.eval", "unknown argument type ");
                goto wrapup;
            }

            // To deal with null values, we use the numpy masked array structure
            // The masked array structure is an object with two arrays of equal size, a data array and a mask array
            // The mask array is a boolean array that has the value 'True' when the element is NULL, and 'False' otherwise
            // If the BAT has Null values, we construct this masked array
            if (b->T->nil) {
                PyObject *mask;
                PyObject *mafunc = PyObject_GetAttrString(PyImport_Import(PyString_FromString("numpy.ma")), "masked_array");
                PyObject *maargs = PyTuple_New(2);
                // We will now construct the Masked array, we start by setting everything to False
                PyArrayObject* nullmask = (PyArrayObject*) PyArray_ZEROS(1,
                                (npy_intp[1]) {(t_end - t_start)}, NPY_BOOL, 0);

                // Now we will loop over the BAT, for every value that is Null we set the corresponding mask attribute to True
                const void *nil = ATOMnilptr(b->ttype);
                int (*atomcmp)(const void *, const void *) = ATOMcompare(b->ttype);
                BATiter bi = bat_iterator(b);

                for (j = 0; j < t_end - t_start; j++) {
                    if ((*atomcmp)(BUNtail(bi, BUNfirst(b) + t_start + j), nil) == 0) {
                        // Houston we have a NULL
                        PyArray_SETITEM(nullmask, PyArray_GETPTR1(nullmask, j), Py_True);
                    }
                }

                // Now we will actually construct the mask by calling the masked array constructor
                PyTuple_SetItem(maargs, 0, vararray);
                PyTuple_SetItem(maargs, 1, (PyObject*) nullmask);
                    
                mask = PyObject_CallObject(mafunc, maargs);
                if (!mask) {
                    msg = PyError_CreateException("Failed to create mask", NULL);
                    goto wrapup;
                }
                Py_DECREF(vararray);
                Py_DECREF(nullmask);
                Py_DECREF(mafunc);

                vararray = mask;
            }
            PyTuple_SetItem(pArgs, ai++, vararray); // Add the resulting python object to the PyTuple
        }
    }

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

        // The function has been successfully created/compiled, all that remains is to actually call the function
        pResult = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pFunc);
        Py_DECREF(pArgs);

        if (PyErr_Occurred()) {
            msg = PyError_CreateException("Python exception", pycall);
            goto wrapup;
        }

        // Now we need to do some error checking on the result object, because the result object has to have the correct type/size
        // We will also do some converting of result objects to a common type (such as scalar -> [[scalar]])
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
                if (pci->retc == 1)  {
                    //if we only expect a single return value, we can accept scalars by converting it into an array holding an array holding the element (i.e. [[pResult]])
                    PyObject *list = PyList_New(1);
                    PyList_SetItem(list, 0, pResult);
                    pResult = list;

                    list = PyList_New(1);
                    PyList_SetItem(list, 0, pResult);
                    pResult = list;
                }
                else {
                    //the result object is a scalar, yet we expect more than one return value. We can only convert the result into a list with a single element, so the output is necessarily wrong.
                    msg = createException(MAL, "pyapi.eval", "A single scalar was returned, yet we expect a list of %d columns. We can only convert a single scalar into a single column, thus the result is invalid.", pci->retc);
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
                else if (!PyType_IsNumpyMaskedArray(data)) {
                    //it is neither a python array, numpy array or numpy masked array, thus the result is unsupported! Throw an exception!
                    msg = createException(MAL, "pyapi.eval", "Unsupported result object. Expected either an array, a numpy array, a numpy masked array or a pandas data frame, but received an object of type \"%s\"", PyString_AsString(PyObject_Str(PyObject_Type(data))));
                    goto wrapup;
                }

                if (IsSingleArray) {
                    if (pci->retc == 1) {
                        //if we only expect a single return value, we can accept a single array by converting it into an array holding an array holding the element (i.e. [pResult])
                        PyObject *list = PyList_New(1);
                        PyList_SetItem(list, 0, pResult);
                        pResult = list;
                    }
                    else {
                        //the result object is a single array, yet we expect more than one return value. We can only convert the result into a list with a single array, so the output is necessarily wrong.
                        msg = createException(MAL, "pyapi.eval", "A single array was returned, yet we expect a list of %d columns. The result is invalid.", pci->retc);
                        goto wrapup;
                    }
                }
                else {
                    //the return value is an array of arrays, all we need to do is check if it is the correct size
                    int results = 0;
                    if (PyList_Check(data)) results = PyList_Size(data);
                    else results = PyArray_DIMS((PyArrayObject*)data)[0];
                    if (results != pci->retc) {
                        //wrong return size, we expect pci->retc arrays
                        msg = createException(MAL, "pyapi.eval", "An array of size %d was returned, yet we expect a list of %d columns. The result is invalid.", results, pci->retc);
                        goto wrapup;
                    }
                }
            }
            PyRun_SimpleString("del pyfun");
        }
        else {
            msg = createException(MAL, "pyapi.eval", "Invalid result object. No result object could be generated.");
            goto wrapup;
        }
    }

    VERBOSE_MESSAGE("Collecting return values.\n");

    // Now we have executed the Python function, we have to collect the return values and convert them to BATs
    // We will first collect header information about the Python return objects and extract the underlying C arrays
    // We will store this header information in a PyReturn object

    // The reason we are doing this as a separate step is because this preprocessing requires us to call the Python API
    // Whereas the actual returning does not require us to call the Python API
    // This means we can do the actual returning without holding the GIL
    for (i = 0; i < pci->retc; i++) {
        // Refers to the current Numpy mask (if it exists)
        PyObject *pMask = NULL;
        // Refers to the current Numpy array
        PyObject * pColO = NULL;
        // This is the PyReturn header information for the current return value, we will fill this now
        PyReturn *ret = &pyreturn_values[i];
        // This is the expected BAT result type (the type of BAT we have to make)
        int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));

        ret->multidimensional = FALSE;
        // There are three possibilities (we have ensured this right after executing the Python call)
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
            ret->numpy_array = (PyArrayObject*)pResult;                   
            ret->numpy_mask = (PyArrayObject*)pMask;   
            ret->array_data = PyArray_DATA(ret->numpy_array);
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA(ret->numpy_mask);                 
            ret->memory_size = PyArray_DESCR(ret->numpy_array)->elsize;   
        }
        else {
            // If it isn't we need to convert pColO to the expected Numpy Array type
            ret->numpy_array = NULL;
            (void) bat_type;
            //if (bat_type != TYPE_str) ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO, PyArray_DescrFromType(BatType_ToPyType(bat_type)), 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
            if (ret->numpy_array == NULL) {
                // If this conversion fails, we will set the expected type to NULL, this means it will automatically pick a type for us
                ret->numpy_array = (PyArrayObject*) PyArray_FromAny(pColO, NULL, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
                if (ret->numpy_array == NULL) {
                    msg = createException(MAL, "pyapi.eval", "Could not create a Numpy array from the return type.\n");
                    goto wrapup;
                }
            }
            ret->result_type = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->type_num; // We read the result type from the resulting array
            ret->memory_size = PyArray_DESCR(ret->numpy_array)->elsize;
            ret->count = PyArray_DIMS(ret->numpy_array)[0];
            ret->array_data = PyArray_DATA(ret->numpy_array);
            // If pColO is a Masked array, we convert the mask to a NPY_BOOL numpy array     
            if (PyObject_HasAttrString(pColO, "mask")) {
                pMask = PyObject_GetAttrString(pColO, "mask");
                if (pMask != NULL) {
                    ret->numpy_mask = (PyArrayObject*) PyArray_FromAny(pMask, PyArray_DescrFromType(NPY_BOOL), 1, 1,  NPY_ARRAY_CARRAY, NULL);
                    if (ret->numpy_mask == NULL || PyArray_DIMS(ret->numpy_mask)[0] != (int)ret->count)
                    {
                        pMask = NULL;
                        ret->numpy_mask = NULL;                            
                    }
                }
            }
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA(ret->numpy_mask); 
        }
    }

#ifndef WIN32
    // This is where the child process stops executing
    // We have successfully executed the Python function and converted the result object to a C array
    // Now all that is left is to copy the C array to share memory so the main process can read it and return it
    if (mapped && process_id) {
        int value = 0;
        char *shm_ptr;
        ReturnBatDescr *ptr;

        // First we will fill in the header information, we will need to get a pointer to the header data first
        // The main process has already created the header data for all the child processes
        VERBOSE_MESSAGE("Getting shared memory.\n");
        shm_ptr = get_shared_memory(shm_id, memory_size);
        if (shm_ptr == NULL) {
            msg = createException(MAL, "pyapi.eval", "Failed to allocate shared memory for header data.\n");
            goto wrapup;
        }

        VERBOSE_MESSAGE("Writing headers.\n");
        // Now we will write data about our result (memory size, type, number of elements) to the header
        ptr = (ReturnBatDescr*)shm_ptr;
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

        // After writing the header information, we want to write the actual C array to the shared memory
        // However, if we have multiple processes enabled, we need to wait for all the processes to complete
        // The reason for this is that we only want to copy the return values one time
        // So our plan is: 
        // Wait for all processes to complete and write their header information
        // Now by reading that header information, every process knows the total return size of all the return values combined
        // And because the processes know their process_id number and which processes are before them, they also know where to write their values
        // So after all processes have completed, we can allocate the shared memory for the return value, 
        //                            and all the processes can write their return values simultaneously
        // The way we accomplish this is by using two semaphores
        // The first semaphore was initialized exactly to process_count, so when all the processes have passed the semaphore, it has the value of 0
        if (process_count > 1)
        {
            VERBOSE_MESSAGE("Process %d entering the first semaphore\n", process_id);
            change_semaphore_value(sem_id, 0, -1);
            value = get_semaphore_value(sem_id, 0);
            VERBOSE_MESSAGE("Process %d waiting on semaphore, currently at value %d\n", process_id, value);
        }
        if (value == 0)
        {
            // So if we get here, we know all processes have finished and that we are the last process to pass the first semaphore
            // Since we are the last process, it is our job to create the shared memory for each of the return values
            for (i = 0; i < pci->retc; i++) 
            {
                int return_size = 0;
                int mask_size = 0;
                bool has_mask = false;
                // Now we will count the size of the return values for each of the processes
                for(j = 0; j < process_count; j++)
                {
                     ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[j * pci->retc + i]);
                     return_size += descr->bat_size;
                     mask_size += descr->bat_count * sizeof(bool);
                     has_mask = has_mask || descr->has_mask;
                }
                // Then we allocate the shared memory for this return value
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

            // Now all the other processes have been waiting for the last process to get here at the second semaphore
            // Since we are now here, we set the second semaphore to process_count, so all the processes can continue again
            if (process_count > 1) change_semaphore_value(sem_id, 1, process_count);
        }

        if (process_count > 1) 
        {
            // If we get here and value != 0, then not all processes have finished writing their header information
            // So we will wait on the second semaphore (which is initialized to 0, so we cannot pass until the value changes)
            change_semaphore_value(sem_id, 1, -1); 

            // Now all processes have passed the first semaphore and the header information is written
            // However, we do not know if any of the other childs have failed
            // If they have, they have written descr->npy_type to -1 in one of their headers
            // So we check for that
            for (i = 0; i < pci->retc; i++) 
            {
                for(j = 0; j < process_count; j++)
                {
                    ReturnBatDescr *descr = &(((ReturnBatDescr*)ptr)[j * pci->retc + i]);
                    if (descr->npy_type < 0)
                    {
                        // If any of the child processes have failed, exit without an error code because we did not fail
                        // The child that failed will execute with an error code and will report his error to the main process
                        exit(0);
                    }
                }
            }
        }

        // Now we can finally return the values
        for (i = 0; i < pci->retc; i++) 
        {
            char *mem_ptr;
            PyReturn *ret = &pyreturn_values[i];
            // First we compute the position where we will start writing in shared memory by looking at the processes before us
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

                // We also check if ANY of the processes returns a mask, not just if we return a mask
                has_mask = descr->has_mask || descr->has_mask;
            }
            // Now we can copy our return values to the shared memory
            VERBOSE_MESSAGE("Process %d returning values in range %zu-%zu\n", process_id, start_size / ret->memory_size, start_size / ret->memory_size + ret->count);
            mem_ptr = get_shared_memory(shm_id + (i + 1), return_size);
            if (mem_ptr == NULL)
            {
                msg = createException(MAL, "pyapi.eval", "Failed to get pointer to shared memory for data.\n");
                goto wrapup;
            }
            memcpy(&mem_ptr[start_size], PyArray_DATA(ret->numpy_array), ret->memory_size * ret->count);

            if (has_mask) {
                bool *mask_ptr = (bool*)get_shared_memory(shm_id + pci->retc + (i + 1), mask_size);
                // If any of the processes return a mask, we need to write our mask values to the shared memory

                if (mask_ptr == NULL) {
                    msg = createException(MAL, "pyapi.eval", "Failed to get pointer to shared memory for pointer.\n");
                    goto wrapup;
                }

                if (ret->numpy_mask == NULL) { 
                    // If we do not return a mask, simply write false everywhere
                    for(iu = 0; iu < ret->count; iu++) {
                        mask_ptr[mask_start + iu] = false;
                    }
                }
                else {
                    // If we do return a mask, write our mask values to the shared memory
                    for(iu = 0; iu < ret->count; iu++) {
                        mask_ptr[mask_start + iu] = ret->mask_data[iu];
                    }
                }
            }
        }
        // Exit without an error code
        exit(0);
    }
returnvalues:
#endif
    VERBOSE_MESSAGE("Returning values.\n");
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

        if (ret->multidimensional) index_offset = i;
        VERBOSE_MESSAGE("- Returning a Numpy Array of type %s of size %zu and storing it in a BAT of type %s\n", PyType_Format(ret->result_type), ret->count,  BatType_Format(bat_type));
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

                if (ret->result_type != NPY_OBJECT) {
                    utf8_string = GDKzalloc(64 + ret->memory_size + 1); 
                    utf8_string[64 + ret->memory_size] = '\0';       
                }

                b = BATnew(TYPE_void, TYPE_str, ret->count, TRANSIENT);    
                BATseqbase(b, seqbase); b->T->nil = 0; b->T->nonil = 1;         
                b->tkey = 0; b->tsorted = 0; b->trevsorted = 0;
                VERBOSE_MESSAGE("- Collecting return values of type %s.\n", PyType_Format(ret->result_type));
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
                                utf32_to_utf8(0, ret->memory_size / 4, utf8_string, (const uint32_t*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));
                                BUNappend(b, utf8_string, FALSE);
                            }                                                       
                        }    
                        break;
                    case NPY_OBJECT:
                        //The resulting array is an array of pointers to various python objects
                        //Because the python objects can be of any size, we need to allocate a different size utf8_string for every object
                        for (iu = 0; iu < ret->count; iu++) {          
                            if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {                
                                b->T->nil = 1;    
                                BUNappend(b, str_nil, FALSE);
                            } else {
                                //we try to handle as many types as possible
                                PyObject *obj = *((PyObject**) &data[(index_offset * ret->count + iu) * ret->memory_size]);
                                if (PyString_Check(obj)) {
                                    char *str = ((PyStringObject*)obj)->ob_sval;
                                    utf8_string = GDKzalloc(strlen(str) * 4);
                                    if (!string_copy(str, utf8_string, strlen(str) + 1)) {
                                        msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");       
                                        goto wrapup;    
                                    }
                                } else if (PyUnicode_Check(obj)) {
                                    uint32_t *str = (uint32_t*)((PyUnicodeObject*)obj)->str;
                                    utf8_string = GDKzalloc(((PyUnicodeObject*)obj)->length * 4);
                                    utf32_to_utf8(0, ((PyUnicodeObject*)obj)->length, utf8_string, str);
                                } else if (PyBool_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj) || PyFloat_Check(obj)) { 
#ifdef HAVE_HGE
                                    hge h;
                                    py_to_hge(obj, &h);
                                    utf8_string = GDKzalloc(64);
                                    hge_to_string(utf8_string, h);
#else
                                    lng h;
                                    py_to_lng(obj, &h);
                                    utf8_string = GDKzalloc(32);
                                    lng_to_string(utf8_string, h);
#endif
                                } else {
                                    msg = createException(MAL, "pyapi.eval", "Unrecognized Python object. Could not convert to NPY_UNICODE.\n");       
                                    goto wrapup; 
                                }
                                BUNappend(b, utf8_string, FALSE); 
                                GDKfree(utf8_string);
                            }                                                       
                        }
                        break;
                    default:
                        msg = createException(MAL, "pyapi.eval", "Unrecognized type. Could not convert to NPY_UNICODE.\n");       
                        goto wrapup;    
                }                   
                if (ret->result_type != NPY_OBJECT) {           
                    GDKfree(utf8_string);   
                }    
                                                    
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
        // If we get here, something went wrong in a child process,

        char *shm_ptr, *error_mem;
        ReturnBatDescr *ptr;

        shm_ptr = get_shared_memory(shm_id, memory_size);
        if (shm_ptr == NULL) goto wrapup;

        // To indicate that we failed, we will write information to our header
        ptr = (ReturnBatDescr*)shm_ptr;
        for (i = 0; i < pci->retc; i++) {
            ReturnBatDescr *descr = &ptr[(process_id - 1) * pci->retc + i];
            // We will write descr->npy_type to -1, so other processes can see that we failed
            descr->npy_type = -1;
            // We will write the memory size of our error message to the bat_size, so the main process can access the shared memory
            descr->bat_size = strlen(msg) * sizeof(char);
        }

        // Now create the shared memory to write our error message to
        // We can simply use the slot shm_id + 1, even though this is normally used for return values
        // This is because, if any one process fails, no values will be returned
        error_mem = create_shared_memory(shm_id + 1, strlen(msg) * sizeof(char));
        for(iu = 0; iu < strlen(msg); iu++) {
            // Copy the error message to the shared memory
            error_mem[iu] = msg[iu]; 
        }

        // To prevent the other processes from stalling, we set the value of the second semaphore to process_count
        // This allows the other processes to exit
        if (process_count > 1) change_semaphore_value(sem_id, 1, process_count);

        // Now we exit the program with an error code
        VERBOSE_MESSAGE("%s\n", msg);
        exit(1);
    }
#endif

    VERBOSE_MESSAGE("Cleaning up.\n");

    // Actual cleanup
    for (i = 0; i < pci->retc; i++) {
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

    // Now release some GDK memory we alloced for strings and input values
    GDKfree(pyreturn_values);
    GDKfree(pyinput_values);
    for (i = 0; i < pci->argc; i++)
        if (args[i] != NULL)
            GDKfree(args[i]);
    GDKfree(args);
    GDKfree(pycall);
    //GDKfree(expr_ind);
    VERBOSE_MESSAGE("%s\n", msg);

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


char *PyError_CreateException(char *error_text, char *pycall)
{
    PyObject *py_error_type, *py_error_value, *py_error_traceback;
    char *py_error_string;
    lng line_number;

    PyErr_Fetch(&py_error_type, &py_error_value, &py_error_traceback);
    if (py_error_value) {
        PyObject *error;
        PyErr_NormalizeException(&py_error_type, &py_error_value, &py_error_traceback);
        error = PyObject_Str(py_error_value);

        py_error_string = PyString_AS_STRING(error);
        Py_XDECREF(error);
        if (pycall != NULL && strlen(pycall) > 0) {
            // If pycall is given, we try to parse the line number from the error string and format pycall so it only displays the lines around the line number
            // (This code is pretty ugly, sorry)
            char line[] = "line ";
            char linenr[32]; //we only support functions with at most 10^32 lines, mostly out of philosophical reasons
            size_t i = 0, j = 0, pos = 0, nrpos = 0;

            // First parse the line number from py_error_string
            for(i = 0; i < strlen(py_error_string); i++) {
                if (pos < strlen(line)) {
                    if (py_error_string[i] == line[pos]) {
                        pos++;
                    }
                } else {
                    if (py_error_string[i] == '0' || py_error_string[i] == '1' || py_error_string[i] == '2' || 
                        py_error_string[i] == '3' || py_error_string[i] == '4' || py_error_string[i] == '5' || 
                        py_error_string[i] == '6' || py_error_string[i] == '7' || py_error_string[i] == '8' || py_error_string[i] == '9') {
                        linenr[nrpos++] = py_error_string[i];
                    }
                }
            }
            linenr[nrpos] = '\0';
            if (!str_to_lng(linenr, nrpos, &line_number)) {
                // No line number in the error, so just display a normal error
                goto finally;
            }

            // Now only display the line numbers around the error message, we display 5 lines around the error message
            {
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
                            lng_to_string(linenr, nrpos); // Convert the current line number to string and add it to lineinformation
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
                Py_XDECREF(py_error_type);
                Py_XDECREF(py_error_value);
                Py_XDECREF(py_error_traceback);
                return createException(MAL, "pyapi.eval", "%s\n%s\n%s", error_text, lineinformation, py_error_string);
            }
        }
    }
    else {
        py_error_string = "";
    }
finally:
    Py_XDECREF(py_error_type);
    Py_XDECREF(py_error_value);
    Py_XDECREF(py_error_traceback);
    if (pycall == NULL) return createException(MAL, "pyapi.eval", "%s\n%s", error_text, py_error_string);
    return createException(MAL, "pyapi.eval", "%s\n%s\n%s", error_text, pycall, py_error_string);
}


char* FormatCode(char* code, char **args, size_t argcount, size_t tabwidth)
{
    // Format the python code by fixing the indentation levels
    // We do two passes, first we get the length of the resulting formatted code and then we actually create the resulting code
    size_t i = 0, j = 0, k = 0;
    size_t length = strlen(code);
    size_t size = 0;
    size_t spaces_per_level = 2;

    size_t code_location = 0;
    char *newcode = NULL;

    size_t indentation_count = 0;
    size_t max_indentation = 100;
    // This keeps track of the different indentation levels
    // indentation_levels is a sorted array with how many spaces of indentation that specific array has
    // so indentation_levels[0] = 4 means that the first level (level 0) has 4 spaces in the source code
    // after this array is constructed we can count the amount of spaces before a statement and look in this
    // array to immediately find the indentation level of the statement
    size_t *indentation_levels = (size_t*)GDKzalloc(max_indentation * sizeof(size_t));
    // statements_per_level keeps track of how many statements are at the specified indentation level
    // this is needed to compute the size of the resulting formatted code
    // for every indentation level i, we add statements_per_level[i] * (i + 1) * spaces_per_level spaces
    size_t *statements_per_level = (size_t*)GDKzalloc(max_indentation * sizeof(size_t));

    size_t initial_spaces = 0;
    size_t statement_size = 0;
    bool seen_statement = false;
    bool multiline_statement = false;
    int multiline_quotes = 0;

    char base_start[] = "def pyfun(";
    char base_end[] = "):\n";

    if (indentation_levels == NULL || statements_per_level == NULL) goto finally;

    // Base function definition size
    // For every argument, add a comma, and add another entry for the '\0'
    size += strlen(base_start) + strlen(base_end) + argcount + 1;
    for(i = 0; i < argcount; i++) {
        if (args[i] != NULL) {
            size += strlen(args[i]) + 1; 
        }
    }
    // First remove the "{" at the start and the "};" at the end of the function, this is added when we have a function created through SQL and python doesn't like them
    // We need to be careful to only remove ones at the start/end, otherwise we might invalidate some otherwise valid python code containing them
    for(i = length - 1, j = 0; i > 0; i--)
    {
        if (code[i] != '\n' && code[i] != ' ' && code[i] != '\t' && code[i] != ';' && code[i] != '}') break;
        if (j == 0) {
            if (code[i] == ';') {
                code[i] = ' ';
                j = 1;
            }
        }
        else if (j == 1) {
            if (code[i] == '}') {
                code[i] = ' ';
                break;
            }
        }
    }
    for(i = 0; i < length; i++) {
        if (code[i] != '\n' && code[i] != ' ' && code[i] != '\t' && code[i] != '{') break;
        if (code[i] == '{') {
            code[i] = ' ';
        }
    }
    // We indent using spaces, four spaces per level
    // We also erase empty lines
    for(i = 0; i < length; i++) {
        // handle multiline strings (strings that start with """)
        if (code[i] == '\"') {
            if (!multiline_statement) {
                multiline_quotes++;
                multiline_statement = multiline_quotes == 3;
            } else {
                multiline_quotes--;
                multiline_statement = multiline_quotes != 0;
            }
        } else {
            multiline_quotes = multiline_statement ? 3 : 0;
        }

        if (!seen_statement) {
            // We have not seen a statement on this line yet
            if (code[i] == '\n'){ 
                // Empty line, skip to the next one
                initial_spaces = 0;
            } else if (code[i] == ' ') {
                initial_spaces++;
            } else if (code[i] == '\t') {
                initial_spaces += tabwidth;
            } else {
                // Statement starts here
                seen_statement = true;
            }
        }
        if (seen_statement) {
            // We have seen a statement on this line, check the indentation level
            statement_size++;

            if (code[i] == '\n' || i == length - 1) {
                // Statement ends here
                bool placed = false;
                size_t level = 0;

                if (multiline_statement) {
                    //if we are in a multiline statement, we don't want to mess with the indentation
                    size += statement_size;
                    initial_spaces = 0;
                    statement_size = 0;
                    continue;
                }
                // First put the indentation in the indentation table
                if (indentation_count >= max_indentation) {
                    // If there is no room in the indentation arrays we will extend them
                    // This probably will never happen unless in really extreme code (or if max_indentation is set very low)
                    size_t *new_indentation = GDKzalloc(2 * max_indentation * sizeof(size_t));
                    size_t *new_statements_per_level;
                    if (new_indentation == NULL) goto finally;
                    new_statements_per_level = GDKzalloc(2 * max_indentation * sizeof(size_t));
                    if (new_statements_per_level == NULL) goto finally;

                    for(i = 0; i < max_indentation; i++) {
                        new_indentation[i] = indentation_levels[i];
                        new_statements_per_level[i] = statements_per_level[i];
                    }
                    GDKfree(indentation_levels);
                    GDKfree(statements_per_level);
                    indentation_levels = new_indentation;
                    statements_per_level = new_statements_per_level;
                    max_indentation *= 2;
                }

                for(j = 0; j < indentation_count; j++) {
                    if (initial_spaces == indentation_levels[j]) {
                        // The exact space count is already in the array, so we can stop
                        level = j;
                        placed = true;
                        break;
                    }

                    if (initial_spaces < indentation_levels[j]) {
                        // The indentation level is smaller than this level (but bigger than the previous level)
                        // So the indentation level belongs here, so we move every level past this one upward one level
                        // and put the indentation level here
                        for(k = indentation_count; k > j; k--) {
                            indentation_levels[k] = indentation_levels[k - 1];
                            statements_per_level[k] = statements_per_level[k - 1];
                        }
                        indentation_count++;
                        statements_per_level[j] = 0;
                        indentation_levels[j] = initial_spaces;
                        level = j;
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    // The space count is the biggest we have seen, so we add it to the end of the array
                    level = indentation_count;
                    indentation_levels[indentation_count++] = initial_spaces;
                }
                statements_per_level[level]++;
                size += statement_size;
                seen_statement = false;
                initial_spaces = 0;
                statement_size = 0;
            }
        }
    }
    // Add the amount of spaces we will add to the size
    for(i = 0; i < indentation_count; i++) {
        size += (i + 1) * spaces_per_level * statements_per_level[i];
    }

    // Allocate space for the function
    newcode = GDKzalloc(size);
    if (newcode == NULL) goto finally;
    initial_spaces = 0;
    seen_statement = false;

    // First print in the function definition and arguments
    for(i = 0; i < strlen(base_start); i++) {
        newcode[code_location++] = base_start[i];
    }
    for(i = 0; i < argcount; i++) {
        if (args[i] != NULL) {
            for(j = 0; j < strlen(args[i]); j++) {
                newcode[code_location++] = args[i][j];
            }
            if (i != argcount - 1) {
                newcode[code_location++] = ',';
            }
        }
    }
    for(i = 0; i < strlen(base_end); i++) {
        newcode[code_location++] = base_end[i];
    }

    // Now the second pass, actually construct the code
    for(i = 0; i < length; i++) {
        //handle multiline statements
        if (code[i] == '\"') {
            if (!multiline_statement) {
                multiline_quotes++;
                multiline_statement = multiline_quotes == 3;
            } else {
                multiline_quotes--;
                multiline_statement = multiline_quotes != 0;
            }
        } else {
            multiline_quotes = multiline_statement ? 3 : 0;
        }

        if (!seen_statement) {
            if (multiline_statement) seen_statement = true; //if we are in a multiline string, we simply want to copy everything (including indentation)
            // We have not seen a statement on this line yet
            else if (code[i] == '\n'){ 
                // Empty line, skip to the next one
                initial_spaces = 0;
            } else if (code[i] == ' ') {
                initial_spaces++;
            } else if (code[i] == '\t') {
                initial_spaces += tabwidth;
            } else {
                // Statement starts here
                seen_statement = true;
                // Look through the indentation_levels array to find the level of the statement
                // from the amount of initial spaces
                bool placed = false;
                int level = 0;
                for(j = 0; j < indentation_count; j++) {
                    if (initial_spaces == indentation_levels[j]) {
                        level = j;
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    // This should never happen, because it means the initial spaces was not present in the array
                    // When we just did exactly the same loop over the array, we should have encountered this statement
                    // This means that something happened to either the indentation_levels array or something happened to the code
                    printf("WHAT HAPPENED\n");
                    goto finally;
                }
                for(j = 0; j < (level + 1) * spaces_per_level; j++) {
                    // Add spaces to the code
                    newcode[code_location++] = ' ';
                }
            }
        }
        if (seen_statement) {
            // We have seen a statement on this line, copy it
            newcode[code_location++] = code[i];
            if (code[i] == '\n') {
                // The statement has ended, move on to the next line
                seen_statement = false;
                initial_spaces = 0;
                statement_size = 0;
            }
        }
    }
    newcode[code_location] = '\0';
    if (code_location >= size) {
        // Something went wrong with our size computation, this also should never happen
        printf("WHAT HAPPENED\n");
        goto finally;
    }
finally:
    GDKfree(indentation_levels);
    GDKfree(statements_per_level);
    return newcode;
}
