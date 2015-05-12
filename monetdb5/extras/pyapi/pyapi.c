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

#include <string.h>

const char* pyapi_enableflag = "embedded_py";
char *NPYConstToString(int);
bool IsPyArrayObject(PyObject *);

int PyAPIEnabled(void) {
	return (GDKgetenv_istrue(pyapi_enableflag)
			|| GDKgetenv_isyes(pyapi_enableflag));
}


// TODO: exclude pyapi from mergetable, too
// TODO: add to SQL layer
// TODO: can we call the Python interpreter in a multi-thread environment?
static MT_Lock pyapiLock;
static int pyapiInitialized = FALSE;


#define BAT_TO_NP(bat, mtpe, nptpe)                                   \
		PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {BATcount(bat)},  \
        nptpe, NULL, (mtpe*) Tloc(bat, BUNfirst(bat)), 0,             \
		NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);

#define NP_TO_BAT(bat, mtpe, nptpe) {                                 \
		PyArrayObject* pCol = (PyArrayObject*) PyArray_FromAny(pColO, \
			PyArray_DescrFromType(nptpe), 1, 1, NPY_ARRAY_CARRAY |    \
			NPY_ARRAY_FORCECAST, NULL);                               \
		size_t cnt = 0;                                               \
		if (pCol == NULL)											  \
		{															  \
			pCol = (PyArrayObject*) PyArray_FromAny(pColO, NULL, 1, 1,  NPY_ARRAY_CARRAY, NULL);  \
			msg = createException(MAL, "pyapi.eval", "Wrong return type in python function. Expected an array of type \"%s\" as return value, but the python function returned an array of type \"%s\".", #mtpe, NPYConstToString(PyArray_DTYPE(pCol)->type_num));	      \
			goto wrapup;	 										  \
		}															  \
		cnt = PyArray_DIMS(pCol)[0], j;                               \
		bat = BATnew(TYPE_void, TYPE_##mtpe, cnt, TRANSIENT);         \
		BATseqbase(bat, 0); bat->T->nil = 0; bat->T->nonil = 1;       \
		bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;         \
		for (j =0; j < cnt; j++) {                                    \
			((mtpe*) Tloc(bat, BUNfirst(bat)))[j] =                   \
					*(mtpe*) PyArray_GETPTR1(pCol, j); }              \
		BATsetcount(bat, cnt); }

//todo: NULL
// TODO: also handle the case if someone returns a masked array

#define _PYAPI_DEBUG_

str PyAPIeval(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped);

str 
PyAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
	(void) cntxt;
	return PyAPIeval(mb, stk, pci, 0);
}

str 
PyAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
	(void) cntxt;
	return PyAPIeval(mb, stk, pci, 1);
}

typedef enum {
	NORMAL, SEENNL, INQUOTES, ESCAPED
} pyapi_scan_state;

str PyAPIeval(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped) {
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

	size_t count;
	size_t maxsize;
	size_t j;

	if (!PyAPIEnabled()) {
		throw(MAL, "pyapi.eval",
			  "Embedded Python has not been enabled. Start server with --set %s=true",
			  pyapi_enableflag);
	}

	pycalllen = strlen(exprStr) + sizeof(argnames) + 1000;
	expr_ind_len = strlen(exprStr) + 1000;

	pycall =      GDKzalloc(pycalllen);
	expr_ind =    GDKzalloc(expr_ind_len);
	args = (str*) GDKzalloc(sizeof(str) * pci->argc);

	if (args == NULL || pycall == NULL) {
		throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
		// TODO: free args and rcall
	}

	// this isolates our interpreter, so it's safe to run pyapi multithreaded
	// TODO: verify this
	/*tstate = Py_NewInterpreter();*/

	// first argument after the return contains the pointer to the sql_func structure
	if (sqlfun != NULL && sqlfun->ops->cnt > 0) {
		int carg = pci->retc + 2;
		argnode = sqlfun->ops->h;
		while (argnode) {
			char* argname = ((sql_arg*) argnode->data)->name;
			args[carg] = GDKstrdup(argname);
			carg++;
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

	// create function argument tuple, we pass a tuple of numpy arrays
	pArgs = PyTuple_New(pci->argc-(pci->retc + 2));

	// for each input column (BAT):
	for (i = pci->retc + 2; i < pci->argc; i++) {
		PyObject *vararray = NULL;
		// null mask for masked array

		// turn scalars into one-valued BATs
		// TODO: also do this for Python? Or should scalar values be 'simple' variables?
		if (!isaBatType(getArgType(mb,pci,i))) {
			b = BATnew(TYPE_void, getArgType(mb, pci, i), 0, TRANSIENT);
			if (b == NULL) {
				msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
				goto wrapup;
			}
			if ( getArgType(mb,pci,i) == TYPE_str)
				BUNappend(b, *getArgReference_str(stk, pci, i), FALSE);
			else
				BUNappend(b, getArgReference(stk, pci, i), FALSE);
			BATsetcount(b, 1);
			BATseqbase(b, 0);
			BATsettrivprop(b);
		} else {
			b = BATdescriptor(*getArgReference_bat(stk, pci, i));
			if (b == NULL) {
				msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
				goto wrapup;
			}
		}

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

			//create a NPY_STRING array object
			vararray = PyArray_New(
				&PyArray_Type, 
				1, 
				(npy_intp[1]) {count},  
        		NPY_STRING, 
        		NULL, 
        		NULL, 
        		maxsize,             
				0, 
				NULL);

			//fill the NPY_STRING array object using the PyArray_SETITEM function
			j = 0;
			BATloop(b, p, q)
			{
				const char *t = (const char *) BUNtail(li, p);
				PyArray_SETITEM((PyArrayObject*)vararray, PyArray_GETPTR1((PyArrayObject*)vararray, j), PyString_FromString(t));
				j++;
			}
			break;
		case TYPE_hge:
			vararray = BAT_TO_NP(b, hge, NPY_LONGLONG);
			break;

		// TODO: implement other types (boolean)
		default:
			msg = createException(MAL, "pyapi.eval", "unknown argument type ");
			goto wrapup;
		}

		// we use numpy.ma to deal with possible NULL values in the data
		// once numpy comes with proper NA support, this will change
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
	}

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
		}
		// make sure this is terminated.
		expr_ind[py_ind_pos++] = 0;
	}

	if (snprintf(pycall, pycalllen,
		 "def pyfun(%s):\n%s",
		 argnames, expr_ind) >= (int) pycalllen) {
		msg = createException(MAL, "pyapi.eval", "Command too large");
		goto wrapup;
	}
	{
		int pyret;
		PyObject *pFunc, *pModule;

		// TODO: does this create overhead?, see if we can share the import

		pModule = PyImport_Import(PyString_FromString("__main__"));
		pyret = PyRun_SimpleString(pycall);
		pFunc = PyObject_GetAttrString(pModule, "pyfun");

		//fprintf(stdout, "%s\n", pycall);
		if (pyret != 0 || !pModule || !pFunc || !PyCallable_Check(pFunc)) {
			msg = createException(MAL, "pyapi.eval", "could not parse Python code %s", pycall);
			goto wrapup;
		}

		pResult = PyObject_CallObject(pFunc, pArgs);
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

		if (!pResult || !PyList_Check(pResult) || PyList_Size(pResult) != pci->retc) 
		{
			//the object is not a PyList, but maybe it is a single PyArrayObject, so we will check if it is
			if (IsPyArrayObject(pResult))
			{
				//a single array is returned rather than a list of arrays
				//convert the single array to a list of size 1
				PyObject *list = PyList_New(1);
				PyList_SetItem(list, 0, pResult);
				pResult = list;
			}
			else
			{
				//it is neither a PyList nor a PyArrayObject, so throw an error
				msg = createException(MAL, "pyapi.eval", "Invalid result object. Need list of size %d containing numpy arrays", pci->retc);
				goto wrapup;
			}
		}
		// delete the function again
		PyRun_SimpleString("del pyfun");
	}

	// collect the return values
	for (i = 0; i < pci->retc; i++) {
		PyArrayObject *pCol = NULL;
		PyObject * pColO = PyList_GetItem(pResult, i);
		int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));

		// TODO null handling
		switch (bat_type) {
		case TYPE_bte:
			NP_TO_BAT(b, bte, NPY_INT8);
			break;
		case TYPE_sht:
			NP_TO_BAT(b, sht, NPY_INT16);
			break;
		case TYPE_int:
			NP_TO_BAT(b, int, NPY_INT32);
			break;
		case TYPE_lng:
			NP_TO_BAT(b, lng, NPY_INT64);
			break;
		case TYPE_flt:
			NP_TO_BAT(b, flt, NPY_FLOAT32);
			break;
		case TYPE_dbl:
			NP_TO_BAT(b, dbl, NPY_FLOAT64);
			break;
		case TYPE_str:
			//convert the returned column to an ArrayObject of String type
			pCol = (PyArrayObject*) PyArray_FromAny(pColO, PyArray_DescrFromType(NPY_STRING), 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);

			if (pCol == NULL)
			{
				//if the conversion isn't possible, throw an exception and finish the execution
				//we find out which actual type the array is by setting NPY_<type> to NULL, and then obtaining the type of the resulting array using PyArray_DTYPE
				pCol = (PyArrayObject*) PyArray_FromAny(pColO, NULL, 1, 1,  NPY_ARRAY_CARRAY, NULL); 
				msg = createException(MAL, "pyapi.eval", 
					"Wrong return type in python function. Expected an array of type \"%s\" as return value, but the python function returned an array of type \"%s\".", 
					"str", NPYConstToString(PyArray_DTYPE(pCol)->type_num));	      
				goto wrapup;	
			}
			//find the amount of elements in the array
			count = PyArray_DIMS(pCol)[0];

			//create and initialize a new BAT
			b = BATnew(TYPE_void, TYPE_str, count, TRANSIENT);

			BATseqbase(b, 0); 
			b->T->nil = 0; 
			b->T->nonil = 1;
			b->tkey = 0; 
			b->tsorted = 0; 
			b->trevsorted = 0;
			b->tdense = 1;

			for (j = 0; j < count; j++) 
			{
				//for every string in the array, obtain the string and append it to the BAT
				PyObject *obj = PyArray_GETITEM(pCol, PyArray_GETPTR1(pCol, j));
				BUNappend(b, PyString_AsString(PyObject_Str(obj)), FALSE);
			}              
			BATsetcount(b, count); 
			break;
		case TYPE_hge:
			NP_TO_BAT(b, hge, NPY_LONGLONG);
			break;
		default:
			msg = createException(MAL, "pyapi.eval",
								  "unknown return type for return argument %d: %d", i,
								  bat_type);
			goto wrapup;
		}

		// bat return
		if (isaBatType(getArgType(mb,pci,i))) {
			*getArgReference_bat(stk, pci, i) = b->batCacheid;
			BBPkeepref(b->batCacheid);
		} else { // single value return, only for non-grouped aggregations
			VALinit(&stk->stk[pci->argv[i]], bat_type, Tloc(b, BUNfirst(b)));
		}
		msg = MAL_SUCCEED;
	}
  wrapup:
	//MT_lock_unset(&pyapiLock, "pyapi.evaluate");
	//Py_EndInterpreter(tstate);

	GDKfree(args);
	GDKfree(pycall);
	GDKfree(expr_ind);

	return msg;
}

str PyAPIprelude(void *ret) {
	(void) ret;
	MT_lock_init(&pyapiLock, "pyapi_lock");
	if (PyAPIEnabled()) {
		MT_lock_set(&pyapiLock, "pyapi.evaluate");
		if (!pyapiInitialized) {
			char* iar = NULL;
			Py_Initialize();
			import_array1(iar);
			PyRun_SimpleString("import numpy");
			pyapiInitialized++;
		}
		MT_lock_unset(&pyapiLock, "pyapi.evaluate");
		fprintf(stdout, "# MonetDB/Python module loaded\n");
	}
	return MAL_SUCCEED;
}


char *NPYConstToString(int NPY)
{
	switch (NPY)
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

bool IsPyArrayObject(PyObject *object)
{
	PyArrayObject *arr = NULL;
	PyArray_Descr *dtype = NULL;
	int ndim = 0;
	npy_intp dims[NPY_MAXDIMS];
	if (PyArray_GetArrayParamsFromObject(object, NULL, 1, &dtype, &ndim, dims, &arr, NULL) < 0) 
	{
	    return false;
	}
	return true;
}
