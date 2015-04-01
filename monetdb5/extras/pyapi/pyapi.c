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

#include <Python.h>

// other headers
#include <string.h>

const char* pyapi_enableflag = "embedded_py";

int PyAPIEnabled(void) {
	return (GDKgetenv_istrue(pyapi_enableflag)
			|| GDKgetenv_isyes(pyapi_enableflag));
}

// TODO: can we call the Python interpreter in a multi-thread environment?
static MT_Lock pyapiLock;
static int pyapiInitialized = FALSE;


static int PyAPIinitialize(void) {
	Py_Initialize();
	pyapiInitialized++;
	return 0;
}

pyapi_export str PyAPIevalStd(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
							InstrPtr pci) {
	return PyAPIeval(cntxt, mb, stk, pci, 0);
}
pyapi_export str PyAPIevalAggr(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
							 InstrPtr pci) {
	return PyAPIeval(cntxt, mb, stk, pci, 1);
}

str PyAPIeval(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, bit grouped) {

	sql_func * sqlfun = *(sql_func**) getArgReference(stk, pci, pci->retc);
		str exprStr = *getArgReference_str(stk, pci, pci->retc + 1);

		int i = 1, ai = 0;
		char argbuf[64];
		char argnames[1000] = "";
		size_t pos;
		char* rcall = NULL;
		size_t rcalllen;
		size_t ret_rows = 0;
		//int ret_cols = 0; /* int because pci->retc is int, too*/
		str *args;
		//int evalErr;
		char *msg = MAL_SUCCEED;
		BAT *b;
		BUN cnt;
		node * argnode;
		int seengrp = FALSE;
		PyObject *pArgs; // this is going to be the parameter tuple!

		// we don't need no context, but the compiler needs us to touch it (...)
		(void) cntxt;

		if (!PyAPIEnabled()) {
			throw(MAL, "pyapi.eval",
				  "Embedded Python has not been enabled. Start server with --set %s=true",
				  pyapi_enableflag);
		}

		rcalllen = strlen(exprStr) + sizeof(argnames) + 100;
		rcall = malloc(rcalllen);
		args = (str*) GDKzalloc(sizeof(str) * pci->argc);

		if (args == NULL || rcall == NULL) {
			throw(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
			// TODO: free args and rcall
		}

		// TODO: do we need this lock for Python as well?
		MT_lock_set(&pyapiLock, "pyapi.evaluate");

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
		pArgs = PyTuple_New(pci->argc - pci->retc + 2);

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

		// for each input column (BAT):
		for (i = pci->retc + 2; i < pci->argc; i++) {
			// turn scalars into one-valued BATs
			// TODO: also do this for Python? Or should scalar values be 'simple' variables?
			if (!isaBatType(getArgType(mb,pci,i))) {
				b = BATnew(TYPE_void, getArgType(mb, pci, i), 0, TRANSIENT);
				if (b == NULL) {
					msg = createException(MAL, "rapi.eval", MAL_MALLOC_FAIL);
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

		    PyObject *varlist = PyList_New(BATcount(b));
		    size_t j;

			switch (ATOMstorage(getColumnType(getArgType(mb,pci,i)))) {
			case TYPE_int:
			//	BAT_TO_INTSXP(b, int, varvalue);
				for (j = 0; j < BATcount(b); j++) {
							int v = ((int*) Tloc(b, BUNfirst(b)))[j];
							//if ( v == int_nil)
							//	PyList_SET_ITEM(varlist, j, );
							//else
						        PyList_SET_ITEM(varlist, j, PyInt_FromLong(v));
						}
				break;
			// TODO: implement other types
			default:
				msg = createException(MAL, "pyapi.eval", "unknown argument type ");
				goto wrapup;
			}
			BBPunfix(b->batCacheid);

			PyTuple_SetItem(pArgs, ai++, varlist);
		}

		pos = 0;
		for (i = pci->retc + 2; i < pci->argc && pos < sizeof(argnames); i++) {
			pos += snprintf(argnames + pos, sizeof(argnames) - pos, "%s%s",
							args[i], i < pci->argc - 1 ? ", " : "");
		}
		if (pos >= sizeof(argnames)) {
			msg = createException(MAL, "pyapi.eval", "Command too large");
			goto wrapup;
		}
		if (snprintf(rcall, rcalllen,
					 "ret <- as.data.frame((function(%s){%s})(%s), nm=NA, stringsAsFactors=F)\n",
					 argnames, exprStr, argnames) >= (int) rcalllen) {
			msg = createException(MAL, "pyapi.eval", "Command too large");
			goto wrapup;
		}
	#ifdef _PYAPI_DEBUG_
		printf("# Python call %s\n",rcall);
	#endif

		// parse the code and create the function
		// TODO: do this in a temporary namespace, how?
		// TODO: actually include user code
		// TODO: use numpy arrays for columns!
		// TODO: How do we make nice indentation so it parses? Force user code to use level-1 indentation?
		{
			int pyret;
			PyObject *pFunc, *pModule, *pResult;

			pModule = PyImport_Import(PyString_FromString("__main__"));
			pyret = PyRun_SimpleString("def pyfun(x):\n  return list(([e+1 for e in x],1))");
			pFunc = PyObject_GetAttrString(pModule, "pyfun");

			if (pyret != 0 || !pModule || !pFunc || !PyCallable_Check(pFunc)) {
				// TODO: include parsed code
				msg = createException(MAL, "pyapi.eval", "could not parse blubb");
				goto wrapup; // shudder
			}

			// TODO: use other interface, here we can assign each value. We know how many params there will be
			// this is it
			pResult = PyObject_CallObject(pFunc, pArgs);
			if (!pResult || !PyList_Check(pResult) || !PyList_Size(pResult)) {
				msg = createException(MAL, "pyapi.eval", "invalid result object");
				goto wrapup;
			}


			// delete the function again
			PyRun_SimpleString("del pyfun");

		}

		// collect the return values
		for (i = 0; i < pci->retc; i++) {
			//SEXP ret_col = VECTOR_ELT(retval, i);
			int bat_type = ATOMstorage(getColumnType(getArgType(mb,pci,i)));
			cnt = (BUN) ret_rows;

			switch (bat_type) {
			case TYPE_int: {
				// TODO
				break;
			}
			// TODO: implement other types

			default:
				msg = createException(MAL, "pyapi.eval",
									  "unknown return type for return argument %d: %d", i,
									  bat_type);
				goto wrapup;
			}
			BATsetcount(b, cnt);

			// bat return
			if (isaBatType(getArgType(mb,pci,i))) {
				*getArgReference_bat(stk, pci, i) = b->batCacheid;
				BBPkeepref(b->batCacheid);
			} else { // single value return, only for non-grouped aggregations
				VALinit(&stk->stk[pci->argv[i]], bat_type,
						Tloc(b, BUNfirst(b)));
			}
			msg = MAL_SUCCEED;
		}
	  wrapup:
		MT_lock_unset(&pyapiLock, "pyapi.evaluate");
		GDKfree(args);

		return msg;
}

str PyAPIprelude(void *ret) {
	(void) ret;
	MT_lock_init(&pyapiLock, "pyapi_lock");

	if (PyAPIEnabled()) {
		MT_lock_set(&pyapiLock, "pyapi.evaluate");
		/* startup internal R environment  */
		if (!pyapiInitialized) {
			PyAPIinitialize();
		}
		MT_lock_unset(&pyapiLock, "pyapi.evaluate");
		fprintf(stdout, "# MonetDB/Python module loaded\n");
	}
	return MAL_SUCCEED;
}
